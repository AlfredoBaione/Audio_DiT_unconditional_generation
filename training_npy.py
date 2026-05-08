# training_npy.py
#
# Training for Audio DiT with Rectified Flow.
# No patching with Audio DiT L — every frame DAC is a token.
#
# Feature:
#   - DiT with AMP mixed precision
#   - EMA
#   - Audios generated every AUDIO_INTERVAL step on TensorBoard
#   - FD-DAC + FAD computed every METRICS_INTERVAL step
#     with reference pre-computed on all the validation set
#   - Loss train + val on TensorBoard
#
# Note: FD-DAC metric sobstitute the old KL diagonal.
#       Frechet distance with full covariance is computed on DAC latents.

import os
import copy
from pathlib import Path
from io import BytesIO

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from audio_dataset_npy import build_datasets, DAC_LATENT_DIM, DAC_SAMPLE_RATE
from network import AudioDiT, TOKEN_DIM
from metrics import (
    FADCalculator,
    evaluate_generation,
    precompute_fd_dac_reference,
)

# ======================
# CONFIG
# ======================

# Output di preprocess_dataset.py
DATASET_ROOT    = "/data/anasynth_nonbp/baione/dataset_ready/latents"
WAV_ROOT        = "/data/anasynth_nonbp/baione/dataset_ready/wav"

NORMALIZER_PATH = "/data/anasynth_nonbp/baione/checkpoints_v2/normalizer.pt"

# Cache delle statistiche reference per le metriche (calcolate una volta sola)
FD_DAC_CACHE_PATH = "/data/anasynth_nonbp/baione/checkpoints_v2/fd_dac_cache/fd_dac_ref_stats.pt"
FAD_CACHE_DIR     = "/data/anasynth_nonbp/baione/checkpoints_v2/fad_cache"

BATCH_SIZE    = 16
GRAD_ACCUM    = 1
EFFECTIVE_BS  = BATCH_SIZE * GRAD_ACCUM

NUM_STEPS     = 1000000
LR            = 1e-4
VAL_INTERVAL  = 1000       # Loss di validazione
CKPT_INTERVAL = 50000      # Checkpoint periodico
AUDIO_INTERVAL = 25000     # Genera audio su TensorBoard
METRICS_INTERVAL = 50000   # Calcola FD-DAC + FAD
N_AUDIO_SAMPLES = 4        # Audio generati ogni AUDIO_INTERVAL
N_METRICS_SAMPLES = 128     # Sample generati per FD-DAC e FAD
EULER_STEPS    = 50

LOG_DIR       = "/data/anasynth_nonbp/baione/runs/audio_rf_v2"
CKPT_DIR      = "/data/anasynth_nonbp/baione/checkpoints_v2/"

MODEL_KIND    = 'L'
DURATION_S    = 5.0

RESUME_FROM   = None 

T_MIN = 0.001
T_MAX = 0.999

EMA_DECAY     = 0.9999
EMA_START     = 5000

USE_AMP       = True


# ======================
# LR SCHEDULE
# ======================
def get_lr(step: int, warmup_steps: int = 5000) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    decay_start = int(NUM_STEPS * 0.8)
    if step < decay_start:
        return 1.0
    progress = (step - decay_start) / (NUM_STEPS - decay_start)
    return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())


# ======================
# T SAMPLING
# ======================
def sample_logit_normal(batch_size, device, mean=0.0, std=1.0):
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u).clamp(T_MIN, T_MAX)


# ======================
# EMA
# ======================
class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.lerp_(p.data, 1.0 - self.decay)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)


# ======================
# LOSS
# ======================
def compute_loss(model, batch, device):
    x1, _ = batch
    x1 = x1.to(device).float()
    B  = x1.shape[0]

    x0 = torch.randn_like(x1)
    t  = sample_logit_normal(B, device)

    t_expand = t.view(B, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0

    with torch.amp.autocast('cuda', enabled=USE_AMP):
        pred = model(xt, t)
        loss = F.mse_loss(pred, target)

    return loss


# ======================
# AUDIO/SPECTROGRAM UTILITIES
# ======================

def plot_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    import PIL.Image as Image
    import torchvision
    img = torchvision.transforms.ToTensor()(Image.open(buf))
    buf.close()
    return img


def make_spectrogram(waveform, sr, title=""):
    import torchaudio
    spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512
    )(waveform.cpu().float())
    spec_db = torchaudio.transforms.AmplitudeToDB()(spec)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(spec_db[0].numpy(), aspect='auto', origin='lower',
              cmap='viridis', vmin=-80, vmax=0)
    ax.set_title(title)
    ax.set_xlabel("Frame"); ax.set_ylabel("Mel Bin")
    plt.colorbar(ax.images[0], ax=ax, label="dB")
    img = plot_to_image(fig)
    plt.close(fig)
    return img


@torch.no_grad()
def euler_sample(model, n_frames, device, steps=EULER_STEPS):
    model.eval()
    x = torch.randn(1, n_frames, TOKEN_DIM, device=device)
    dt = (T_MAX - T_MIN) / steps
    for i in range(steps):
        t_val = T_MIN + i * dt
        t = torch.ones(1, device=device) * t_val
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            v = model(x, t)
        x = x + v.float() * dt
    return x[0].cpu()


@torch.no_grad()
def generate_and_log_audio(
    model, normalizer, n_frames, step, writer, device,
    output_dir, n_samples=N_AUDIO_SAMPLES, prefix="EMA",
):
    """Genera audio, decodifica con DAC su CPU, logga su TensorBoard."""
    generated_frames = []
    for i in range(n_samples):
        gen = euler_sample(model, n_frames, device)
        generated_frames.append(gen)

    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu")
    dac_model.eval()

    for i, gen in enumerate(generated_frames):
        if not torch.isfinite(gen).all():
            continue

        z = gen.T
        z = normalizer.denormalize(z)
        z_in = z.unsqueeze(0).float()
        waveform = dac_model.decode(z_in).squeeze(0)

        wn = waveform / (waveform.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/{prefix}/step{step:07d}_sample_{i:02d}", wn,
            global_step=step, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(
            waveform, DAC_SAMPLE_RATE,
            f"{prefix} sample {i} — step {step}",
        )
        writer.add_image(
            f"Spectrogram/{prefix}/step{step:07d}_sample_{i:02d}",
            spec_img, global_step=step,
        )
        wav_path = os.path.join(output_dir, f"step{step:07d}_{prefix}_{i:02d}.wav")
        sf.write(wav_path, waveform.squeeze().numpy(), DAC_SAMPLE_RATE)

    del dac_model


@torch.no_grad()
def log_real_audio_samples(dataset, normalizer, writer, n_samples=N_AUDIO_SAMPLES):
    """Logga audio reali dal dataset per confronto su TensorBoard."""
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu")
    dac_model.eval()

    total = len(dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()

    for i, idx in enumerate(indices):
        frames, _ = dataset[idx]
        z = frames.T
        z = normalizer.denormalize(z)
        z_in = z.unsqueeze(0).float()
        waveform = dac_model.decode(z_in).squeeze(0)

        wn = waveform / (waveform.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/Real/sample_{i:02d}", wn,
            global_step=0, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(waveform, DAC_SAMPLE_RATE, f"Real sample {i}")
        writer.add_image(
            f"Spectrogram/Real/sample_{i:02d}",
            spec_img, global_step=0,
        )

    del dac_model
    print(f"  {n_samples} audio reali loggati su TensorBoard")


# ======================
# METRICS EVALUATION
# ======================

@torch.no_grad()
def evaluate_and_log_metrics(
    model, normalizer, val_dataset, step, writer, device, output_dir,
    fad_calculator, fd_dac_ref_stats, n_samples=N_METRICS_SAMPLES,
):
    """
    Genera N sample, calcola FD-DAC + FAD contro reference pre-calcolato,
    e logga tutto su TensorBoard.
    """
    print(f"\n  Calcolo metriche: {n_samples} sample generati "
          f"vs reference ({fad_calculator.ref_n_samples} sample)...")

    results = evaluate_generation(
        model=model,
        normalizer=normalizer,
        val_dataset=val_dataset,
        n_samples=n_samples,
        euler_steps=EULER_STEPS,
        device=device,
        use_amp=USE_AMP,
        fad_calculator=fad_calculator,
        fd_dac_ref_stats=fd_dac_ref_stats,
    )

    fd_dac = results["fd_dac"]
    fad    = results["fad"]

    writer.add_scalar("Metrics/FD_DAC", fd_dac, step)
    writer.add_scalar("Metrics/FAD", fad, step)

    print(f"  FD-DAC: {fd_dac:.4f} | FAD: {fad:.4f}")

    # Logga alcuni audio generati
    for i, wav in enumerate(results["generated_wavs"][:4]):
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
        wn = wav / (wav.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/Metrics_Generated/step{step:07d}_sample_{i:02d}", wn,
            global_step=step, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(
            wav, DAC_SAMPLE_RATE,
            f"Metrics Gen {i} — step {step} — FD-DAC={fd_dac:.2f} FAD={fad:.2f}",
        )
        writer.add_image(
            f"Spectrogram/Metrics_Generated/step{step:07d}_sample_{i:02d}",
            spec_img, global_step=step,
        )

    # Salva audio su disco
    for i, wav in enumerate(results["generated_wavs"][:4]):
        path = os.path.join(output_dir, f"metrics_step{step:07d}_gen_{i:02d}.wav")
        wav_np = wav.numpy() if wav.dim() == 1 else wav.squeeze().numpy()
        sf.write(path, wav_np, DAC_SAMPLE_RATE)

    return fd_dac, fad


# ======================
# DATALOADER
# ======================

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


# ======================
# MAIN
# ======================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")

    # ======================
    # DATA
    # ======================
    print("\nCaricamento dataset...")
    train_dataset, val_dataset, normalizer, label_map = build_datasets(
        root_dir=DATASET_ROOT,
        duration_s=DURATION_S,
        normalizer_path=NORMALIZER_PATH if os.path.exists(NORMALIZER_PATH) else None,
        preload=False,
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.join(CKPT_DIR, "audio"), exist_ok=True)

    # Salva il normalizer solo se non esiste gia (evita riscrittura inutile)
    if not os.path.exists(f"{CKPT_DIR}/normalizer.pt"):
        normalizer.save(f"{CKPT_DIR}/normalizer.pt")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
        drop_last=False,
    )

    train_iter = infinite_loader(train_loader)
    val_iter   = infinite_loader(val_loader)

    # ======================
    # PRE-COMPUTAION REFERENCE STATS (only one time)
    # ======================
    print("\nPre-calcolo statistiche reference per metriche...")

    # FD-DAC: mu e sigma (covarianza piena) sui latenti DAC del val set
    fd_dac_ref_stats = precompute_fd_dac_reference(
        val_dataset,
        cache_path=FD_DAC_CACHE_PATH,
    )

    # FAD: embedding Encodec di tutti i WAV di validazione
    fad_calculator = FADCalculator(device="cuda")
    fad_calculator.precompute_reference_stats(
        val_dataset=val_dataset,
        normalizer=normalizer,
        wav_root=WAV_ROOT,
        latent_root=DATASET_ROOT,
        sr=DAC_SAMPLE_RATE,
        cache_dir=FAD_CACHE_DIR,
    )

    print(f"Reference stats pronte: "
          f"FD-DAC su {len(val_dataset)} sample latenti, "
          f"FAD su {fad_calculator.ref_n_samples} sample audio\n")

    # ======================
    # MODEL + EMA
    # ======================
    model     = AudioDiT(kind=MODEL_KIND).to(device)
    ema       = EMAModel(model, decay=EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    writer    = SummaryWriter(LOG_DIR)

    best_val_loss = float("inf")
    start_step    = 0

    # ======================
    # RESUME
    # ======================
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Riprendendo training da: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scheduler.last_epoch = ckpt["step"]
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        else:
            ema = EMAModel(model, decay=EMA_DECAY)
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step    = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  → Step {start_step} | best_val_loss: {best_val_loss:.6f}")
    else:
        print("Training da zero.")

    # ======================
    # INFO
    # ======================
    n_frames = train_dataset.n_frames
    print(f"\n{'='*60}")
    print(f"Training su {device} | AudioDiT-{MODEL_KIND}")
    print(f"Steps: {NUM_STEPS} | Batch: {EFFECTIVE_BS}")
    print(f"LR: {LR} | EMA decay: {EMA_DECAY} | AMP: {USE_AMP}")
    print(f"Sequenza: {n_frames} frame = {n_frames} token di dim {TOKEN_DIM}")
    print(f"Train: {len(train_dataset)} chunk | Val: {len(val_dataset)} chunk")
    print(f"Audio ogni {AUDIO_INTERVAL} step | Metriche ogni {METRICS_INTERVAL} step")
    print(f"Metriche: {N_METRICS_SAMPLES} generati vs {fad_calculator.ref_n_samples} reference")
    print(f"DATASET_ROOT: {DATASET_ROOT}")
    print(f"WAV_ROOT: {WAV_ROOT}")
    print(f"{'='*60}\n")

    # ======================
    # LOGGA AUDIO REALI
    # ======================
    print("Log audio reali su TensorBoard...")
    log_real_audio_samples(val_dataset, normalizer, writer)

    # ======================
    # TRAIN LOOP
    # ======================
    val_loss = None
    audio_dir = os.path.join(CKPT_DIR, "audio")

    pbar = tqdm(range(start_step, NUM_STEPS), initial=start_step, total=NUM_STEPS,
                desc="Training", unit="step")

    last_step = start_step

    try:
        for step in pbar:
            last_step = step

            model.train()

            accum_loss = 0.0
            for _ in range(GRAD_ACCUM):
                batch = next(train_iter)
                loss  = compute_loss(model, batch, device) / GRAD_ACCUM
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if step >= EMA_START:
                ema.update(model)

            writer.add_scalar("Loss/train", accum_loss, step)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], step)

            pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")

            # ======================
            # VALIDAZIONE (loss)
            # ======================
            if step % VAL_INTERVAL == 0:
                model.eval()
                n_val = 20
                with torch.no_grad():
                    val_losses = []
                    for _ in range(n_val):
                        vb = next(val_iter)
                        vl = compute_loss(model, vb, device).item()
                        val_losses.append(vl)
                    val_loss = sum(val_losses) / len(val_losses)

                    ema_val_loss = val_loss
                    if step >= EMA_START:
                        ema_vl = []
                        for _ in range(n_val):
                            vb = next(val_iter)
                            evl = compute_loss(ema.model, vb, device).item()
                            ema_vl.append(evl)
                        ema_val_loss = sum(ema_vl) / len(ema_vl)
                        writer.add_scalar("Loss/val_ema", ema_val_loss, step)

                writer.add_scalar("Loss/val", val_loss, step)

                ema_str = f" | EMA Val {ema_val_loss:.6f}" if step >= EMA_START else ""
                pbar.write(f"Step {step:7d} | Train {accum_loss:.6f} | "
                           f"Val {val_loss:.6f}{ema_str} | "
                           f"LR {scheduler.get_last_lr()[0]:.2e}")

                # Best model
                check_loss = ema_val_loss if step >= EMA_START else val_loss
                if check_loss < best_val_loss:
                    best_val_loss = check_loss
                    save_path = f"{CKPT_DIR}/best_model_step{step}.pt"
                    torch.save({
                        "model_state_dict":     model.state_dict(),
                        "ema_state_dict":       ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict":    scaler.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "model_kind": MODEL_KIND,
                        "label_map": label_map,
                        "n_frames": n_frames,
                    }, save_path)
                    for old in Path(CKPT_DIR).glob("best_model_step*.pt"):
                        if old.resolve() != Path(save_path).resolve():
                            old.unlink()
                    pbar.write(f"  → Best model: {save_path}")

            # ======================
            # GENERA AUDIO
            # ======================
            if step > 0 and step % AUDIO_INTERVAL == 0:
                pbar.write(f"\n  Generazione audio step {step}...")

                gen_model = ema.model if step >= EMA_START else model

                generate_and_log_audio(
                    model=gen_model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer,
                    device=device, output_dir=audio_dir,
                    n_samples=N_AUDIO_SAMPLES,
                    prefix="EMA" if step >= EMA_START else "Model",
                )

                pbar.write(f"  Audio loggati (step {step})\n")
                model.train()

            # ======================
            # METRICHE (FD-DAC + FAD)
            # ======================
            if step > 0 and step % METRICS_INTERVAL == 0:
                gen_model = ema.model if step >= EMA_START else model

                fd_dac, fad = evaluate_and_log_metrics(
                    model=gen_model,
                    normalizer=normalizer,
                    val_dataset=val_dataset,
                    step=step,
                    writer=writer,
                    device=device,
                    output_dir=audio_dir,
                    fad_calculator=fad_calculator,
                    fd_dac_ref_stats=fd_dac_ref_stats,
                    n_samples=N_METRICS_SAMPLES,
                )

                pbar.write(f"  Metriche: FD-DAC={fd_dac:.4f} | FAD={fad:.4f}\n")
                model.train()

            # ======================
            # CHECKPOINT PERIODICO
            # ======================
            if step % CKPT_INTERVAL == 0 and step > 0:
                p = f"{CKPT_DIR}/checkpoint_step{step}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "model_kind": MODEL_KIND,
                    "label_map": label_map,
                    "n_frames": n_frames,
                }, p)
                pbar.write(f"  → Checkpoint: {p}")

    finally:
        # Salva SEMPRE l'ultimo checkpoint: a fine training, dopo Ctrl+C, o errore
        last_path = f"{CKPT_DIR}/checkpoint_last_step{last_step}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": last_step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "model_kind": MODEL_KIND,
            "label_map": label_map,
            "n_frames": n_frames,
        }, last_path)
        print(f"\n  → Ultimo checkpoint salvato: {last_path}")
        pbar.close()
        writer.close()
        print("Training terminato.")
