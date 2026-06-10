# training_npy_adv.py
#
# Training con MSE + Adversarial loss per Audio DiT.
#
# Struttura:
#   - Flow matching standard con MSE sulla velocity (come prima)
#   - PIU un discriminatore che distingue x1_real da x1_pred
#   - Il modello flow matching viene addestrato a foolare il discriminatore
#
# Differenze rispetto a training_npy.py:
#   - Aggiunto LatentDiscriminator
#   - Loss del generatore: MSE + lambda_adv * G_loss
#   - Training alternato: step generatore poi step discriminatore
#   - R1 regularization periodica per stabilita
#   - Warmup del lambda_adv (parte da 0, cresce gradualmente)
#
# Nota su val loss:
#   - Best model selection basato su MSE val (segnale stabile)
#   - val_adv loggata separatamente come DIAGNOSTICA (oscilla per natura GAN)

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
from discriminator import (
    LatentDiscriminator,
    discriminator_loss,
    generator_adversarial_loss,
    r1_regularization,
)
from metrics import (
    FADCalculator,
    evaluate_generation,
    precompute_fd_dac_reference,
)


# ======================
# CONFIG
# ======================

DATASET_ROOT    = "./dataset_ready/latents"
WAV_ROOT        = "./dataset_ready/wav"
NORMALIZER_PATH = "./checkpoints_v2/normalizer.pt"

# Cache delle statistiche reference per le metriche (calcolate una volta sola)
FD_DAC_CACHE_PATH = "./checkpoints_v2/fd_dac_cache/fd_dac_ref_stats.pt"
FAD_CACHE_DIR     = "./checkpoints_v2/fad_cache"

BATCH_SIZE    = 8
GRAD_ACCUM    = 1
NUM_STEPS     = 500000
LR            = 1e-4
LR_D          = 1e-4          # LR discriminatore (spesso uguale o leggermente inferiore)
VAL_INTERVAL  = 1000
CKPT_INTERVAL = 10000
AUDIO_INTERVAL = 10000
METRICS_INTERVAL = 50000
N_AUDIO_SAMPLES = 2
N_METRICS_SAMPLES = 64
EULER_STEPS    = 50

LOG_DIR       = "runs/audio_rf_adv"
CKPT_DIR      = "checkpoints_adv"

MODEL_KIND    = 'B'
DISC_KIND     = 'M'           # discriminator 'S', 'M', 'L'
DURATION_S    = 5.0

# Resume: se vuoi partire da un checkpoint MSE-only, metti il path qui
# Il discriminator parte sempre da zero, il modello flow matching no
RESUME_MSE_ONLY_FROM = None    # es. "checkpoints_v2/checkpoint_step200000.pt"
RESUME_FROM = None             # per riprendere training adversarial

T_MIN = 0.001
T_MAX = 0.999
EMA_DECAY     = 0.9999
EMA_START     = 5000
USE_AMP       = True

# ======================
# CONFIG ADVERSARIAL
# ======================
LAMBDA_ADV_TARGET = 0.1         # peso finale dell'adversarial loss
ADV_WARMUP_STEPS  = 10000       # step per raggiungere LAMBDA_ADV_TARGET da 0
ADV_START_STEP    = 5000        # non applicare adv prima di questo step
                                 # (lascia che MSE si stabilizzi)
R1_GAMMA          = 10.0        # peso del R1 penalty
R1_INTERVAL       = 16          # applica R1 ogni N step (per velocita)
DISC_UPDATE_RATIO = 1           # N step D per ogni step G (1 = alternato)


# ======================
# SCHEDULE
# ======================
def get_lr(step, warmup_steps=5000):
    if step < warmup_steps:
        return step / warmup_steps
    ds = int(NUM_STEPS * 0.8)
    if step < ds:
        return 1.0
    p = (step - ds) / (NUM_STEPS - ds)
    return 0.5 * (1 + torch.cos(torch.tensor(p * 3.14159)).item())


def get_lambda_adv(step):
    """Warmup dell'adversarial loss."""
    if step < ADV_START_STEP:
        return 0.0
    progress = min(1.0, (step - ADV_START_STEP) / ADV_WARMUP_STEPS)
    return LAMBDA_ADV_TARGET * progress

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
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.lerp_(p.data, 1.0 - self.decay)

    def state_dict(self): return self.model.state_dict()
    def load_state_dict(self, sd): self.model.load_state_dict(sd)


# ======================
# GENERATOR LOSS: MSE + ADVERSARIAL
# ======================

def compute_generator_loss(model, discriminator, batch, device,
                            lambda_adv, step):
    """
    Loss del generatore (flow matching model):
        MSE(v_pred, v_true) + lambda_adv * G_adv(D(x1_pred))

    x1_pred viene ricostruito dalla velocity predetta:
        Dal flow matching: v_true = x1 - x0 e xt = (1-t)*x0 + t*x1
        Quindi: x1_pred = xt + (1-t) * v_pred
                         = (1-t)*x0 + t*x1 + (1-t)*(x1-x0)
                         = x1 (se v_pred e corretta)
    """
    x1, _ = batch
    x1 = x1.to(device).float()
    B = x1.shape[0]

    x0 = torch.randn_like(x1)
    t = sample_logit_normal(B, device)
    t_expand = t.view(B, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0

    with torch.amp.autocast('cuda', enabled=USE_AMP):
        v_pred = model(xt, t)
        mse = F.mse_loss(v_pred, target)

        if lambda_adv > 0:
            # Ricostruisci x1 dalla velocity predetta
            # x1_pred = xt + (1-t) * v_pred
            # (questa formula ha senso per t piccoli; per t grandi e meglio
            # usare direttamente x1_pred = xt/t + (1 - 1/t)*x0 ma richiede
            # dividere per t che puo essere piccolo)
            # Usiamo formulazione robusta:
            x1_pred = xt + (1 - t_expand) * v_pred

            # Discriminatore dice se x1_pred sembra reale
            d_fake = discriminator(x1_pred)
            adv = generator_adversarial_loss(d_fake)

            total = mse + lambda_adv * adv
            return total, {"mse": mse.item(), "adv": adv.item(),
                           "lambda_adv": lambda_adv}

    return mse, {"mse": mse.item(), "adv": 0.0, "lambda_adv": 0.0}


# ======================
# DISCRIMINATOR LOSS
# ======================

def compute_discriminator_loss(discriminator, model, batch, device,
                                 apply_r1: bool):
    """
    Loss del discriminatore: separa x1 reali da x1_pred generati.
    Opzionalmente applica R1 penalty sui sample reali.
    """
    x1, _ = batch
    x1 = x1.to(device).float()
    B = x1.shape[0]

    # Genera x1_pred con il modello (congelato per questo step)
    with torch.no_grad():
        x0 = torch.randn_like(x1)
        t = sample_logit_normal(B, device)
        t_expand = t.view(B, 1, 1)
        xt = (1 - t_expand) * x0 + t_expand * x1

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            v_pred = model(xt, t)
        x1_pred = xt + (1 - t_expand) * v_pred.float()

    # D su reali e fake
    if apply_r1:
        x1_real_input = x1.detach().clone().requires_grad_(True)
    else:
        x1_real_input = x1

    d_real = discriminator(x1_real_input)
    d_fake = discriminator(x1_pred.detach())

    d_loss = discriminator_loss(d_real, d_fake)

    info = {
        "d_loss": d_loss.item(),
        "d_real_mean": d_real.mean().item(),
        "d_fake_mean": d_fake.mean().item(),
    }

    if apply_r1:
        r1 = r1_regularization(d_real, x1_real_input)
        total = d_loss + (R1_GAMMA / 2) * r1
        info["r1"] = r1.item()
    else:
        total = d_loss

    return total, info


# ======================
# VALIDATION LOSS (MSE + adv diagnostic)
# ======================

@torch.no_grad()
def compute_val_losses(model, discriminator, batch, device, lambda_adv):
    """
    Calcola MSE val (per best model selection) e adv val (solo diagnostica).
    Il discriminatore NON viene aggiornato qui (eval mode + no_grad).
    """
    x1, _ = batch
    x1 = x1.to(device).float()
    B = x1.shape[0]

    x0 = torch.randn_like(x1)
    t = sample_logit_normal(B, device)
    t_expand = t.view(B, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0

    with torch.amp.autocast('cuda', enabled=USE_AMP):
        v_pred = model(xt, t)
        mse = F.mse_loss(v_pred, target).item()

    # Adv diagnostic: solo se il discriminator e attivo
    adv = 0.0
    if lambda_adv > 0:
        x1_pred = xt + (1 - t_expand) * v_pred.float()
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            d_fake = discriminator(x1_pred)
            adv = generator_adversarial_loss(d_fake).item()

    return mse, adv


# ======================
# UTILITIES audio/spectrogram (invariate)
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
        sample_rate=sr, n_mels=128, n_fft=2048, hop_length=512,
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
def generate_and_log_audio(model, normalizer, n_frames, step, writer, device,
                             output_dir, n_samples=N_AUDIO_SAMPLES, prefix="EMA"):
    generated = [euler_sample(model, n_frames, device) for _ in range(n_samples)]

    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu"); dac_model.eval()

    for i, gen in enumerate(generated):
        if not torch.isfinite(gen).all():
            continue
        z = normalizer.denormalize(gen.T)
        waveform = dac_model.decode(z.unsqueeze(0).float()).squeeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/{prefix}/step{step:07d}_sample_{i:02d}", wn,
            global_step=step, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(waveform, DAC_SAMPLE_RATE,
                                     f"{prefix} sample {i} — step {step}")
        writer.add_image(
            f"Spectrogram/{prefix}/step{step:07d}_sample_{i:02d}",
            spec_img, global_step=step,
        )
        sf.write(os.path.join(output_dir, f"step{step:07d}_{prefix}_{i:02d}.wav"),
                 waveform.squeeze().numpy(), DAC_SAMPLE_RATE)

    del dac_model


@torch.no_grad()
def log_real_audio_samples(dataset, normalizer, writer, n_samples=N_AUDIO_SAMPLES):
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu"); dac_model.eval()

    total = len(dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()

    for i, idx in enumerate(indices):
        frames, _ = dataset[idx]
        z = normalizer.denormalize(frames.T)
        waveform = dac_model.decode(z.unsqueeze(0).float()).squeeze(0)
        wn = waveform / (waveform.abs().max() + 1e-8)
        writer.add_audio(f"Audio/Real/sample_{i:02d}", wn,
                          global_step=0, sample_rate=DAC_SAMPLE_RATE)
        spec_img = make_spectrogram(waveform, DAC_SAMPLE_RATE, f"Real sample {i}")
        writer.add_image(f"Spectrogram/Real/sample_{i:02d}", spec_img, global_step=0)

    del dac_model
    print(f"  {n_samples} audio reali loggati su TensorBoard")

# ======================
# METRICS EVALUATION
# ======================

@torch.no_grad()
def evaluate_and_log_metrics(model, normalizer, val_dataset, step, writer,
                               device, output_dir, fad_calculator, fd_dac_ref_stats,
                               n_samples=N_METRICS_SAMPLES):
    print(f"\n  Metriche su {n_samples} sample vs {fad_calculator.ref_n_samples} ref...")

    results = evaluate_generation(
        model=model, normalizer=normalizer, val_dataset=val_dataset,
        n_samples=n_samples, euler_steps=EULER_STEPS, device=device,
        use_amp=USE_AMP, fad_calculator=fad_calculator, fd_dac_ref_stats=fd_dac_ref_stats,
    )
    fd_dac, fad = results["fd_dac"], results["fad"]

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
        for b in loader:
            yield b


# ======================
# MAIN
# ======================

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # Dataset
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
        num_workers=0, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"), drop_last=False,
    )

    train_iter = infinite_loader(train_loader)
    val_iter = infinite_loader(val_loader)

    # Reference stats per metriche
    print("\nPre-calcolo reference...")
    fd_dac_ref_stats = precompute_fd_dac_reference(
        val_dataset,
        cache_path=FD_DAC_CACHE_PATH,
    )
    fad_calculator = FADCalculator(device="cuda")
    fad_calculator.precompute_reference_stats(
        val_dataset=val_dataset, normalizer=normalizer,
        wav_root=WAV_ROOT, latent_root=DATASET_ROOT, sr=DAC_SAMPLE_RATE,
        cache_dir=FAD_CACHE_DIR,
    )

    print(f"Reference stats pronte: "
          f"FD-DAC su {len(val_dataset)} sample latenti, "
          f"FAD su {fad_calculator.ref_n_samples} sample audio\n")

    # ======================
    # MODEL + EMA + DISCRIMINATOR + OPTIMIZER
    # ======================
    print("\nInizializzazione modelli...")
    model = AudioDiT(kind=MODEL_KIND).to(device)
    ema = EMAModel(model, decay=EMA_DECAY)

    # Discriminator
    discriminator = LatentDiscriminator(kind=DISC_KIND).to(device)

    # Optimizers separati
    opt_g = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2,
                              betas=(0.9, 0.99))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=LR_D, weight_decay=0,
                              betas=(0.5, 0.9))  # betas GAN-friendly

    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, get_lr)
    sched_d = torch.optim.lr_scheduler.LambdaLR(opt_d, get_lr)

    scaler_g = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    scaler_d = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    writer = SummaryWriter(LOG_DIR)

    best_val_loss = float("inf")
    start_step = 0

    # Resume options
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Resume completo da: {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "discriminator_state_dict" in ckpt:
            discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        opt_g.load_state_dict(ckpt["opt_g_state_dict"])
        if "opt_d_state_dict" in ckpt:
            opt_d.load_state_dict(ckpt["opt_d_state_dict"])
        sched_g.load_state_dict(ckpt["sched_g_state_dict"])
        sched_g.last_epoch = ckpt["step"]
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  → Step {start_step}")

    elif RESUME_MSE_ONLY_FROM and os.path.exists(RESUME_MSE_ONLY_FROM):
        print(f"Partendo da modello MSE-only: {RESUME_MSE_ONLY_FROM}")
        print(f"  → Il discriminatore parte da zero, il flow matching parte dal checkpoint")
        ckpt = torch.load(RESUME_MSE_ONLY_FROM, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        # Riparto da step 0 per il schedule adversarial
        start_step = 0

    else:
        print("Training adversarial da zero.")

    # info
    n_frames = train_dataset.n_frames
    print(f"\n{'='*60}")
    print(f"Training MSE + ADVERSARIAL")
    print(f"  Generator:       AudioDiT-{MODEL_KIND}")
    print(f"  Discriminator:   LatentDiscriminator-{DISC_KIND}")
    print(f"  Batch:           {BATCH_SIZE}")
    print(f"  LR (G/D):        {LR} / {LR_D}")
    print(f"  Lambda adv:      {LAMBDA_ADV_TARGET} (warmup {ADV_WARMUP_STEPS} step)")
    print(f"  Adv start step:  {ADV_START_STEP}")
    print(f"  R1 gamma:        {R1_GAMMA} (ogni {R1_INTERVAL} step)")
    print(f"{'='*60}\n")

    print("Log audio reali...")
    log_real_audio_samples(val_dataset, normalizer, writer)

    # Training loop
    val_loss = None
    audio_dir = os.path.join(CKPT_DIR, "audio")

    pbar = tqdm(range(start_step, NUM_STEPS), initial=start_step, total=NUM_STEPS,
                desc="Training", unit="step")

    last_step = start_step

    try:
        for step in pbar:
            last_step = step
            lambda_adv = get_lambda_adv(step)

            # =============================
            # STEP GENERATORE
            # =============================
            model.train()
            discriminator.eval()  # non aggiorna D in questo step
            for p in discriminator.parameters():
                p.requires_grad_(False)

            batch = next(train_iter)
            g_loss, g_info = compute_generator_loss(
                model, discriminator, batch, device, lambda_adv, step,
            )

            scaler_g.scale(g_loss).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler_g.step(opt_g)
            scaler_g.update()
            opt_g.zero_grad()
            sched_g.step()

            if step >= EMA_START:
                ema.update(model)

            # =============================
            # STEP DISCRIMINATORE
            # =============================
            if lambda_adv > 0:
                model.eval()
                discriminator.train()
                for p in discriminator.parameters():
                    p.requires_grad_(True)

                for _ in range(DISC_UPDATE_RATIO):
                    batch_d = next(train_iter)
                    apply_r1 = (step % R1_INTERVAL == 0)

                    d_loss, d_info = compute_discriminator_loss(
                        discriminator, model, batch_d, device, apply_r1,
                    )

                    # Il discriminatore non usa AMP sulla sua forward perche R1
                    # richiede grad non-mixed (puo crashare in fp16)
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
                    opt_d.step()
                    opt_d.zero_grad()

                sched_d.step()
            else:
                d_info = {}

            # =============================
            # LOG
            # =============================
            writer.add_scalar("Loss/train_total", g_loss.item(), step)
            writer.add_scalar("Loss/train_mse", g_info["mse"], step)
            writer.add_scalar("LambdaAdv", g_info["lambda_adv"], step)
            writer.add_scalar("LR/G", sched_g.get_last_lr()[0], step)

            if lambda_adv > 0:
                writer.add_scalar("Loss/train_adv", g_info["adv"], step)
                writer.add_scalar("Loss/D", d_info["d_loss"], step)
                writer.add_scalar("D/real_mean", d_info["d_real_mean"], step)
                writer.add_scalar("D/fake_mean", d_info["d_fake_mean"], step)
                if "r1" in d_info:
                    writer.add_scalar("D/r1", d_info["r1"], step)

            postfix = {
                "mse": f"{g_info['mse']:.3f}",
                "λ": f"{g_info['lambda_adv']:.2f}",
            }
            if lambda_adv > 0:
                postfix["adv"] = f"{g_info['adv']:+.2f}"
                postfix["D"] = f"{d_info['d_loss']:.2f}"
            pbar.set_postfix(**postfix)

            # =============================
            # VALIDAZIONE
            # MSE per best model selection (segnale stabile)
            # adv loggata SOLO come diagnostica (oscilla per natura GAN)
            # =============================
            if step % VAL_INTERVAL == 0:
                model.eval()
                discriminator.eval()

                with torch.no_grad():
                    # Val loss su modello principale
                    mse_losses = []
                    adv_losses = []
                    for _ in range(20):
                        vb = next(val_iter)
                        mse_v, adv_v = compute_val_losses(
                            model, discriminator, vb, device, lambda_adv,
                        )
                        mse_losses.append(mse_v)
                        adv_losses.append(adv_v)
                    val_loss = sum(mse_losses) / len(mse_losses)
                    val_adv  = sum(adv_losses) / len(adv_losses)

                    # Val loss su EMA (solo MSE)
                    ema_val_loss = val_loss
                    ema_val_adv  = val_adv
                    if step >= EMA_START:
                        mse_losses_ema = []
                        adv_losses_ema = []
                        for _ in range(20):
                            vb = next(val_iter)
                            mse_e, adv_e = compute_val_losses(
                                ema.model, discriminator, vb, device, lambda_adv,
                            )
                            mse_losses_ema.append(mse_e)
                            adv_losses_ema.append(adv_e)
                        ema_val_loss = sum(mse_losses_ema) / len(mse_losses_ema)
                        ema_val_adv  = sum(adv_losses_ema) / len(adv_losses_ema)
                        writer.add_scalar("Loss/val_ema_mse", ema_val_loss, step)
                        if lambda_adv > 0:
                            writer.add_scalar("Loss/val_ema_adv_diagnostic",
                                              ema_val_adv, step)

                writer.add_scalar("Loss/val_mse", val_loss, step)
                if lambda_adv > 0:
                    writer.add_scalar("Loss/val_adv_diagnostic", val_adv, step)

                # Stampa
                ema_str = ""
                if step >= EMA_START:
                    ema_str = f" | EMA val MSE {ema_val_loss:.4f}"
                adv_str = ""
                if lambda_adv > 0:
                    adv_str = f" | val_adv {val_adv:+.3f}"

                pbar.write(f"Step {step:7d} | train MSE {g_info['mse']:.4f} "
                            f"| val MSE {val_loss:.4f}{ema_str}{adv_str} "
                            f"| λ={lambda_adv:.3f}")

                # Best model: SEMPRE su MSE (stabile e confrontabile)
                check_loss = ema_val_loss if step >= EMA_START else val_loss
                if check_loss < best_val_loss:
                    best_val_loss = check_loss
                    sp = f"{CKPT_DIR}/best_model_step{step}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "ema_state_dict": ema.state_dict(),
                        "discriminator_state_dict": discriminator.state_dict(),
                        "opt_g_state_dict": opt_g.state_dict(),
                        "opt_d_state_dict": opt_d.state_dict(),
                        "sched_g_state_dict": sched_g.state_dict(),
                        "step": step, "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "model_kind": MODEL_KIND, "disc_kind": DISC_KIND,
                        "label_map": label_map, "n_frames": n_frames,
                    }, sp)
                    for old in Path(CKPT_DIR).glob("best_model_step*.pt"):
                        if old.resolve() != Path(sp).resolve():
                            old.unlink()
                    pbar.write(f"  → Best: {sp}")

            # =============================
            # AUDIO
            # =============================
            if step > 0 and step % AUDIO_INTERVAL == 0:
                gen_model = ema.model if step >= EMA_START else model
                generate_and_log_audio(
                    model=gen_model, normalizer=normalizer, n_frames=n_frames,
                    step=step, writer=writer, device=device, output_dir=audio_dir,
                    n_samples=N_AUDIO_SAMPLES,
                    prefix="EMA" if step >= EMA_START else "Model",
                )

                pbar.write(f"  Audio loggati (step {step})\n")
                model.train()

            # =============================
            # METRICHE
            # =============================
            if step > 0 and step % METRICS_INTERVAL == 0:
                gen_model = ema.model if step >= EMA_START else model
                fd_dac, fad = evaluate_and_log_metrics(
                    model=gen_model, normalizer=normalizer, val_dataset=val_dataset,
                    step=step, writer=writer, device=device, output_dir=audio_dir,
                    fad_calculator=fad_calculator, fd_dac_ref_stats=fd_dac_ref_stats,
                    n_samples=N_METRICS_SAMPLES,
                )
                pbar.write(f"  Metriche: FD-DAC={fd_dac:.4f} | FAD={fad:.4f}")
                model.train()

            # =============================
            # CHECKPOINT
            # =============================
            if step % CKPT_INTERVAL == 0 and step > 0:
                p = f"{CKPT_DIR}/checkpoint_step{step}.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "opt_g_state_dict": opt_g.state_dict(),
                    "opt_d_state_dict": opt_d.state_dict(),
                    "sched_g_state_dict": sched_g.state_dict(),
                    "step": step, "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "model_kind": MODEL_KIND, "disc_kind": DISC_KIND,
                    "label_map": label_map, "n_frames": n_frames,
                }, p)
                pbar.write(f"  → Checkpoint: {p}")

    finally:
        # Salva SEMPRE l'ultimo checkpoint: a fine training, dopo Ctrl+C, o errore
        last_path = f"{CKPT_DIR}/checkpoint_last_step{last_step}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "sched_g_state_dict": sched_g.state_dict(),
            "step": last_step, "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "model_kind": MODEL_KIND, "disc_kind": DISC_KIND,
            "label_map": label_map, "n_frames": n_frames,
        }, last_path)
        print(f"\n  → Ultimo checkpoint salvato: {last_path}")
        pbar.close()
        writer.close()
        print("Training terminato!")
