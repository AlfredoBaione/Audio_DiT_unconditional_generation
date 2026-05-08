# Training for Audio DiT with Rectified Flow.
#
# Feature:
#   - DiT with AMP mixed precision
#   - EMA
#   - Audio generated every intervals.audio step on TensorBoard
#   - FD with DAC + FAD computed every intervals.metrics step
#     with reference pre-computed on all the validation set
#   - Loss train + val on TensorBoard
#   - Configuration with OmegaConf YAML (configs/uncond_default.yaml)
#
# Usage:
#   python training_npy.py
#   python training_npy.py --config configs/altro.yaml
#   python training_npy.py training.lr=2e-4 data.batch_size=16
#   python training_npy.py --resume checkpoints_v2/checkpoint_stepxxxxx.pt

import os
import copy
import argparse
from datetime import datetime
from pathlib import Path
from io import BytesIO


os.environ.setdefault("XDG_CACHE_HOME", "/data/anasynth_nonbp/baione/.cache")

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
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
# CONFIG LOADING
# ======================
def load_config():
    """
    Loads the config from YAML, applies override CLI in dotlist
    (es. training.lr=2e-4 data.batch_size=16) and handles --resume + --run_name.

    Returns:
        cfg:      OmegaConf with the final config (CLI override already applied)
        run_name: string identifying the run (default timestamp)
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str,
                        default="configs/uncond_default.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path checkpoint  for resume (override YAML). "
                             "Starts a new run from given checkpoint.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Directory name of the run. "
                             "Default: timestamp YYYY-MM-DD_HH-MM-SS")
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config non found: {args.config}")

    cfg = OmegaConf.load(args.config)

    if unknown:
        cli_cfg = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # CLI --resume prevail over YAML
    if args.resume is not None:
        cfg.paths.resume_from = args.resume

    # Run name: CLI --run_name > YAML paths.run_name > timestamp default
    if args.run_name is not None:
        run_name = args.run_name
    elif cfg.paths.get("run_name") is not None:
        run_name = cfg.paths.run_name
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Persist final value in cfg so it appears in the dumped config.yaml
    cfg.paths.run_name = run_name

    # Derivates
    cfg.data.effective_bs = cfg.data.batch_size * cfg.data.grad_accum

    return cfg, run_name


# ======================
# LR SCHEDULE (factory: gets num_steps and schedule via closure)
# ======================
def make_lr_lambda(num_steps: int, warmup_steps: int, decay_start_frac: float):
    decay_start = int(num_steps * decay_start_frac)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / (num_steps - decay_start)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return lr_lambda


# ======================
# T SAMPLING (receives t_min/t_max explicitly)
# ======================
def sample_logit_normal(batch_size, device, t_min, t_max, mean=0.0, std=1.0):
    u = torch.randn(batch_size, device=device) * std + mean
    return torch.sigmoid(u).clamp(t_min, t_max)


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
def compute_loss(model, batch, device, use_amp, t_min, t_max):
    x1, _ = batch
    x1 = x1.to(device).float()
    B  = x1.shape[0]

    x0 = torch.randn_like(x1)
    t  = sample_logit_normal(B, device, t_min, t_max)

    t_expand = t.view(B, 1, 1)
    xt = (1 - t_expand) * x0 + t_expand * x1
    target = x1 - x0

    with torch.amp.autocast('cuda', enabled=use_amp):
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
def euler_sample(model, n_frames, device, steps, t_min, t_max, use_amp):
    model.eval()
    x = torch.randn(1, n_frames, TOKEN_DIM, device=device)
    dt = (t_max - t_min) / steps
    for i in range(steps):
        t_val = t_min + i * dt
        t = torch.ones(1, device=device) * t_val
        with torch.amp.autocast('cuda', enabled=use_amp):
            v = model(x, t)
        x = x + v.float() * dt
    return x[0].cpu()


@torch.no_grad()
def generate_and_log_audio(
    model, normalizer, n_frames, step, writer, device,
    output_dir, n_samples, sampling_cfg, use_amp, prefix="EMA",
):
    """Generates audio, decodes with DAC on CPU, logs on TensorBoard."""
    generated_frames = []
    for i in range(n_samples):
        gen = euler_sample(
            model, n_frames, device,
            steps=sampling_cfg.euler_steps,
            t_min=sampling_cfg.t_min,
            t_max=sampling_cfg.t_max,
            use_amp=use_amp,
        )
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
            f"Validation/Audio_generated_{i:02d}", wn,
            global_step=step, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(
            waveform, DAC_SAMPLE_RATE,
            f"{prefix} sample {i} - step {step}",
        )
        writer.add_image(
            f"Validation/Spectrogram_generated_{i:02d}",
            spec_img, global_step=step,
        )

        wav_path = os.path.join(output_dir, f"step{step:07d}_{prefix}_{i:02d}.wav")
        sf.write(wav_path, waveform.squeeze().numpy(), DAC_SAMPLE_RATE)

    del dac_model


@torch.no_grad()
def log_real_audio_samples(dataset, normalizer, writer, n_samples):
    """Logs real audio from dataset for comparison on TensorBoard."""
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
            f"Validation/Audio_real_{i:02d}", wn,
            global_step=0, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(waveform, DAC_SAMPLE_RATE, f"Real sample {i}")
        writer.add_image(
            f"Validation/Spectrogram_real_{i:02d}",
            spec_img, global_step=0,
        )

    del dac_model
    print(f"  {n_samples} real audios logged on TensorBoard")


# ======================
# METRICS EVALUATION
# ======================

@torch.no_grad()
def evaluate_and_log_metrics(
    model, normalizer, val_dataset, step, writer, device, output_dir,
    fad_calculator, fd_dac_ref_stats, n_samples, sampling_cfg, use_amp,
):
    """
    Generates N samples, computes FD-DAC + FAD against pre-computed references,
    and logs everything on TensorBoard.
    """
    print(f"\n  Compute metrics: {n_samples} generated samples"
          f"vs reference ({fad_calculator.ref_n_samples} samples)...")

    results = evaluate_generation(
        model=model,
        normalizer=normalizer,
        val_dataset=val_dataset,
        n_samples=n_samples,
        euler_steps=sampling_cfg.euler_steps,
        device=device,
        use_amp=use_amp,
        fad_calculator=fad_calculator,
        fd_dac_ref_stats=fd_dac_ref_stats,
    )

    fd_dac = results["fd_dac"]
    fad    = results["fad"]

    writer.add_scalar("Validation/Metrics/Fd_dac", fd_dac, step)
    writer.add_scalar("Validation/Metrics/Fad", fad, step)

    print(f"  FD-DAC: {fd_dac:.4f} | FAD: {fad:.4f}")

    # Logga alcuni audio generati
    for i, wav in enumerate(results["generated_wavs"][:4]):
        wav = wav.unsqueeze(0) if wav.dim() == 1 else wav
        wn = wav / (wav.abs().max() + 1e-8)
        writer.add_audio(
            f"Validation/Audio_generated_for_metrics_{i:02d}", wn,
            global_step=step, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(
            wav, DAC_SAMPLE_RATE,
            f"Metrics Gen {i} — step {step} — FD-DAC={fd_dac:.2f} FAD={fad:.2f}",
        )
        writer.add_image(
            f"Validation/Spectrogram_generated_for_metrics_{i:02d}",
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

    

    cfg, run_name = load_config()
    
    print(f"[RUN NAME] {run_name}")

    # ======================
    # RUN DIRECTORY (self-contained) + CACHE DIRECTORY (shared)
    # ======================
    run_dir   = os.path.join(cfg.paths.runs_dir, run_name)
    ckpt_dir  = os.path.join(run_dir, "checkpoints")
    audio_dir = os.path.join(run_dir, "audio")
    cache_dir = cfg.paths.cache_dir

    os.makedirs(run_dir,   exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    # Paths derived from the above directories 
    normalizer_path   = os.path.join(cache_dir, "normalizer.pt")
    fd_dac_cache_path = os.path.join(cache_dir, "fd_dac_ref_stats.pt")
    fad_cache_dir     = os.path.join(cache_dir, "fad_cache")

    # Config's dump (with CLI override already applied) in the run dir
    config_dump_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(cfg, config_dump_path)
    print(f"[CONFIG DUMP] {config_dump_path}")
    print(f"[RUN DIR]     {run_dir}")
    print(f"[CACHE DIR]   {cache_dir}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")
    # ======================
    # DATA
    # ======================
    print("Loading dataset...")
    train_dataset, val_dataset, normalizer, label_map = build_datasets(
        root_dir=cfg.paths.dataset_root,
        duration_s=cfg.model.duration_s,
        normalizer_path=(normalizer_path
                         if os.path.exists(normalizer_path) else None),
        preload=False,
    )

    # Save the normalizer in the cache_dir 
    if not os.path.exists(normalizer_path):
        normalizer.save(normalizer_path)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.batch_size, shuffle=True,
        num_workers=0, pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.batch_size, shuffle=False,
        num_workers=0, pin_memory=(device == "cuda"),
        drop_last=False,
    )

    train_iter = infinite_loader(train_loader)
    val_iter   = infinite_loader(val_loader)

    # ======================
    # PRE-COMPUTAION REFERENCE STATS (one time only, cached in cache_dir)
    # ======================
    print("\nPre-computation of reference statistics for the metrics...")

    # FD-DAC: mu and sigma (full copvariance) on the DAC latents of the validation set
    fd_dac_ref_stats = precompute_fd_dac_reference(
        val_dataset,
        cache_path=fd_dac_cache_path,
    )

    # FAD: embedding Encodec of every validation WAV
    fad_calculator = FADCalculator(device="cuda")
    fad_calculator.precompute_reference_stats(
        val_dataset=val_dataset,
        normalizer=normalizer,
        wav_root=cfg.paths.wav_root,
        latent_root=cfg.paths.dataset_root,
        sr=DAC_SAMPLE_RATE,
        cache_dir=fad_cache_dir,
    )

    print(f"Reference stats ready: "
          f"FD-DAC on {len(val_dataset)} latent samples, "
          f"FAD on {fad_calculator.ref_n_samples} audio samples\n")

    # ======================
    # MODEL + EMA
    # ======================
    model     = AudioDiT(kind=cfg.model.kind).to(device)
    ema       = EMAModel(model, decay=cfg.training.ema_decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    lr_lambda = make_lr_lambda(
        num_steps=cfg.training.num_steps,
        warmup_steps=cfg.training.warmup_steps,
        decay_start_frac=cfg.training.decay_start_frac,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda', enabled=cfg.training.use_amp)

    # SummaryWriter points directly to the run directory
    writer = SummaryWriter(run_dir)

    # Logs the config in TB (tab "Text" -> visible on the dashboard)
    writer.add_text(
        "config",
        "```yaml\n" + OmegaConf.to_yaml(cfg) + "\n```",
        global_step=0,
    )

    best_val_loss = float("inf")
    start_step    = 0

    # ======================
    # RESUME
    # ======================
    resume_from = cfg.paths.resume_from
    if resume_from and os.path.exists(resume_from):
        print(f"Riprendendo training da: {resume_from}")
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scheduler.last_epoch = ckpt["step"]
        if "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        else:
            ema = EMAModel(model, decay=cfg.training.ema_decay)
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step    = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  -> Step {start_step} | best_val_loss: {best_val_loss:.6f}")
        # Logs in TB the checkpoint resume
        writer.add_text("resumed_from", resume_from, global_step=start_step)
    else:
        print("Training from zero.")

    # ======================
    # INFO
    # ======================
    n_frames = train_dataset.n_frames
    print(f"\n{'='*60}")
    print(f"Training on {device} | AudioDiT-{cfg.model.kind}")
    print(f"Steps: {cfg.training.num_steps} | Effective Batch: {cfg.data.effective_bs}")
    print(f"LR: {cfg.training.lr} | EMA decay: {cfg.training.ema_decay} | "
          f"AMP: {cfg.training.use_amp}")
    print(f"Sequence: {n_frames} frame = {n_frames} token of dim {TOKEN_DIM}")
    print(f"Train: {len(train_dataset)} chunk | Val: {len(val_dataset)} chunk")
    print(f"Audio every {cfg.intervals.audio} step | "
          f"Metrics every {cfg.intervals.metrics} step")
    print(f"Metrics: {cfg.sampling.n_metrics_samples} generated vs "
          f"{fad_calculator.ref_n_samples} reference")
    print(f"DATASET_ROOT: {cfg.paths.dataset_root}")
    print(f"WAV_ROOT:     {cfg.paths.wav_root}")
    print(f"RUN DIR:      {run_dir}")
    print(f"{'='*60}\n")

    # ======================
    # LOGS REAL AUDIOS
    # ======================
    print("Logging real audio on TensorBoard...")
    log_real_audio_samples(
        val_dataset, normalizer, writer,
        n_samples=cfg.sampling.n_audio_samples,
    )

    # ======================
    # TRAIN LOOP
    # ======================
    val_loss = None

    pbar = tqdm(range(start_step, cfg.training.num_steps),
                initial=start_step, total=cfg.training.num_steps,
                desc="Training", unit="step")

    last_step = start_step

    try:
        for step in pbar:
            last_step = step

            model.train()

            accum_loss = 0.0
            for _ in range(cfg.data.grad_accum):
                batch = next(train_iter)
                loss  = compute_loss(
                    model, batch, device,
                    use_amp=cfg.training.use_amp,
                    t_min=cfg.sampling.t_min,
                    t_max=cfg.sampling.t_max,
                ) / cfg.data.grad_accum
                scaler.scale(loss).backward()
                accum_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            if step >= cfg.training.ema_start:
                ema.update(model)

            writer.add_scalar("Train/Loss", accum_loss, step)
            writer.add_scalar("Train/Learning rate", scheduler.get_last_lr()[0], step)

            pbar.set_postfix(loss=f"{accum_loss:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.1e}")

            # ======================
            # VALIDATION (loss)
            # ======================
            if step % cfg.intervals.val == 0:
                model.eval()
                n_val = cfg.data.num_val_batches
                with torch.no_grad():
                    val_losses = []
                    for _ in range(n_val):
                        vb = next(val_iter)
                        vl = compute_loss(
                            model, vb, device,
                            use_amp=cfg.training.use_amp,
                            t_min=cfg.sampling.t_min,
                            t_max=cfg.sampling.t_max,
                        ).item()
                        val_losses.append(vl)
                    val_loss = sum(val_losses) / len(val_losses)

                    ema_val_loss = val_loss
                    if step >= cfg.training.ema_start:
                        ema_vl = []
                        for _ in range(n_val):
                            vb = next(val_iter)
                            evl = compute_loss(
                                ema.model, vb, device,
                                use_amp=cfg.training.use_amp,
                                t_min=cfg.sampling.t_min,
                                t_max=cfg.sampling.t_max,
                            ).item()
                            ema_vl.append(evl)
                        ema_val_loss = sum(ema_vl) / len(ema_vl)
                        writer.add_scalar("Validation/Loss_ema", ema_val_loss, step)

                writer.add_scalar("Validation/Loss", val_loss, step)

                ema_str = (f" | EMA Val {ema_val_loss:.6f}"
                           if step >= cfg.training.ema_start else "")
                pbar.write(f"Step {step:7d} | Train {accum_loss:.6f} | "
                           f"Val {val_loss:.6f}{ema_str} | "
                           f"LR {scheduler.get_last_lr()[0]:.2e}")

                # Best model
                check_loss = (ema_val_loss
                              if step >= cfg.training.ema_start
                              else val_loss)
                if check_loss < best_val_loss:
                    best_val_loss = check_loss
                    save_path = os.path.join(ckpt_dir, f"best_model_step{step}.pt")
                    torch.save({
                        "model_state_dict":     model.state_dict(),
                        "ema_state_dict":       ema.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict":    scaler.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "best_val_loss": best_val_loss,
                        "model_kind": cfg.model.kind,
                        "label_map": label_map,
                        "n_frames": n_frames,
                        "run_name": run_name,
                    }, save_path)
                    for old in Path(ckpt_dir).glob("best_model_step*.pt"):
                        if old.resolve() != Path(save_path).resolve():
                            old.unlink()
                    pbar.write(f"  -> Best model: {save_path}")

            # ======================
            # AUDIO GENERATION
            # ======================
            if step > 0 and step % cfg.intervals.audio == 0:
                pbar.write(f"\n Audio generation step {step}...")

                gen_model = (ema.model
                             if step >= cfg.training.ema_start
                             else model)

                generate_and_log_audio(
                    model=gen_model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer,
                    device=device, output_dir=audio_dir,
                    n_samples=cfg.sampling.n_audio_samples,
                    sampling_cfg=cfg.sampling,
                    use_amp=cfg.training.use_amp,
                    prefix=("EMA"
                            if step >= cfg.training.ema_start
                            else "Model"),
                )

                pbar.write(f"  Audio logged (step {step})\n")
                model.train()

            # ======================
            # METRICS (FD-DAC + FAD)
            # ======================
            if step > 0 and step % cfg.intervals.metrics == 0:
                gen_model = (ema.model
                             if step >= cfg.training.ema_start
                             else model)

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
                    n_samples=cfg.sampling.n_metrics_samples,
                    sampling_cfg=cfg.sampling,
                    use_amp=cfg.training.use_amp,
                )

                pbar.write(f"  Metrics: FD-DAC={fd_dac:.4f} | FAD={fad:.4f}\n")
                model.train()

            # ======================
            # PERIODICAL CHECKPOINT
            # ======================
            if step % cfg.intervals.ckpt == 0 and step > 0:
                p = os.path.join(ckpt_dir, f"checkpoint_step{step}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "model_kind": cfg.model.kind,
                    "label_map": label_map,
                    "n_frames": n_frames,
                    "run_name": run_name,
                }, p)
                pbar.write(f"  -> Checkpoint: {p}")

    finally:
        # Always save the last checkpoint: at the end of the training, after Ctrl+C, or error
        last_path = os.path.join(ckpt_dir, f"checkpoint_last_step{last_step}.pt")
        torch.save({
            "model_state_dict": model.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "step": last_step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "model_kind": cfg.model.kind,
            "label_map": label_map,
            "n_frames": n_frames,
            "run_name": run_name,
        }, last_path)
        print(f"\n  -> Last checkpoint saved: {last_path}")
        pbar.close()
        writer.close()
        print("Training concluded.")
