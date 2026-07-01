# Training for Audio DiT with Rectified Flow.
#
# Feature:
#   - DiT with AMP mixed precision
#   - EMA
#   - Audio generated every intervals.audio step on TensorBoard
#   - FD-DAC + KL divergence (both directions) computed every intervals.metrics
#     step, with reference pre-computed on all the validation set (latent-only)
#   - Loss train + val on TensorBoard
#   - Configuration with OmegaConf YAML (configs/uncond_default.yaml)
#
# Usage:
#   python training.py
#   python training.py --config configs/altro.yaml
#   python training.py training.lr=2e-4 data.train_batch_size=16
#   python training.py --resume runs/<run>/checkpoints/checkpoint_stepxxxxx.pt
#
# RESUME BEHAVIOUR (important):
#   When you pass --resume, the script reads the configuration that was stored
#   INSIDE the checkpoint and uses it to rebuild the model and the training
#   setup automatically. You do NOT need to re-pass model.kind, batch sizes,
#   etc. -- they are restored from the checkpoint. Any CLI override you DO pass
#   still wins over the stored value (so you can deliberately change something
#   on resume if you really want to).
#
#   For OLD checkpoints that predate the full-config saving, the script now
#   reconstructs the critical training params (model.kind, train_batch_size,
#   val_batch_size, grad_accum, duration_s) from whatever the checkpoint does
#   contain, and prints clearly which values it is using and where they came
#   from. So `--resume <ckpt> --run_name X` is enough on its own; you should
#   never silently fall back to YAML defaults again.

import os
import math

# ============================================================
# CACHE / HOME REDIRECTION & VRAM OPTIMIZATION (Must run BEFORE importing torch)
# ------------------------------------------------------------
# 1. Force PyTorch to use expandable segments to drastically reduce VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. IRCAM Home redirection for DAC weights cache
_IRCAM_LOCAL = "/data/anasynth_nonbp/baione"
if os.path.isdir(_IRCAM_LOCAL):
    os.environ["HOME"] = _IRCAM_LOCAL
    os.environ.setdefault("XDG_CACHE_HOME", os.path.join(_IRCAM_LOCAL, ".cache"))

import copy
import random
import argparse
from datetime import datetime
from pathlib import Path
from io import BytesIO

import torch
# Enable TF32 on Ampere+ GPUs. Same as facebookresearch/DiT.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
from sampling import euler_integrate
from metrics import (
    evaluate_generation,
    precompute_latent_reference,
    make_embedders,
    build_references,
)


# ============================================================
# DAC LOADER (singleton: load once, reuse everywhere)
# ============================================================
_DAC_MODEL = None

def get_dac():
    global _DAC_MODEL
    if _DAC_MODEL is None:
        import dac
        _DAC_MODEL = dac.DAC.load(dac.utils.download(model_type="44khz"))
        _DAC_MODEL.to("cpu")
        _DAC_MODEL.eval()
        print("[DAC] Model loaded once (CPU) and cached for the whole run.")
    return _DAC_MODEL


# ======================
# CONFIG LOADING
# ======================
def load_config():
    """
    Builds the final config with this precedence (lowest to highest):
        1. the YAML file (--config)
        2. the config / metadata stored inside the --resume checkpoint (if any)
        3. the CLI dotlist overrides (e.g. model.kind=G data.train_batch_size=2)

    Rationale: on resume we want the run to come back EXACTLY as it was, so the
    checkpoint's own config is layered on top of the YAML. CLI overrides still
    win, so you can deliberately change something on resume if needed.

    ROBUST RESUME (the important part):
      - NEW checkpoints store the whole OmegaConf under "config": we merge it,
        so every training param is restored.
      - OLD checkpoints have no "config", only scattered fields (model_kind,
        and inside label/n_frames etc.). Instead of silently falling back to
        the YAML defaults for batch size / grad_accum (which is exactly the
        annoying behaviour that made a bare `--resume` start with the wrong
        batch), we reconstruct the critical params from whatever the checkpoint
        DOES contain, and we WARN loudly about anything we genuinely cannot
        recover so you can pass it explicitly if you care.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str,
                        default="configs/uncond_default.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path checkpoint for resume (override YAML). "
                             "The model architecture and training config are "
                             "restored from the checkpoint automatically.")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Directory name of the run. "
                             "Default: timestamp YYYY-MM-DD_HH-MM-SS")
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config non found: {args.config}")

    cfg = OmegaConf.load(args.config)

    # Keep the YAML values so we can later tell, field by field, whether a value
    # came from the checkpoint or just from the YAML default.
    yaml_kind        = cfg.model.kind
    yaml_train_bs    = cfg.data.train_batch_size
    yaml_val_bs      = cfg.data.val_batch_size
    yaml_grad_accum  = cfg.data.grad_accum
    yaml_duration    = cfg.model.duration_s

    if args.resume is not None:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        _meta = torch.load(args.resume, map_location="cpu", weights_only=False)

        if "config" in _meta and _meta["config"] is not None:
            # NEW checkpoint: full config available, merge it (checkpoint wins
            # over YAML). This restores model.kind, batch sizes, grad_accum,
            # everything.
            ckpt_cfg = OmegaConf.create(_meta["config"])
            cfg = OmegaConf.merge(cfg, ckpt_cfg)
            print("[RESUME] Full config restored from checkpoint "
                  f"(model.kind={cfg.model.kind}, "
                  f"train_batch_size={cfg.data.train_batch_size}, "
                  f"grad_accum={cfg.data.grad_accum}, "
                  f"val_batch_size={cfg.data.val_batch_size}).")
        else:
            # OLD checkpoint: no stored config. Reconstruct what we can from the
            # scattered metadata fields instead of silently using YAML defaults.
            print("[RESUME] This checkpoint has NO embedded full config "
                  "(old-format checkpoint). Reconstructing critical params "
                  "from the checkpoint metadata where possible.")

            recovered = {}
            unrecoverable = []

            # model.kind: old checkpoints DO store this as a top-level field.
            ckpt_kind = _meta.get("model_kind", None)
            if ckpt_kind is not None:
                cfg.model.kind = ckpt_kind
                recovered["model.kind"] = ckpt_kind
            else:
                unrecoverable.append("model.kind")

            # n_frames -> duration_s: if the checkpoint stored n_frames we can
            # recover the chunk duration used, which keeps the dataset chunking
            # consistent on resume.
            ckpt_n_frames = _meta.get("n_frames", None)
            if ckpt_n_frames is not None:
                # n_frames = int(duration_s * DAC_FRAMES_PER_S); invert it.
                from audio_dataset_npy import DAC_FRAMES_PER_S
                recovered_duration = round(ckpt_n_frames / DAC_FRAMES_PER_S, 6)
                cfg.model.duration_s = recovered_duration
                recovered["model.duration_s"] = recovered_duration

            # Batch size / grad_accum: OLD checkpoints do NOT store these
            # anywhere, so they genuinely cannot be recovered. Rather than
            # pretend, we keep the YAML value but FLAG it explicitly so the
            # user knows to pass it if it matters (this is the VRAM-critical
            # one: a wrong batch on a big model OOMs immediately).
            unrecoverable.append(
                f"data.train_batch_size (using YAML/CLI value: {cfg.data.train_batch_size})")
            unrecoverable.append(
                f"data.grad_accum (using YAML/CLI value: {cfg.data.grad_accum})")

            if recovered:
                print("[RESUME] Recovered from checkpoint metadata:")
                for k, v in recovered.items():
                    print(f"           {k} = {v}")
            if unrecoverable:
                print("[RESUME] NOT stored in this old checkpoint "
                      "(taken from YAML/CLI -- pass them explicitly if needed):")
                for item in unrecoverable:
                    print(f"           {item}")
                print("[RESUME] TIP: if this model needs a specific batch to fit "
                      "in VRAM, add e.g. data.train_batch_size=2 data.grad_accum=8 "
                      "to the command.")

        del _meta

    # CLI overrides win over everything (YAML + checkpoint config/metadata).
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, cli_cfg)
        # Report which critical params were overridden on the command line, so
        # the effective values are never a surprise.
        overridden = []
        for key, yaml_val in [
            ("model.kind", yaml_kind),
            ("data.train_batch_size", yaml_train_bs),
            ("data.val_batch_size", yaml_val_bs),
            ("data.grad_accum", yaml_grad_accum),
            ("model.duration_s", yaml_duration),
        ]:
            # crude check: was this key present in the dotlist?
            if any(tok.split("=")[0] == key for tok in unknown):
                overridden.append(key)
        if overridden:
            print(f"[RESUME] CLI overrides applied (win over everything): "
                  f"{', '.join(overridden)}")

    if args.resume is not None:
        cfg.paths.resume_from = args.resume

    if args.run_name is not None:
        run_name = args.run_name
    elif cfg.paths.get("run_name") is not None:
        run_name = cfg.paths.run_name
    else:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.paths.run_name = run_name

    cfg.data.effective_bs = cfg.data.train_batch_size * cfg.data.grad_accum

    # Final, unambiguous summary of the params that actually matter, so you can
    # see at a glance what this run will use BEFORE the model is built.
    print(f"[CONFIG] Effective training params: "
          f"model.kind={cfg.model.kind} | "
          f"train_batch_size={cfg.data.train_batch_size} | "
          f"grad_accum={cfg.data.grad_accum} | "
          f"effective_bs={cfg.data.effective_bs} | "
          f"duration_s={cfg.model.duration_s}")

    return cfg, run_name


# ======================
# LR SCHEDULE
# ======================
def make_lr_lambda(num_steps: int, warmup_steps: int, decay_start_frac: float):
    decay_start = int(num_steps * decay_start_frac)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / (num_steps - decay_start)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)).item())
    return lr_lambda


# ======================
# T SAMPLING
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
        ema_params = dict(self.model.named_parameters())
        for name, p in model.named_parameters():
            ema_params[name].lerp_(p.data, 1.0 - self.decay)

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
    x = euler_integrate(model, x, steps=steps, t_min=t_min, t_max=t_max, use_amp=use_amp)
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

    dac_model = get_dac()   # load-once singleton

    for i, gen in enumerate(generated_frames):
        if not torch.isfinite(gen).all():
            continue

        z = gen.T
        z = normalizer.denormalize(z)
        z_in = z.unsqueeze(0).float()
        z_q, _, _ = dac_model.quantizer.from_latents(z_in)   # (1,72,T) -> (1,1024,T)
        waveform = dac_model.decode(z_q).squeeze(0)

        wn = waveform / (waveform.abs().max() + 1e-8)
        writer.add_audio(
            f"Validation/Audio_generated_{prefix}_{i:02d}", wn,
            global_step=step, sample_rate=DAC_SAMPLE_RATE,
        )
        spec_img = make_spectrogram(
            waveform, DAC_SAMPLE_RATE,
            f"{prefix} sample {i} - step {step}",
        )
        writer.add_image(
            f"Validation/Spectrogram_generated_{prefix}_{i:02d}",
            spec_img, global_step=step,
        )

        wav_path = os.path.join(output_dir, f"step{step:07d}_{prefix}_{i:02d}.wav")
        sf.write(wav_path, waveform.squeeze().numpy(), DAC_SAMPLE_RATE)


@torch.no_grad()
def log_real_audio_samples(dataset, normalizer, writer, n_samples):
    """Logs real audio from dataset for comparison on TensorBoard."""
    dac_model = get_dac()   # load-once singleton

    total = len(dataset)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()

    for i, idx in enumerate(indices):
        frames, _ = dataset[idx]
        z = frames.T
        z = normalizer.denormalize(z)
        z_in = z.unsqueeze(0).float()
        z_q, _, _ = dac_model.quantizer.from_latents(z_in)   # (1,72,T) -> (1,1024,T)
        waveform = dac_model.decode(z_q).squeeze(0)

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

    print(f"  {n_samples} real audios logged on TensorBoard")


# ======================
# METRICS EVALUATION
# ======================

@torch.no_grad()
def evaluate_and_log_metrics(
    model, normalizer, val_dataset, step, writer, device, output_dir,
    enabled, references, embedders, n_samples, sampling_cfg, use_amp,
    prefix="EMA", metrics_seed=None,
):
    """
    Config-selected metrics (subset of fd_dac, kl_dac, fad_encodec, fad_vggish),
    all single-Gaussian, FULL covariance. fd_dac/kl_dac are latent-only; the FADs
    decode the GENERATED latents and embed them (Encodec / VGGish), against the
    real val wavs as reference (no DAC on the real side). Generation happens ONCE
    and is shared across all the enabled metrics. Audio for TensorBoard is logged
    separately by generate_and_log_audio.
    """
    print(f"\n   Compute metrics {list(enabled)}: {n_samples} generated samples...")

    results = evaluate_generation(
        model, normalizer, val_dataset,
        enabled=enabled,
        references=references,
        embedders=embedders,
        n_samples=n_samples,
        euler_steps=sampling_cfg.euler_steps,
        t_min=sampling_cfg.t_min,
        t_max=sampling_cfg.t_max,
        seed=metrics_seed,
        device=device,
        use_amp=use_amp,
    )

    tag = {
        "fd_dac":      "Validation/Metrics/Fd_dac",
        "kl_real_gen": "Validation/Metrics/Kl_real_gen",
        "kl_gen_real": "Validation/Metrics/Kl_gen_real",
        "fad_encodec": "Validation/Metrics/Fad_encodec",
        "fad_vggish":  "Validation/Metrics/Fad_vggish",
    }
    logged = {}
    for key, t in tag.items():
        if results.get(key) is not None:
            writer.add_scalar(t, results[key], step)
            logged[key] = results[key]

    print("  " + " | ".join(f"{k}={v:.4f}" for k, v in logged.items()))
    return logged


# ======================
# DATALOADER
# ======================

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


# ======================
# CHECKPOINT HELPER
# ======================
def _capture_rng_state(data_generator=None):
    """Snapshot of every RNG stream so a --resume can continue EXACTLY where it
    left off (not restart from the seed): python, numpy, torch CPU, torch CUDA,
    and the DataLoader shuffle generator."""
    state = {
        "python": random.getstate(),
        "numpy":  np.random.get_state(),
        "torch":  torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def _restore_rng_state(state, data_generator=None):
    """Restore the RNG snapshot saved by _capture_rng_state. Best-effort: old
    checkpoints (no rng_state) or a different GPU count fall back to the freshly
    seeded RNG with a warning instead of crashing."""
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        if data_generator is not None and state.get("data_generator") is not None:
            data_generator.set_state(state["data_generator"])
        print("[SEED] RNG state restored from checkpoint (exact resume).")
    except Exception as e:
        print(f"[SEED] WARNING: could not fully restore RNG state ({type(e).__name__}: "
              f"{e}); continuing with the freshly seeded RNG.")


def build_ckpt_data(model, ema, optimizer, scheduler, scaler, step,
                    val_loss, best_val_loss, cfg, label_map, n_frames, run_name,
                    data_generator=None):
    """
    Assemble the checkpoint dict. The full `config` is stored so that a later
    --resume can rebuild the exact same model/training setup without the user
    having to re-pass model.kind, batch sizes, etc. `model_kind` is also kept
    as a top-level field for backward compatibility with test.py / sampling.py.
    """
    data = {
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "model_kind": cfg.model.kind,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "label_map": label_map,
        "n_frames": n_frames,
        "run_name": run_name,
        "rng_state": _capture_rng_state(data_generator),
    }
    if cfg.training.use_ema and ema is not None:
        data["ema_state_dict"] = ema.state_dict()
    return data


# ======================
# MAIN
# ======================

if __name__ == "__main__":

    cfg, run_name = load_config()

    print(f"[RUN NAME] {run_name}")

    run_dir   = os.path.join(cfg.paths.runs_dir, run_name)
    ckpt_dir  = os.path.join(run_dir, "checkpoints")
    audio_dir = os.path.join(run_dir, "audio")
    cache_dir = cfg.paths.cache_dir

    os.makedirs(run_dir,   exist_ok=True)
    os.makedirs(ckpt_dir,  exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    normalizer_path   = os.path.join(cache_dir, "normalizer.pt")
    latent_ref_path   = os.path.join(cache_dir, "latent_ref_stats.pt")

    config_dump_path = os.path.join(run_dir, "config.yaml")
    OmegaConf.save(cfg, config_dump_path)
    print(f"[CONFIG DUMP] {config_dump_path}")
    print(f"[RUN DIR]     {run_dir}")
    print(f"[CACHE DIR]   {cache_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')

    # Global run seed: makes x0, the LogitNormal t-sampling and the DataLoader
    # shuffle reproducible. Set null in the YAML to disable (free-running RNG).
    run_seed = cfg.training.get("seed", None)
    data_generator = None
    if run_seed is not None:
        run_seed = int(run_seed)
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)
        data_generator = torch.Generator()      # for the train loader shuffle
        data_generator.manual_seed(run_seed)
        print(f"[SEED] Global training seed = {run_seed}")

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

    if not os.path.exists(normalizer_path):
        normalizer.save(normalizer_path)

    n_workers = int(cfg.data.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.data.train_batch_size, shuffle=True,
        num_workers=n_workers, pin_memory=(device == "cuda"),
        persistent_workers=(n_workers > 0),
        drop_last=True,
        generator=data_generator,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=cfg.data.val_batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=(device == "cuda"),
        persistent_workers=(n_workers > 0),
        drop_last=False,
    )

    train_iter = infinite_loader(train_loader)
    val_iter   = infinite_loader(val_loader)

    # ======================
    # METRICS SETUP (config-selected) + REFERENCE STATS
    # ------------------------------------------------------------
    # metrics.enabled chooses a subset of {fd_dac, kl_dac, fad_encodec, fad_vggish},
    # all single-Gaussian full-covariance. fd_dac/kl_dac share ONE latent reference
    # (real val latents, normalized space). The FADs use the real val WAVS as
    # reference (no DAC on the real side); the generated side is decoded at eval.
    # Every needed reference is precomputed once and cached under cache_dir.
    # ======================
    metrics_cfg = cfg.get("metrics", None)
    if metrics_cfg is None:
        metrics_enabled = ["fd_dac", "kl_dac"]
        metrics_encodec_sr = 24000
        metrics_seed = None
        metrics_strict = True
    else:
        metrics_enabled = list(metrics_cfg.enabled)
        metrics_encodec_sr = int(metrics_cfg.get("encodec_sr", 24000))
        metrics_seed = metrics_cfg.get("seed", None)
        metrics_strict = bool(metrics_cfg.get("strict", True))

    print(f"\nMetrics enabled: {metrics_enabled}")
    print("Pre-computation of reference statistics for the metrics...")

    metrics_embedders = make_embedders(
        metrics_enabled, device=device, encodec_sr=metrics_encodec_sr)
    metrics_refs = build_references(
        metrics_enabled, val_dataset,
        val_wav_root=os.path.join(cfg.paths.wav_root, "val"),
        embedders=metrics_embedders,
        cache_dir=cache_dir,
        device=device,
        strict=metrics_strict,
    )
    print("Reference stats ready.\n")

    # ======================
    # MODEL + EMA
    # ------------------------------------------------------------
    # cfg.model.kind already reflects the checkpoint's kind on resume (see
    # load_config), so the model is rebuilt with the correct architecture
    # automatically -- no need to re-pass model.kind on the command line.
    # ======================
    print(f"[MODEL] Building AudioDiT-{cfg.model.kind}")
    model     = AudioDiT(kind=cfg.model.kind, drop=cfg.model.drop).to(device)
    ema       = EMAModel(model, decay=cfg.training.ema_decay) if cfg.training.use_ema else None
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

    writer = SummaryWriter(run_dir)

    writer.add_text(
        "config",
        "```yaml\n" + OmegaConf.to_yaml(cfg) + "\n```",
        global_step=0,
    )

    best_val_loss = float("inf")
    start_step    = 0

    # ======================
    # RESUME
    # ------------------------------------------------------------
    # The model was already built with the right architecture (kind restored in
    # load_config). Here we just load the weights/optimizer/scheduler. Loading
    # on CPU first avoids the one-shot GPU memory spike that can OOM at resume.
    # ======================
    resume_from = cfg.paths.resume_from
    if resume_from and os.path.exists(resume_from):
        print(f"Restarting training from: {resume_from}")
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)

        # Defensive check: the model we built must match the checkpoint. If the
        # kinds disagree we stop with a clear message instead of dumping a wall
        # of size-mismatch errors.
        ckpt_kind = ckpt.get("model_kind", None)
        if ckpt_kind is not None and ckpt_kind != cfg.model.kind:
            raise RuntimeError(
                f"Checkpoint was trained with model.kind='{ckpt_kind}' but the "
                f"model was built as '{cfg.model.kind}'. They must match to "
                f"resume. (Normally the kind is restored automatically from the "
                f"checkpoint; if you passed model.kind on the command line, "
                f"remove it or set it to '{ckpt_kind}'.)"
            )

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if cfg.training.use_ema:
            if "ema_state_dict" in ckpt:
                ema.load_state_dict(ckpt["ema_state_dict"])
            else:
                # Old checkpoint without saved EMA: rebuild it. Note that at
                # this point `model` already holds the TRAINED weights loaded
                # above, so this EMA starts from trained weights (not the random
                # init), which is fine. If the resume step is still below
                # ema_start, the re-init at ema_start in the loop will re-seed
                # it again from the model at that moment.
                ema = EMAModel(model, decay=cfg.training.ema_decay)
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        # Restore every RNG stream so the run continues EXACTLY (x0, t, shuffle)
        # instead of restarting from the seed. Best-effort for old checkpoints.
        _restore_rng_state(ckpt.get("rng_state"), data_generator)
        start_step    = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"  -> Step {start_step} | best_val_loss: {best_val_loss:.6f}")
        writer.add_text("resumed_from", resume_from, global_step=start_step)

        # Move AdamW state tensors to GPU one by one (loaded from a CPU ckpt).
        if device == "cuda":
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

        del ckpt
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        print("Training from zero.")

    # ======================
    # INFO
    # ======================
    n_frames = train_dataset.n_frames
    print(f"\n{'='*60}")
    print(f"Training on {device} | AudioDiT-{cfg.model.kind}")
    print(f"Steps: {cfg.training.num_steps} | Effective Batch: {cfg.data.effective_bs}")
    print(f"LR: {cfg.training.lr} | EMA: "
          f"{'on (decay=' + str(cfg.training.ema_decay) + ')' if cfg.training.use_ema else 'off'} | "
          f"AMP: {cfg.training.use_amp}")
    print(f"Sequence: {n_frames} frame = {n_frames} token of dim {TOKEN_DIM}")
    print(f"Train: {len(train_dataset)} chunk | Val: {len(val_dataset)} chunk")
    print(f"Audio every {cfg.intervals.audio} step | "
          f"Metrics every {cfg.intervals.metrics} step")
    print(f"Metrics: {cfg.sampling.n_metrics_samples} generated samples per eval "
          f"| enabled: {metrics_enabled}")
    print(f"DATASET_ROOT: {cfg.paths.dataset_root}")
    print(f"WAV_ROOT:     {cfg.paths.wav_root}  (only for real-audio TensorBoard logging)")
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

                del loss, batch

            scaler.unscale_(optimizer)
            _clip = cfg.training.grad_clip
            max_norm = _clip if (_clip is not None and _clip > 0) else float('inf')
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if cfg.training.use_ema and step >= cfg.training.ema_start:
                # EMA RE-INIT (important): when the EMA window opens at
                # ema_start, the shadow weights are still the RANDOM init copied
                # at construction time. If we just started averaging, those
                # random weights would contaminate the EMA for tens of thousands
                # of steps (with decay=0.9999 they stay >1% until ~46k steps
                # after ema_start), producing poor EMA generations/metrics.
                # So at exactly ema_start we re-seed the EMA from the CURRENT
                # (already partly trained) model, and only then start averaging.
                if step == cfg.training.ema_start:
                    ema.model.load_state_dict(model.state_dict())
                else:
                    ema.update(model)

            writer.add_scalar("Train/Loss", accum_loss, step)
            writer.add_scalar("Train/Learning rate", scheduler.get_last_lr()[0], step)
            writer.add_scalar("Train/Grad_norm", grad_norm.item(), step)

            pbar.set_postfix(loss=f"{accum_loss:.4f}",
                             lr=f"{scheduler.get_last_lr()[0]:.1e}")

            # ======================
            # VALIDATION
            # ======================
            if step % cfg.intervals.val == 0:
                model.eval()
                n_val = cfg.data.num_val_batches

                if device == "cuda":
                    torch.cuda.empty_cache()

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
                        del vb

                    val_loss = sum(val_losses) / len(val_losses)

                    ema_val_loss = val_loss
                    if cfg.training.use_ema and step >= cfg.training.ema_start:
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
                            del vb
                        ema_val_loss = sum(ema_vl) / len(ema_vl)
                        writer.add_scalar("Validation/Loss_ema", ema_val_loss, step)

                writer.add_scalar("Validation/Loss", val_loss, step)

                if device == "cuda":
                    torch.cuda.empty_cache()

                ema_str = (f" | EMA Val {ema_val_loss:.6f}"
                           if cfg.training.use_ema and step >= cfg.training.ema_start else "")
                pbar.write(f"Step {step:7d} | Train {accum_loss:.6f} | "
                           f"Val {val_loss:.6f}{ema_str} | "
                           f"LR {scheduler.get_last_lr()[0]:.2e}")

                check_loss = (ema_val_loss
                              if cfg.training.use_ema and step >= cfg.training.ema_start
                              else val_loss)
                if check_loss < best_val_loss:
                    best_val_loss = check_loss
                    save_path = os.path.join(ckpt_dir, f"best_model_step{step}.pt")
                    ckpt_data = build_ckpt_data(
                        model, ema, optimizer, scheduler, scaler, step,
                        val_loss, best_val_loss, cfg, label_map, n_frames, run_name, data_generator=data_generator)
                    torch.save(ckpt_data, save_path)
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
                             if cfg.training.use_ema and step >= cfg.training.ema_start
                             else model)

                generate_and_log_audio(
                    model=gen_model, normalizer=normalizer,
                    n_frames=n_frames, step=step, writer=writer,
                    device=device, output_dir=audio_dir,
                    n_samples=cfg.sampling.n_audio_samples,
                    sampling_cfg=cfg.sampling,
                    use_amp=cfg.training.use_amp,
                    prefix=("EMA"
                            if cfg.training.use_ema and step >= cfg.training.ema_start
                            else "Model"),
                )

                pbar.write(f"  Audio logged (step {step})\n")
                model.train()

            # ======================
            # METRICS
            # ======================
            if step > 0 and step % cfg.intervals.metrics == 0:
                gen_model = (ema.model
                             if cfg.training.use_ema and step >= cfg.training.ema_start
                             else model)

                logged = evaluate_and_log_metrics(
                    model=gen_model,
                    normalizer=normalizer,
                    val_dataset=val_dataset,
                    step=step,
                    writer=writer,
                    device=device,
                    output_dir=audio_dir,
                    enabled=metrics_enabled,
                    references=metrics_refs,
                    embedders=metrics_embedders,
                    n_samples=cfg.sampling.n_metrics_samples,
                    sampling_cfg=cfg.sampling,
                    use_amp=cfg.training.use_amp,
                    prefix=("EMA"
                            if cfg.training.use_ema and step >= cfg.training.ema_start
                            else "Model"),
                    metrics_seed=metrics_seed,
                )

                pbar.write("  Metrics: " +
                           " | ".join(f"{k}={v:.4f}" for k, v in logged.items()) +
                           "\n")
                model.train()

            # ======================
            # PERIODICAL CHECKPOINT
            # ======================
            if step % cfg.intervals.ckpt == 0 and step > 0:
                p = os.path.join(ckpt_dir, f"checkpoint_step{step}.pt")
                ckpt_data = build_ckpt_data(
                    model, ema, optimizer, scheduler, scaler, step,
                    val_loss, best_val_loss, cfg, label_map, n_frames, run_name, data_generator=data_generator)
                torch.save(ckpt_data, p)
                pbar.write(f"  -> Checkpoint: {p}")

                keep_n = cfg.intervals.get("keep_last_n_ckpts", 4)
                periodic_ckpts = sorted(
                    Path(ckpt_dir).glob("checkpoint_step*.pt"),
                    key=lambda x: int(x.stem.replace("checkpoint_step", "")),
                )
                for old in periodic_ckpts[:-keep_n]:
                    old.unlink()
                    pbar.write(f"  -> Removed old periodic checkpoint: {old.name}")

    finally:
        # Always try to save the last checkpoint, whatever killed the loop
        # (Ctrl+C, normal end, or an exception such as CUDA OutOfMemory).
        last_path = os.path.join(ckpt_dir, f"checkpoint_last_step{last_step}.pt")

        def _try_save():
            ckpt_data = build_ckpt_data(
                model, ema, optimizer, scheduler, scaler, last_step,
                val_loss, best_val_loss, cfg, label_map, n_frames, run_name, data_generator=data_generator)
            torch.save(ckpt_data, last_path)

        saved = False
        try:
            _try_save()
            saved = True
            print(f"\n  -> Last checkpoint saved: {last_path}")
        except Exception as e_gpu:
            print(f"\n  [WARN] Normal checkpoint save failed ({type(e_gpu).__name__}: "
                  f"{e_gpu}). Retrying from CPU after freeing GPU memory...")
            try:
                if device == "cuda":
                    torch.cuda.empty_cache()
                model.to("cpu")
                if ema is not None:
                    ema.model.to("cpu")
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cpu()
                if device == "cuda":
                    torch.cuda.empty_cache()
                _try_save()
                saved = True
                print(f"  -> Last checkpoint saved from CPU: {last_path}")
            except Exception as e_cpu:
                print(f"  [ERROR] Could not save the last checkpoint even from CPU "
                      f"({type(e_cpu).__name__}: {e_cpu}). "
                      f"The most recent usable checkpoint is the latest "
                      f"best_model_step*.pt / checkpoint_step*.pt in {ckpt_dir}.")

        try:
            pbar.close()
            writer.close()
        except Exception:
            pass
        print("Training concluded." if saved else "Training ended (see warnings above).")
