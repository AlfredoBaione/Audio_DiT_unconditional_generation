# test.py
#
# Generates audio with EMA model and compares to real audio on TensorBoard.
# Aligned with the refactored training.py (OmegaConf + run_name layout).
#
# Usage:
#   python test.py --ckpt runs/<run_name>/checkpoints/best_model.pt
#   python test.py --ckpt runs/<run_name>/checkpoints/best_model.pt --config configs/uncond_default.yaml
#   python test.py --ckpt path/to/ckpt.pt --n_samples 16 --steps 100
#
# Outputs:
#   - WAV files in runs/<run_name>/test_outputs/
#   - TensorBoard logs in runs/<run_name>/test_logs/
#     (visible alongside the training logs of the same run)

import os

# Use a machine-local cache for HuggingFace / DAC weights (avoids NFS issues).
os.environ.setdefault("XDG_CACHE_HOME", "/data/anasynth_nonbp/baione/.cache")

import argparse
import sys
from io import BytesIO
from pathlib import Path

import torch
import soundfile as sf
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from audio_dataset_npy import (
    AudioLatentDataset,
    LatentNormalizer,
    decode_latents,
    DAC_SAMPLE_RATE,
)
from network import AudioDiT, TOKEN_DIM
from sampling import euler_sampling


# ============================================================
# CLI / CONFIG LOADING
# ============================================================
def load_config():
    """
    Loads the same YAML used for training, then applies CLI overrides.
    Returns (cfg, args).
    """
    parser = argparse.ArgumentParser(
        description="Generate audio with a trained Audio DiT and compare with real samples.",
        add_help=True,
    )
    parser.add_argument("--config", type=str,
                        default="configs/uncond_default.yaml",
                        help="YAML config file (default: configs/uncond_default.yaml)")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint (.pt) — typically "
                             "runs/<run_name>/checkpoints/best_model.pt")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Override run_name. If not given, it is inferred "
                             "from the checkpoint path "
                             "(runs/<run_name>/checkpoints/...).")
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of samples to generate (default: 8)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Number of Euler steps "
                             "(default: cfg.sampling.euler_steps)")
    parser.add_argument("--duration_s", type=float, default=None,
                        help="Audio duration in seconds "
                             "(default: cfg.model.duration_s)")
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg = OmegaConf.load(args.config)

    # CLI dotlist overrides (e.g. sampling.euler_steps=80)
    if unknown:
        cli_cfg = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # CLI scalars take priority over YAML when explicitly set
    if args.steps is not None:
        cfg.sampling.euler_steps = args.steps
    if args.duration_s is not None:
        cfg.model.duration_s = args.duration_s

    # Infer run_name from --ckpt if not given
    # Expected layout: runs/<run_name>/checkpoints/<file>.pt
    if args.run_name is not None:
        run_name = args.run_name
    else:
        ckpt_path = Path(args.ckpt).resolve()
        # Walk up: <run_name>/checkpoints/<file>
        if ckpt_path.parent.name == "checkpoints":
            run_name = ckpt_path.parent.parent.name
        else:
            run_name = "test"  # fallback if checkpoint is not in the expected layout
    cfg.paths.run_name = run_name

    return cfg, args


# ============================================================
# UTILITIES
# ============================================================
def plot_to_image(fig):
    """Convert a matplotlib figure to a torch tensor (3, H, W) for TensorBoard."""
    import torchvision
    import PIL.Image as Image

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img = torchvision.transforms.ToTensor()(Image.open(buf))
    buf.close()
    return img


def make_spectrogram_image(waveform, sample_rate, title=""):
    """Build a mel-spectrogram image (tensor) for TensorBoard."""
    spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=128, n_fft=2048, hop_length=512,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()
    spec_db = amp_to_db(spec_transform(waveform.cpu().float()))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(spec_db[0].numpy(), aspect="auto", origin="lower",
              cmap="viridis", vmin=-80, vmax=0)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mel Bin")
    plt.colorbar(ax.images[0], ax=ax, label="dB")
    img = plot_to_image(fig)
    plt.close(fig)
    return img


# ============================================================
# MAIN
# ============================================================
def main():
    cfg, args = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] Device: {device}")
    print(f"[test] Config: {args.config}")
    print(f"[test] Checkpoint: {args.ckpt}")
    print(f"[test] Run name: {cfg.paths.run_name}")

    # Output paths
    run_dir = Path(cfg.paths.runs_dir) / cfg.paths.run_name
    output_dir = run_dir / "test_outputs"
    log_dir    = run_dir / "test_logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[test] Outputs:    {output_dir}")
    print(f"[test] TB logs:    {log_dir}")

    writer = SummaryWriter(str(log_dir))

    # ============================================================
    # LOAD CHECKPOINT + MODEL
    # ============================================================
    print(f"\n[test] Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location=device)
    model_kind = ckpt.get("model_kind", cfg.model.kind)
    print(f"[test] Model kind: {model_kind}")

    model = AudioDiT(kind=model_kind).to(device)
    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("[test] Using EMA weights")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("[test] EMA not available — using main model weights")
    model.eval()

    # ============================================================
    # NORMALIZER
    # ============================================================
    # Resolve normalizer path: prefer cache_dir from config, fallback to run_dir
    cache_dir = Path(cfg.paths.cache_dir)
    normalizer_candidates = [
        cache_dir / "normalizer.pt",
        run_dir / "checkpoints" / "normalizer.pt",
    ]
    normalizer_path = None
    for p in normalizer_candidates:
        if p.exists():
            normalizer_path = p
            break
    if normalizer_path is None:
        raise FileNotFoundError(
            f"normalizer.pt not found in any of: "
            f"{[str(p) for p in normalizer_candidates]}"
        )

    normalizer = LatentNormalizer()
    normalizer.load(str(normalizer_path))
    print(f"[test] Normalizer: {normalizer_path}")

    # Label map (optional, for naming)
    label_map = ckpt.get("label_map", {})
    idx_to_label = {v: k for k, v in label_map.items()}

    # ============================================================
    # TEST SET
    # ============================================================
    test_dataset = AudioLatentDataset(
        root_dir=cfg.paths.dataset_root,
        split="test",
        duration_s=cfg.model.duration_s,
        normalizer=normalizer,
        preload=False,
    )

    total = len(test_dataset)
    if total == 0:
        raise RuntimeError(f"Empty test set in {cfg.paths.dataset_root}/test")

    n_samples = min(args.n_samples, total)
    indices = torch.linspace(0, total - 1, n_samples).long().tolist()
    print(f"[test] Test set: {total} samples | using {n_samples}")

    # ============================================================
    # GENERATION + LOGGING
    # ============================================================
    n_frames = test_dataset.n_frames
    euler_steps = int(cfg.sampling.euler_steps)
    print(f"\n[test] --- Generating {n_samples} samples "
          f"({n_frames} frames each, {euler_steps} Euler steps) ---")

    for i, idx in enumerate(indices):
        frames_real, label_idx = test_dataset[idx]
        label_name = idx_to_label.get(label_idx, str(label_idx))

        print(f"\n[test] Sample {i+1}/{n_samples} | idx={idx} | label={label_name}")

        # --- Generate latent ---
        with torch.no_grad():
            frames_gen = euler_sampling(
                model=model,
                n_samples=1,
                n_frames=n_frames,
                steps=euler_steps,
                device=device,
            )   # (1, n_frames, 1024)

        # --- Latent → audio (denormalize + DAC decode) ---
        z_gen = frames_gen[0].T                # (1024, n_frames)
        z_gen = normalizer.denormalize(z_gen)
        waveform_gen = decode_latents(z_gen, device=device)

        # Save WAV
        out_path = output_dir / f"generated_{i:04d}_{label_name}.wav"
        sf.write(str(out_path), waveform_gen.cpu().numpy().T, DAC_SAMPLE_RATE)
        print(f"[test]   Saved: {out_path}")

        # --- Log generated audio ---
        wn = waveform_gen / (waveform_gen.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/generated/{label_name}", wn.cpu(),
            global_step=i, sample_rate=DAC_SAMPLE_RATE,
        )

        # --- Generated spectrogram ---
        spec_img_gen = make_spectrogram_image(
            waveform_gen, DAC_SAMPLE_RATE,
            f"Generated — {label_name}",
        )
        writer.add_image(
            f"Spectrogram/generated/{label_name}", spec_img_gen, global_step=i,
        )

        # --- Real audio reference ---
        z_real = frames_real.T                  # (1024, n_frames)
        z_real = normalizer.denormalize(z_real)
        waveform_real = decode_latents(z_real, device=device)

        wrn = waveform_real / (waveform_real.abs().max() + 1e-8)
        writer.add_audio(
            f"Audio/real/{label_name}", wrn.cpu(),
            global_step=i, sample_rate=DAC_SAMPLE_RATE,
        )

        spec_img_real = make_spectrogram_image(
            waveform_real, DAC_SAMPLE_RATE,
            f"Real — {label_name}",
        )
        writer.add_image(
            f"Spectrogram/real/{label_name}", spec_img_real, global_step=i,
        )

    writer.close()
    print(f"\n[test] Done!")
    print(f"[test] WAV files:    {output_dir}")
    print(f"[test] TensorBoard:  tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()

