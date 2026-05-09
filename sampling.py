# sampling.py
#
# Sampling with Euler for Rectified Flow — without patching.
# It works directly on the DAC frames.

import os
import numpy as np
import soundfile as sf
import torch

from audio_dataset_npy import (
    LatentNormalizer,
    decode_latents,
    DAC_FRAMES_PER_S,
    MAX_FRAMES,
)
from network import AudioDiT, TOKEN_DIM

T_MIN = 0.001
T_MAX = 0.999


@torch.no_grad()
def euler_sampling(
    model:     AudioDiT,
    n_samples: int,
    n_frames:  int,
    steps:     int   = 10,
    device:    str   = "cpu",
) -> torch.Tensor:
    """
    Samples from the model with Euler.

    Args:
        model:     AudioDiT (possibly EMA)
        n_samples: numnbers of audio to be generated
        n_frames:  sequence length (= number of tokens/frames)
        steps:     integration steps
        device:    device

    Returns:
        frames: (n_samples, n_frames, 1024) in the normalized space
    """
    model.eval()

    # Initial noise
    x = torch.randn(n_samples, n_frames, TOKEN_DIM, device=device)
    dt = (T_MAX - T_MIN) / steps

    for i in range(steps):
        t_val = T_MIN + i * dt
        t = torch.ones(n_samples, device=device) * t_val
        v = model(x, t)
        x = x + v * dt

    return x.cpu()


@torch.no_grad()
def generate_audio(
    model:       AudioDiT,
    normalizer:  LatentNormalizer,
    n_samples:   int   = 1,
    duration_s:  float = 5.0,
    steps:       int   = 10,
    device:      str   = "cpu",
    output_dir:  str   = "./generated",
    sample_rate: int   = 44100,
) -> list:
    """
    Complete pipeline: noise → frame → denormalize → DAC decode → wav.
    """
    os.makedirs(output_dir, exist_ok=True)

    n_frames = int(duration_s * DAC_FRAMES_PER_S)
    n_frames = min(n_frames, MAX_FRAMES)

    actual_duration = n_frames / DAC_FRAMES_PER_S
    print(f"[generate] {n_samples} audio | "
          f"{duration_s:.1f}s requested → {actual_duration:.1f}s effective "
          f"({n_frames} frame/token)")

    # 1. Euler sampling
    frames = euler_sampling(model, n_samples, n_frames=n_frames, steps=steps, device=device)
    # frames: (B, n_frames, 1024)

    generated_paths = []

    for i in range(n_samples):
        # 2. Transpose: (n_frames, 1024) → (1024, n_frames) for DAC
        z = frames[i].T    # (1024, n_frames)

        # 3. Denormalize
        z = normalizer.denormalize(z)

        # 4. DAC decode
        waveform = decode_latents(z, device=device)  # (1, T)

        # 5. Save
        out_path = os.path.join(output_dir, f"sample_{i:04d}.wav")
        wav_np = waveform.cpu().numpy().T
        sf.write(out_path, wav_np, sample_rate)
        generated_paths.append(out_path)
        print(f"  Saved: {out_path} | shape: {waveform.shape}")

    return generated_paths


@torch.no_grad()
def euler_sampling_with_trajectory(
    model:    AudioDiT,
    n_frames: int  = 430,
    steps:    int  = 50,
    device:   str  = "cpu",
) -> list:
    """Like euler_sampling but saves intermediate snapshots for debug."""
    model.eval()
    x = torch.randn(1, n_frames, TOKEN_DIM, device=device)
    dt = (T_MAX - T_MIN) / steps
    trajectory = [x.clone().cpu()]

    for i in range(steps):
        t_val = T_MIN + i * dt
        t = torch.ones(1, device=device) * t_val
        v = model(x, t)
        x = x + v * dt
        trajectory.append(x.clone().cpu())

    return trajectory


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys

    ckpt_path  = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_v2/best_model.pt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./generated_v2"
    n_samples  = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    duration_s = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
    steps      = int(sys.argv[5]) if len(sys.argv) > 5 else 100

    print(f"Load checkpoint: {ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    model_kind = ckpt.get("model_kind", "L")
    model = AudioDiT(kind=model_kind).to(device)

    if "ema_state_dict" in ckpt:
        model.load_state_dict(ckpt["ema_state_dict"])
        print("  → Using EMA model")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
        print("  → EMA not available")

    normalizer = LatentNormalizer()
    normalizer.load("checkpoints_v2/normalizer.pt")

    paths = generate_audio(
        model=model, normalizer=normalizer,
        n_samples=n_samples, duration_s=duration_s,
        steps=steps, device=device, output_dir=output_dir,
    )
    print(f"\nGenerated {len(paths)} files in '{output_dir}'")
