# Audio Diffusion Transformer for Unconditional Music Generation

Unconditional music generation in the latent space of a neural audio codec, trained with the Rectified Flow objective. Audio is encoded by Descript Audio Codec (DAC) and modelled by a Diffusion Transformer (DiT) operating directly on per-frame latent tokens. Evaluation is performed entirely in the DAC latent space with two complementary distributional metrics: the Fréchet distance (FD-DAC) and the Kullback–Leibler divergence (KL), both under a shared multivariate-Gaussian model of the latent distributions.

This repository was developed at IRCAM (UMR STMS, Sound Analysis-Synthesis team) within the context of a doctoral research project.


## Overview

The pipeline is composed of three stages:

1. **Tokenisation.** Raw audio is encoded into a sequence of continuous latent vectors with [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) at 44.1 kHz. The encoder produces 72-dimensional pre-quantizer latent frames at ≈86 Hz; a 5-second segment yields ≈430 frames. (The DAC decoder consumes the 1024-dim quantized `z`, reconstructed from the 72-dim latents via `quantizer.from_latents()`.)
2. **Generative model.** A Diffusion Transformer (DiT) is trained on the pre-computed latent sequences with a [Rectified Flow](https://arxiv.org/abs/2209.03003) objective: the network predicts the velocity field of a continuous-time interpolation between Gaussian noise and the data distribution.
3. **Decoding.** At inference, latent samples are drawn by Euler integration of the learned velocity field and decoded back to waveform with the frozen DAC decoder.

No conditioning signal is used. The model learns a class-agnostic distribution over the training corpus.


## Architectural choices

### Generator: AudioDiT

The backbone is a transformer-based denoiser with the following components:

- **Token-level inputs.** Each DAC latent frame is treated as a single token; no patching or downsampling is applied.
- **AdaLN-Zero conditioning** of the timestep embedding into every block, following [Peebles & Xie, 2023](https://arxiv.org/abs/2212.09748).
- **Rotary positional embeddings (RoPE)** along the temporal axis ([Su et al., 2021](https://arxiv.org/abs/2104.09864)).
- **SwiGLU feed-forward layers** ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)).
- **Optional dropout** in the feed-forward layers (two masks, timm-style: one on the gated hidden activation, one on the output projection), controlled by `model.drop`. Disabled by default (`0.0`).
- **Five size variants:** S (~30.9 M parameters), B (~129.5 M), G (~348.1 M), L (~463.0 M), XL (~673.5 M). Param counts are post-SwiGLU-2/3-fix.

The **G** variant (18 layers, hidden 1024, 16 heads) is an intermediate size between B and L. It was introduced as the largest model that still fits training at a small batch size on the 12 GB IRCAM GPUs (RTX 4070) in pure fp32 — without mixed precision, gradient checkpointing or optimizer-state compression. It shares the hidden width (1024) and head count (16) with L, so it keeps the same per-head dimension and RoPE configuration, and sits at 18 layers, midway between B (12) and L (24).

The **XL** variant (28 layers, hidden 1152, 16 heads) mirrors the official [facebookresearch/DiT](https://github.com/facebookresearch/DiT/blob/main/models.py) DiT-XL configuration, intended for the larger IRCAM GPUs (chichibu 24 GB, vacqueyras / A6000 49 GB). Unlike S/B/G/L it has `head_dim = 72` (1152 / 16), exactly as in the official DiT-XL; the RoPE setup adapts automatically because the rotary frequencies are derived from `head_dim`. To instead keep the project-wide `head_dim = 64`, use 18 heads (1152 / 18) rather than 16.

### Training objective: Rectified Flow

Given a real latent `x₁` and Gaussian noise `x₀`, an interpolation point `xₜ = (1−t) x₀ + t x₁` is sampled with `t ∼ LogitNormal(0, 1)`. The network is trained to predict the constant velocity `v = x₁ − x₀`:

```
L_RF = E ‖ v_θ(xₜ, t) − (x₁ − x₀) ‖²
```

### Sampling

A first-order Euler integrator from `t = 0.001` to `t = 0.999` over 50 steps is used by default. Additional schedulers can be plugged in `sampling.py`.

### Evaluation metrics

Evaluation uses up to **four** distributional metrics, chosen from config via `metrics.enabled` and computed every `intervals.metrics` training steps. All four model the real and generated feature distributions as multivariate Gaussians `N(μ, Σ)` with **full** covariance (a single Gaussian, no mixtures), so within each space they are directly comparable. Each evaluation **generates the samples once** and reuses them across the enabled metrics.

Two are **latent-only** (no audio decoding), in the (normalized) DAC latent space, with the generated latents taken pre-denormalization:

- **`fd_dac`.** Fréchet (Wasserstein-2) distance between the two Gaussians:
  `FD = ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_r Σ_g)^½)`. Symmetric; distributional fidelity in the codec's compression space.
- **`kl_dac`.** Closed-form Kullback–Leibler divergence between the same two Gaussians, **both directions** (`KL(real‖gen)` and `KL(gen‖real)`), via a numerically-stable Cholesky factorization (log-determinant from the Cholesky diagonal; trace and Mahalanobis terms via triangular solves, no explicit inversion), with covariance regularization.

Two are **audio FADs** (decode + embedding network). The generated latents are DAC-decoded to waveform — unavoidable, since the model only outputs DAC latents — then embedded; the reference is the **real validation wavs embedded directly (no DAC on the real side)**, so the FAD measures the generative gap, not the codec's reconstruction quality. Both are single-Gaussian, full-covariance Fréchet:

- **`fad_encodec`.** On Encodec encoder embeddings (128-D, frame-level ~13 ms), faithful to the supervisor's reference implementation.
- **`fad_vggish`.** On VGGish embeddings (128-D, ~0.96 s temporal window — the only metric with a ~1 s window, intrinsic to VGGish's log-mel front-end).

Reference statistics for every enabled metric are pre-computed once over the validation split and cached (`latent_ref_stats.pt`, `fad_encodec_ref_stats.pt`, `fad_vggish_ref_stats.pt`).

> **Note.** All four are single multivariate Gaussians with full covariance, matching the official FAD/FD/KL definitions. Gaussian mixtures (MW₂ for the Fréchet, variational KL) were considered but not adopted: neither the official metrics nor the audio-generation literature use them, and the mixture KL has no closed form. `fad_encodec`/`fad_vggish` re-introduce audio decoding plus an embedding network (the cost the latent-only metrics avoid), so enable them only when literature-comparable FAD numbers are wanted.


## Repository layout

```
.
├── training.py                  # Main training script (Rectified Flow, EMA, AMP)
├── network.py                   # AudioDiT model definitions (S / B / G / L / XL)
├── sampling.py                  # Euler sampling utilities
├── audio_dataset_npy.py         # Dataset, normaliser, DAC loader
├── preprocess_dataset.py        # Audio → DAC latents pipeline (chunking, loudness norm.)
├── metrics.py                   # fd_dac, kl_dac (latent) + fad_encodec, fad_vggish (config-selected)
├── test.py                      # Generation + comparison with real samples (TensorBoard)
├── launch_training.py           # GPU-lock wrapper for IRCAM servers
├── configs/
│   └── uncond_default.yaml      # Default OmegaConf configuration
└── README.md
```

The runtime artifacts produced by training (`runs/<run_name>/`) and the shared statistics (`cache/`) are excluded from version control by `.gitignore`.


## Configuration

All hyperparameters are declared in a single OmegaConf YAML file (`configs/uncond_default.yaml`). The structure is hierarchical:

```yaml
model:
  kind: 'L'              # S | B | G | L | XL
  duration_s: 5.0
  drop: 0.0              # dropout in the FFN layers (0.0 = off)
data:
  train_batch_size: 8
  val_batch_size: 8
  grad_accum: 1
  num_workers: 4
training:
  num_steps: 1000000
  lr: 1.0e-4
  seed: 42                # global run seed (x0, t, shuffle); null = off
  warmup_steps: 5000
  ema_decay: 0.9999
  use_amp: false
intervals:
  val: 1000
  audio: 25000
  metrics: 50000
  ckpt: 50000
sampling:
  euler_steps: 100
  n_metrics_samples: 128
  t_min: 0.001
  t_max: 0.999
metrics:
  enabled: [fd_dac, kl_dac]   # FADs (fad_encodec, fad_vggish) are opt-in
  encodec_sr: 24000
  seed: 0                 # fixes metric generation noise; null = free-running
  strict: true            # stop at startup if an enabled metric can't be built; false = warn & skip
paths:
  dataset_root: "./dataset_ready/latents"
  wav_root:     "./dataset_ready/wav"
  runs_dir:     "runs"
  cache_dir:    "cache"
```

Each run is identified by a `run_name` (CLI argument or YAML field; defaults to a timestamp). Outputs are organised as:

```
runs/<run_name>/
    config.yaml                  # Effective configuration of this run
    events.out.tfevents.*        # TensorBoard logs
    checkpoints/                 # Periodical and best-model checkpoints
    audio/                       # Generated WAVs and spectrograms
```

Shared statistics (DAC latent normaliser, and the latent reference `μ`/`Σ` used by both FD-DAC and KL) are cached under `cache/` and reused across runs operating on the same dataset.


## Usage

### Pre-processing

The raw audio collection is expected to be organised as one directory per class:

```
raw_audio/
    Baroque/
    Jazz/
    Romanticism_chamber/
    ...
```

Run the pre-processing pipeline (silence trimming, loudness normalisation, chunking, DAC encoding):

```bash
python preprocess_dataset.py raw_audio/ dataset_ready/ \
    --chunk_length 5 \
    --device cuda
```

The output is a structure with `latents/{train,val,test}/<class>/*.npy` and `wav/{val,test}/<class>/*.wav` (training waveforms are not stored).

### Training

The `launch_training.py` wrapper acquires a GPU lock on IRCAM servers before importing PyTorch and forwards every other argument to `training.py`:

```bash
# Default run on a single GPU
python launch_training.py

# CLI overrides (dotlist syntax, parsed by OmegaConf)
python launch_training.py --run_name "DiT_G_lr5e5" \
    model.kind=G training.lr=5e-5

# Resume from a checkpoint: model.kind, batch size, grad_accum and paths are
# restored automatically from the checkpoint, so --resume (optionally with a
# --run_name) is enough.
python launch_training.py --run_name "DiT_G_resumed" \
    --resume runs/old_run/checkpoints/checkpoint_step50000.pt
```

### Inference / evaluation

Generate samples from a trained checkpoint and compare them to real ones on TensorBoard:

```bash
python test.py --ckpt runs/<run_name>/checkpoints/best_model.pt \
    --n_samples 16 --steps 100
```

Outputs are written to `runs/<run_name>/test_outputs/` and `runs/<run_name>/test_logs/`.


## Dependencies

```
python >= 3.10
torch >= 2.0          # tested with 2.5.1 + CUDA 12.1; cu130 build on CUDA 13.x machines
torchaudio
torchvision          # plot_to_image (spectrogram logging); pulls in Pillow
numpy < 2
descript-audio-codec
soundfile
omegaconf
matplotlib
tensorboard
tqdm
scipy
encodec               # for the fad_encodec metric
resampy               # for the fad_vggish metric (VGGish log-mel front-end)
```

The latent metrics (`fd_dac`, `kl_dac`) need no extra feature extractor. The audio FADs add dependencies: `fad_encodec` needs `encodec`; `fad_vggish` loads VGGish via `torch.hub` from `harritaylor/torchvggish`, which downloads the weights on first use — it therefore needs network access once, after which they are cached under `~/.cache/torch/hub` (pre-fetch on a networked machine for offline GPU nodes). Enable only the metrics you need in `metrics.enabled` to avoid these costs.

A reference setup script for IRCAM servers is provided separately.


## Acknowledgements

This work was carried out at IRCAM as part of a doctoral research project on neural audio generation.

## References

- W. Peebles and S. Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- X. Liu, C. Gong, and Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar. *High-Fidelity Audio Compression with Improved RVQGAN* (DAC). NeurIPS 2023. [arXiv:2306.06546](https://arxiv.org/abs/2306.06546)
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- N. Shazeer. *GLU Variants Improve Transformer*. 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)