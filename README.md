# Audio Diffusion Transformer for Unconditional Music Generation

Unconditional music generation in the latent space of a neural audio codec, trained with the Rectified Flow objective. Audio is encoded by Descript Audio Codec (DAC) and modelled by a Diffusion Transformer (DiT) operating directly on per-frame latent tokens. Evaluation is performed with two complementary Fréchet distances: one computed in the DAC latent space (FD-DAC) and one in the Encodec embedding space (FAD).

This repository was developed at IRCAM (UMR STMS, Sound Analysis-Synthesis team) within the context of a doctoral research project.


## Overview

The pipeline is composed of three stages:

1. **Tokenisation.** Raw audio is encoded into a sequence of continuous latent vectors with [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec) at 44.1 kHz. The encoder produces 1024-dimensional latent frames at ≈86 Hz; a 5-second segment yields 431 frames.
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
- **Three size variants:** S (~30 M parameters), B (~150 M), L (~540 M).

### Training objective: Rectified Flow

Given a real latent `x₁` and Gaussian noise `x₀`, an interpolation point `xₜ = (1−t) x₀ + t x₁` is sampled with `t ∼ LogitNormal(0, 1)`. The network is trained to predict the constant velocity `v = x₁ − x₀`:

```
L_RF = E ‖ v_θ(xₜ, t) − (x₁ − x₀) ‖²
```

Optionally, an adversarial term can be added (see `training_npy_adv.py`) to sharpen high-frequency detail. A latent discriminator is trained jointly with non-saturating logistic loss and R1 gradient penalty ([Mescheder et al., 2018](https://arxiv.org/abs/1801.04406)).

### Sampling

A first-order Euler integrator from `t = 0.001` to `t = 0.999` over 50 steps is used by default. Additional schedulers can be plugged in `sampling.py`.

### Evaluation metrics

Two complementary Fréchet distances are computed every `intervals.metrics` training steps:

- **FD-DAC.** Fréchet distance between two multivariate Gaussians fitted to the *DAC latent* frames of (i) the validation set and (ii) the model's generations. This captures distributional fidelity in the codec's compression space.
- **FAD (Fréchet Audio Distance).** Same statistic, but computed on the embeddings of the [Encodec](https://arxiv.org/abs/2210.13438) encoder applied to decoded waveforms ([Kilgour et al., 2019](https://arxiv.org/abs/1812.08466)). Because Encodec embeddings are perceptually meaningful, FAD reflects audio-quality rather than codec-space behaviour.

Reference statistics for both metrics are pre-computed once over the entire validation split and cached.


## Repository layout

```
.
├── training.py                  # Main training script (Rectified Flow, EMA, AMP)
├── network.py                   # AudioDiT model definitions (S / B / L)
├── sampling.py                  # Euler sampling utilities
├── audio_dataset_npyy         # Dataset, normaliser, DAC loader
├── preprocess_dataset.py        # Audio → DAC latents pipeline (chunking, loudness norm.)
├── metrics.py                   # FD-DAC and FAD calculators
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
  kind: 'L'              # S | B | L
  duration_s: 5.0
data:
  batch_size: 4
  grad_accum: 1
training:
  num_steps: 1000000
  lr: 1.0e-4
  warmup_steps: 5000
  ema_decay: 0.9999
  use_amp: true
intervals:
  val: 1000
  audio: 25000
  metrics: 50000
  ckpt: 50000
sampling:
  euler_steps: 50
  n_metrics_samples: 64
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

Shared statistics (DAC latent normaliser, FD-DAC reference, FAD reference WAVs and Encodec stats) are cached under `cache/` and reused across runs operating on the same dataset.


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
python launch_training.py --run_name "DiT_B_lr5e5" \
    model.kind=B training.lr=5e-5

# Resume from a checkpoint
python launch_training.py --resume runs/old_run/checkpoints/checkpoint_step50000.pt
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
torch >= 2.0          # tested with 2.5.1 + CUDA 12.1
torchaudio
numpy < 2
descript-audio-codec
encodec
einops
soundfile
omegaconf
matplotlib
tensorboard
tqdm
scipy
```

A reference setup script for IRCAM servers is provided separately.


## Acknowledgements

This work was carried out at IRCAM as part of a doctoral research project on neural audio generation.

## References

- W. Peebles and S. Xie. *Scalable Diffusion Models with Transformers*. ICCV 2023. [arXiv:2212.09748](https://arxiv.org/abs/2212.09748)
- X. Liu, C. Gong, and Q. Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
- R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar. *High-Fidelity Audio Compression with Improved RVQGAN* (DAC). NeurIPS 2023. [arXiv:2306.06546](https://arxiv.org/abs/2306.06546)
- A. Défossez, J. Copet, G. Synnaeve, and Y. Adi. *High Fidelity Neural Audio Compression* (Encodec). TMLR 2023. [arXiv:2210.13438](https://arxiv.org/abs/2210.13438)
- K. Kilgour, M. Zuluaga, D. Roblek, and M. Sharifi. *Fréchet Audio Distance: A Reference-free Metric for Evaluating Music Enhancement Algorithms*. INTERSPEECH 2019. [arXiv:1812.08466](https://arxiv.org/abs/1812.08466)
- L. Mescheder, A. Geiger, and S. Nowozin. *Which Training Methods for GANs do actually Converge?* ICML 2018. [arXiv:1801.04406](https://arxiv.org/abs/1801.04406)
- J. Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. 2021. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- N. Shazeer. *GLU Variants Improve Transformer*. 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
