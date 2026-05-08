# metrics.py
#
# Evaluation metrics for generating audio:
#   - FD-DAC (Frechet DAC Distance): distanza di Frechet sui latenti DAC
#     con covarianza piena (upgrade della vecchia KL diagonale)
#   - FAD (Frechet Audio Distance): qualita percettiva con Encodec
#
# Differenze chiave rispetto alla KL diagonale precedente:
#   - Covarianza piena (1024x1024) invece di varianza per-canale
#   - Cattura le correlazioni tra dimensioni dei latenti DAC
#   - Usa la stessa formula di Frechet di FAD ma nello spazio dei latenti
#
# Requirements:
#   pip install encodec einops soundfile

import os
import torch
import numpy as np
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.nn.utils')


# ============================================================
# FRECHET DISTANCE — calcolo torch numericamente stabile
# ============================================================

def symmetric_psd_matrix_sqrt(m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Radice quadrata principale di una matrice simmetrica positiva semi-definita.
    Usa eigendecomposizione torch (piu stabile di scipy.linalg.sqrtm).
    """
    m = 0.5 * (m + m.T)
    eigvals, eigvecs = torch.linalg.eigh(m)
    eigvals = torch.clamp(eigvals, min=eps)
    return eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T


def compute_frechet_distance(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Frechet Distance tra due Gaussiane multivariate.

    FD(P1, P2) = ||mu1 - mu2||^2 + Tr(Sigma1) + Tr(Sigma2)
                 - 2 * Tr(sqrt(Sigma1^{1/2} Sigma2 Sigma1^{1/2}))
    """
    mu1 = mu1.reshape(-1).double()
    mu2 = mu2.reshape(-1).double()
    sigma1 = sigma1.double()
    sigma2 = sigma2.double()

    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"

    diff = mu1 - mu2

    eye = torch.eye(sigma1.shape[0], dtype=sigma1.dtype, device=sigma1.device)
    sigma1 = sigma1 + eps * eye
    sigma2 = sigma2 + eps * eye

    sqrt_sigma1 = symmetric_psd_matrix_sqrt(sigma1, eps)
    middle = sqrt_sigma1 @ sigma2 @ sqrt_sigma1
    middle = 0.5 * (middle + middle.T)
    covmean = symmetric_psd_matrix_sqrt(middle, eps)

    fd = (
        diff @ diff
        + torch.trace(sigma1)
        + torch.trace(sigma2)
        - 2.0 * torch.trace(covmean)
    )
    return fd.to(torch.float32)


def compute_mu_sigma(sum_x: torch.Tensor, sum_xx: torch.Tensor, n) -> tuple:
    """Calcola media e covarianza unbiased da somme accumulate."""
    if isinstance(n, torch.Tensor):
        n = n.item()
    mu = sum_x / n
    sigma = (sum_xx - torch.outer(sum_x, mu)) / (n - 1)
    return mu, sigma, n


# ============================================================
# FD-DAC: FRECHET DISTANCE SUI LATENTI DAC (sostituisce KL)
# ============================================================

@torch.no_grad()
def precompute_fd_dac_reference(
    val_dataset,
    cache_path: Optional[str] = None,
    device: Optional[str] = None,
    batch_accum: int = 50,
) -> dict:
    """
    Pre-calcola mu e sigma (covarianza piena) sullo spazio dei latenti DAC
    per il calcolo della Frechet DAC Distance.

    Online accumulation con sum_x e sum_xx per stabilita numerica.
    I latenti vengono presi normalizzati dal dataset (coerenti con cio
    che il modello generera durante il training).

    Cache: se cache_path e' fornito ed esiste, carica le statistiche.
    Se cache_path e' fornito ma non esiste, calcola e salva.
    Se cache_path e' None, calcola senza salvare.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Cache hit: carica le statistiche gia calcolate
    if cache_path is not None and Path(cache_path).exists():
        print(f"[FD-DAC Reference] Carico cache: {cache_path}")
        stats = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        print(f"[FD-DAC Reference] {stats['n_total']} frame, "
              f"dim={stats['mu'].shape[0]}, "
              f"mu range [{stats['mu'].min():.3f}, {stats['mu'].max():.3f}]")
        return stats

    print(f"[FD-DAC Reference] Accumulazione su {len(val_dataset)} sample "
          f"(device={device}, batch_accum={batch_accum})...")

    sum_x = None
    sum_xx = None
    count = 0
    buffer = []

    for idx in tqdm(range(len(val_dataset)), desc="FD-DAC reference"):
        frames, _ = val_dataset[idx]   # (n_frames, dim) normalizzati
        buffer.append(frames)

        if len(buffer) >= batch_accum or idx == len(val_dataset) - 1:
            batch = torch.cat(buffer, dim=0).to(device=device, dtype=torch.float64)
            buffer = []

            if sum_x is None:
                dim = batch.shape[-1]
                sum_x = torch.zeros(dim, dtype=torch.float64, device=device)
                sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=device)

            sum_x = sum_x + batch.sum(dim=0)
            sum_xx = sum_xx + batch.T @ batch
            count = count + batch.shape[0]

    mu, sigma, _ = compute_mu_sigma(sum_x, sum_xx, count)

    # Sposta su CPU per il salvataggio (8MB per sigma 1024x1024)
    mu_cpu = mu.cpu()
    sigma_cpu = sigma.cpu()

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[FD-DAC Reference] Fatto: {count} frame, "
          f"dim={mu_cpu.shape[0]}, "
          f"mu range [{mu_cpu.min():.3f}, {mu_cpu.max():.3f}]")

    stats = {"mu": mu_cpu, "sigma": sigma_cpu, "n_total": count}

    # Salva cache su disco se path fornito
    if cache_path is not None:
        cache_path_obj = Path(cache_path)
        cache_path_obj.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, str(cache_path_obj))
        print(f"[FD-DAC Reference] Cache salvata in: {cache_path}")

    return stats


def compute_fd_dac(
    generated_latents: torch.Tensor,
    fd_dac_ref_stats: dict,
    device: str = "cuda",
) -> float:
    """
    Calcola Frechet DAC Distance tra latenti generati e reference.

    generated_latents: (N, n_frames, dim) tensor di latenti generati
                       (gia normalizzati come escono dal modello)
    fd_dac_ref_stats: dict con 'mu' e 'sigma' del reference
    """
    # (N, n_frames, dim) -> (N*n_frames, dim)
    gen_flat = generated_latents.reshape(-1, generated_latents.shape[-1])
    gen_flat = gen_flat.to(device=device, dtype=torch.float64)

    n = gen_flat.shape[0]
    sum_x = gen_flat.sum(dim=0)
    sum_xx = gen_flat.T @ gen_flat

    mu_gen, sigma_gen, _ = compute_mu_sigma(sum_x, sum_xx, n)

    mu_ref = fd_dac_ref_stats["mu"].to(device)
    sigma_ref = fd_dac_ref_stats["sigma"].to(device)

    fd = compute_frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
    return float(fd.item())


# ============================================================
# FAD CALCULATOR con Encodec
# ============================================================

class FADCalculator:
    """
    FAD con encoder Encodec (codec audio neurale di Facebook).

    Workflow:
      1. precompute_reference_stats(...): prepara WAV reference, calcola embedding
         Encodec, calcola e cacha mu_ref, sigma_ref su file (.pt)
      2. compute_fad_against_reference(...): calcola embedding dei generati
         e restituisce FAD contro reference
    """

    def __init__(self, device: str = "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device

        self.model = None
        self.embedding_dim = None
        self.audio_sr = None
        self.encodec_sr = None

        self.ref_dir = None
        self.ref_stats_path = None
        self.ref_n_samples = 0
        self.mu_ref = None
        self.sigma_ref = None

    def _load_model(self, audio_sr: int = 44100):
        """Carica Encodec, lazy."""
        if self.model is not None:
            return

        from encodec import EncodecModel

        if audio_sr == 48000:
            self.model = EncodecModel.encodec_model_48khz()
            self.encodec_sr = 48000
        else:
            self.model = EncodecModel.encodec_model_24khz()
            self.encodec_sr = 24000

        self.model.set_target_bandwidth(24.0)
        self.embedding_dim = self.model.encoder.dimension
        self.audio_sr = audio_sr

        self.model.to(self.device)
        self.model.eval()

        print(f"[FAD/Encodec] Modello caricato: {self.encodec_sr/1000:.0f}kHz, "
              f"embedding_dim={self.embedding_dim}, device={self.device}")

    @torch.no_grad()
    def _get_embeddings(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Calcola gli embedding Encodec.

        wav: (T,) o (1,T) o (B,T) o (B,1,T)
        return: (N, embedding_dim) dove N = batch * n_frames
        """
        import torchaudio

        # Normalizza shape a (B, 1, T)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0).unsqueeze(0)
        elif wav.dim() == 2:
            wav = wav.unsqueeze(1)
        elif wav.dim() == 3 and wav.shape[1] != 1:
            wav = wav.mean(dim=1, keepdim=True)

        wav = wav.to(self.device).float()

        # Resampling
        if self.audio_sr != self.encodec_sr:
            wav = torchaudio.functional.resample(
                wav, orig_freq=self.audio_sr, new_freq=self.encodec_sr,
            )

        # Encodec 48kHz richiede stereo
        if self.encodec_sr == 48000 and wav.shape[1] != 2:
            wav = torch.cat([wav, wav], dim=1)

        # Embedding pre-quantizzazione
        emb = self.model.encoder(wav)  # (B, D, N_frames)

        from einops import rearrange
        emb = rearrange(emb, "b d n -> (b n) d")
        return emb

    @torch.no_grad()
    def precompute_reference_stats(
        self,
        val_dataset,
        normalizer,
        wav_root: str,
        latent_root: str,
        sr: int = 44100,
        cache_dir: Optional[str] = None,
    ):
        """
        Prepara WAV reference + calcola e cacha mu_ref, sigma_ref.

        cache_dir: directory dove salvare WAV reference e statistiche Encodec.
                   Se None, deve essere passato dal training.
        """
        import soundfile as sf

        if cache_dir is None:
            raise ValueError(
                "cache_dir e' obbligatorio. Passalo dal training script."
            )

        self._load_model(audio_sr=sr)

        n_samples = len(val_dataset)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        ref_dir = cache_dir / "reference_wavs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        self.ref_dir = str(ref_dir)
        self.ref_stats_path = str(cache_dir / "ref_stats_encodec.pt")

        # Cache hit: carica statistiche gia calcolate
        if Path(self.ref_stats_path).exists():
            print(f"[FAD Reference] Carico statistiche cache: {self.ref_stats_path}")
            stats = torch.load(self.ref_stats_path, map_location="cpu", weights_only=False)
            self.mu_ref = stats["mu"].to(self.device)
            self.sigma_ref = stats["sigma"].to(self.device)
            self.ref_n_samples = stats.get("n_samples", n_samples)
            print(f"[FAD Reference] {self.ref_n_samples} sample, "
                  f"dim={self.mu_ref.shape[0]}")
            return

        print(f"[FAD Reference] Preparazione {n_samples} WAV di validazione...")

        # Verifica disponibilita WAV
        need_dac = False
        wav_paths = []
        for idx in range(n_samples):
            npy_path, start, label_idx = val_dataset.samples[idx]
            rel_path = npy_path.relative_to(Path(latent_root))
            wav_path = Path(wav_root) / rel_path.with_suffix(".wav")
            wav_paths.append(wav_path)
            if not wav_path.exists():
                need_dac = True

        dac_model = None
        if need_dac:
            import dac
            dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
            dac_model.to("cpu")
            dac_model.eval()
            print("[FAD Reference] Alcuni WAV mancanti, uso fallback DAC")

        # Copia/genera i WAV reference
        for idx in tqdm(range(n_samples), desc="FAD ref: preparazione WAV"):
            out_path = ref_dir / f"ref_{idx:05d}.wav"
            if out_path.exists():
                continue

            wav_path = wav_paths[idx]
            if wav_path.exists():
                audio, file_sr = sf.read(str(wav_path), dtype='float32')
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                sf.write(str(out_path), audio, sr)
            else:
                frames, _ = val_dataset[idx]
                z = frames.T
                z = normalizer.denormalize(z)
                waveform = dac_model.decode(z.unsqueeze(0).float()).squeeze()
                wav_np = waveform.cpu().numpy()
                if wav_np.ndim > 1:
                    wav_np = wav_np.squeeze()
                sf.write(str(out_path), wav_np, sr)

        if dac_model is not None:
            del dac_model

        self.ref_n_samples = n_samples

        # Online accumulation degli embedding Encodec
        print(f"[FAD Reference] Calcolo embedding Encodec su {n_samples} WAV...")

        dim = self.embedding_dim
        sum_x = torch.zeros(dim, dtype=torch.float64, device=self.device)
        sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        count = 0

        for idx in tqdm(range(n_samples), desc="FAD ref: embedding"):
            wav_file = ref_dir / f"ref_{idx:05d}.wav"
            audio, _ = sf.read(str(wav_file), dtype='float32')
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            wav_t = torch.from_numpy(audio)
            emb = self._get_embeddings(wav_t).double()
            sum_x = sum_x + emb.sum(dim=0)
            sum_xx = sum_xx + emb.T @ emb
            count = count + emb.shape[0]

        # Calcola e salva
        mu_ref, sigma_ref, _ = compute_mu_sigma(sum_x, sum_xx, count)
        self.mu_ref = mu_ref
        self.sigma_ref = sigma_ref

        torch.save({
            "mu": mu_ref.cpu(),
            "sigma": sigma_ref.cpu(),
            "n_samples": n_samples,
            "n_embeddings": count,
            "embedding_dim": dim,
        }, self.ref_stats_path)
        print(f"[FAD Reference] Cache salvata in: {self.ref_stats_path}")

    @torch.no_grad()
    def compute_fad_against_reference(
        self,
        generated_wavs: List[torch.Tensor],
        sr: int = 44100,
    ) -> float:
        """
        Calcola FAD dei generati contro le statistiche reference cachate.
        """
        assert self.mu_ref is not None and self.sigma_ref is not None, \
            "Chiama precompute_reference_stats() prima"

        if self.model is None:
            self._load_model(audio_sr=sr)

        dim = self.embedding_dim
        sum_x = torch.zeros(dim, dtype=torch.float64, device=self.device)
        sum_xx = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        count = 0

        for wav in generated_wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() > 1:
                wav = wav.squeeze()
            emb = self._get_embeddings(wav.float()).double()
            sum_x = sum_x + emb.sum(dim=0)
            sum_xx = sum_xx + emb.T @ emb
            count = count + emb.shape[0]

        if count < 2:
            print("[FAD] Troppi pochi embedding per calcolare la covarianza")
            return float("nan")

        mu_gen, sigma_gen, _ = compute_mu_sigma(sum_x, sum_xx, count)

        fad = compute_frechet_distance(
            self.mu_ref, self.sigma_ref, mu_gen, sigma_gen,
        )
        return float(fad.item())

    @torch.no_grad()
    def compute_fad(
        self,
        real_wavs: List[torch.Tensor],
        generated_wavs: List[torch.Tensor],
        sr: int = 44100,
    ) -> float:
        """FAD standalone senza reference pre-calcolato."""
        if self.model is None:
            self._load_model(audio_sr=sr)

        dim = self.embedding_dim

        # Reali
        sum_x_r = torch.zeros(dim, dtype=torch.float64, device=self.device)
        sum_xx_r = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        count_r = 0
        for wav in real_wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() > 1:
                wav = wav.squeeze()
            emb = self._get_embeddings(wav.float()).double()
            sum_x_r = sum_x_r + emb.sum(dim=0)
            sum_xx_r = sum_xx_r + emb.T @ emb
            count_r = count_r + emb.shape[0]

        # Generati
        sum_x_g = torch.zeros(dim, dtype=torch.float64, device=self.device)
        sum_xx_g = torch.zeros(dim, dim, dtype=torch.float64, device=self.device)
        count_g = 0
        for wav in generated_wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() > 1:
                wav = wav.squeeze()
            emb = self._get_embeddings(wav.float()).double()
            sum_x_g = sum_x_g + emb.sum(dim=0)
            sum_xx_g = sum_xx_g + emb.T @ emb
            count_g = count_g + emb.shape[0]

        mu_r, sigma_r, _ = compute_mu_sigma(sum_x_r, sum_xx_r, count_r)
        mu_g, sigma_g, _ = compute_mu_sigma(sum_x_g, sum_xx_g, count_g)

        fad = compute_frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
        return float(fad.item())


# ============================================================
# EVALUATION FUNCTION (per il training loop)
# ============================================================

@torch.no_grad()
def evaluate_generation(
    model,
    normalizer,
    val_dataset,
    n_samples: int = 64,
    euler_steps: int = 50,
    device: str = "cuda",
    use_amp: bool = True,
    fad_calculator: FADCalculator = None,
    fd_dac_ref_stats: dict = None,
) -> dict:
    """
    Genera N sample e calcola FD-DAC + FAD contro reference pre-calcolati.
    """
    from audio_dataset_npy import DAC_SAMPLE_RATE

    model.eval()
    n_frames = val_dataset.n_frames
    token_dim = 1024
    T_MIN, T_MAX = 0.001, 0.999

    # Genera N sample
    generated_latents_list = []
    for i in range(n_samples):
        x = torch.randn(1, n_frames, token_dim, device=device)
        dt = (T_MAX - T_MIN) / euler_steps
        for s in range(euler_steps):
            t_val = T_MIN + s * dt
            t = torch.ones(1, device=device) * t_val
            with torch.amp.autocast('cuda', enabled=use_amp):
                v = model(x, t)
            x = x + v.float() * dt
        gen_frames = x[0].cpu()
        generated_latents_list.append(gen_frames)

    generated_latents = torch.stack(generated_latents_list)

    # FD-DAC: distanza di Frechet sui latenti DAC (covarianza piena)
    fd_dac = None
    if fd_dac_ref_stats is not None:
        fd_dac = compute_fd_dac(generated_latents, fd_dac_ref_stats, device=device)

    # Decodifica con DAC per calcolare FAD
    import dac
    dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
    dac_model.to("cpu")
    dac_model.eval()

    generated_wavs = []
    for gen_frames in generated_latents_list:
        z = gen_frames.T
        z = normalizer.denormalize(z)
        wav = dac_model.decode(z.unsqueeze(0).float()).squeeze()
        generated_wavs.append(wav.cpu())

    del dac_model

    # FAD
    fad = None
    if fad_calculator is not None and fad_calculator.mu_ref is not None:
        fad = fad_calculator.compute_fad_against_reference(
            generated_wavs, sr=DAC_SAMPLE_RATE,
        )

    return {
        "fd_dac": fd_dac,
        "fad": fad,
        "generated_wavs": generated_wavs,
        "generated_latents": generated_latents,
    }


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    print("Test compute_frechet_distance...")
    d = 128
    torch.manual_seed(0)
    mu1 = torch.randn(d).double()
    A = torch.randn(d, d).double()
    sigma1 = (A @ A.T) / d
    mu2 = torch.randn(d).double()
    B = torch.randn(d, d).double()
    sigma2 = (B @ B.T) / d
    fd = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"  FD: {fd.item():.4f} (deve essere > 0)")
    assert fd.item() > 0
    print("  OK!\n")

    print("Test compute_fd_dac...")
    n_samples, n_frames, dim = 10, 430, 1024
    gen_latents = torch.randn(n_samples, n_frames, dim)
    ref_stats = {
        "mu": torch.zeros(dim, dtype=torch.float64),
        "sigma": torch.eye(dim, dtype=torch.float64),
        "n_total": 1000,
    }
    fd_dac = compute_fd_dac(gen_latents, ref_stats, device="cpu")
    print(f"  FD-DAC: {fd_dac:.4f}")
    print("  OK!\n")

    print("Test FAD con Encodec...")
    try:
        fad_calc = FADCalculator(device="cpu")
        wav1 = [torch.randn(44100) for _ in range(5)]
        wav2 = [torch.randn(44100) * 0.5 for _ in range(5)]
        fad = fad_calc.compute_fad(wav1, wav2, sr=44100)
        print(f"  FAD (random vs rumore attenuato): {fad:.4f}")
        print("  OK!")
    except ImportError as e:
        print(f"  Dipendenza mancante: {e}")
