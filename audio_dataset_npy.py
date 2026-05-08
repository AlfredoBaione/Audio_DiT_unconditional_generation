# audio_dataset_npy.py
#
# Dataset for DAC latents pre-computed in .npy.
# No patching — every frame DAC is a token for the transformer.
#
# Auto-detection of the file .npy lenght:
#   -  30s → 2584 frame → 6 chunk of 5s
#   -  5s  → 431 frame  → 1 chunk of 5s
#   -  10s → 862 frame  → 2 chunk of 5s
#   - ecc.
#
# Expected dataset structure:
#   dataset_root/
#       train/
#           classe_1/   *.npy   ← shape (1024, T), dtype float16
#           classe_2/   *.npy
#       val/
#           ...
#       test/
#           ...

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Optional, Tuple, List


# ============================================================
# COSTANTI
# ============================================================
DAC_SAMPLE_RATE  = 44100
DAC_LATENT_DIM   = 1024
DAC_HOP_LENGTH   = 512
DAC_FRAMES_PER_S = DAC_SAMPLE_RATE / DAC_HOP_LENGTH   # ~86.13

# Upper bound per RoPE precomputation nel network.
# Non limita la lunghezza reale dei file — serve solo per
# preallocare le frequenze di posizione nel transformer.
MAX_FRAMES = 4096

SUPPORTED_EXTS   = {".npy"}

# ============================================================
# LAZY DAC LOADER
# ============================================================
_dac_model = None

def get_dac_model(device: str = "cpu"):
    global _dac_model
    if _dac_model is None:
        try:
            import dac
            _dac_model = dac.DAC.load(dac.utils.download(model_type="44khz"))
            _dac_model.to(device)
            _dac_model.eval()
            print(f"[DAC] Modello caricato su {device}")
        except ImportError:
            raise ImportError("DAC non trovato. Installa con: pip install descript-audio-codec")
    return _dac_model


# ============================================================
# DECODING
# ============================================================
@torch.no_grad()
def decode_latents(z: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    model = get_dac_model(device)
    if z.dim() == 2:
        z = z.unsqueeze(0)
    z = z.to(device)
    waveform = model.decode(z)
    return waveform.squeeze(0)


# ============================================================
# NORMALIZZATORE
# ============================================================

class LatentNormalizer:

    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std:  Optional[torch.Tensor] = None

    def fit_from_chunks(
        self,
        chunks: List[Tuple[Path, int]],
        n_frames: int,
        device: Optional[str] = None,
        batch_accum: int = 50,
    ):
        """
        Calcola mean e std per-canale con Welford parallelo batched.

        Ottimizzazioni vs versione naive:
          - Single-pass (no due passate: uso Welford online)
          - Accelerazione GPU (float64 per stabilità)
          - Cache per evitare letture multiple dello stesso file
          - Accumulo in batch prima di aggiornare le statistiche
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            

        n_chunks = len(chunks)
        print(f"[Normalizer] Welford batched su {n_chunks} chunk "
              f"(device={device}, batch_accum={batch_accum})...")

        from tqdm import tqdm


        mean_acc = None   # (dim, 1) float64
        m2_acc   = None   # (dim, 1) float64
        n_total  = 0

        buffer = []

        for i, (path, start) in enumerate(tqdm(chunks, desc="Normalizer fit")):
            # Leggi solo il chunk necessario, senza cachare (evita OOM)
            z_arr = np.load(str(path), mmap_mode='r')[:, start:start + n_frames]
            z = torch.from_numpy(np.ascontiguousarray(z_arr).astype(np.float32))

            if z.shape[1] != n_frames:
                continue  # skip chunk corti

            buffer.append(z)

            # Flush a batch_accum o fine dataset
            if len(buffer) >= batch_accum or i == n_chunks - 1:
                # Concatena sul tempo: (dim, batch_accum * n_frames)
                batch = torch.cat(buffer, dim=1).to(device=device, dtype=torch.float64)
                buffer = []

                n_new = batch.shape[1]

                if mean_acc is None:
                    dim = batch.shape[0]
                    mean_acc = torch.zeros(dim, 1, dtype=torch.float64, device=device)
                    m2_acc   = torch.zeros(dim, 1, dtype=torch.float64, device=device)

                # Welford parallelo (Chan et al., 1979)
                n_total_new = n_total + n_new
                batch_mean  = batch.mean(dim=1, keepdim=True)
                delta       = batch_mean - mean_acc
                mean_acc    = mean_acc + delta * (n_new / n_total_new)
                batch_m2    = ((batch - batch_mean) ** 2).sum(dim=1, keepdim=True)
                m2_acc      = m2_acc + batch_m2 + (delta ** 2) * (n_total * n_new / n_total_new)
                n_total     = n_total_new

        var = (m2_acc / n_total).float().cpu()
        self.mean = mean_acc.float().cpu()
        self.std  = var.sqrt() + 1e-6

        if device == "cuda":
            torch.cuda.empty_cache()

        print(f"[Normalizer] mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
        print(f"[Normalizer] std range:  [{self.std.min():.3f}, {self.std.max():.3f}]")

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None, "Chiama fit_from_chunks() prima di normalize()"
        return (z - self.mean.to(z.device)) / self.std.to(z.device)

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        assert self.mean is not None
        return z * self.std.to(z.device) + self.mean.to(z.device)

    def save(self, path: str):
        torch.save({"mean": self.mean, "std": self.std}, path)
        print(f"[Normalizer] Salvato in {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.mean = ckpt["mean"]
        self.std  = ckpt["std"]
        print(f"[Normalizer] Caricato da {path}")


# ============================================================
# DATASET
# ============================================================

class AudioLatentDataset(Dataset):
    """
    Dataset che carica chunk da file .npy.
    Auto-rileva la lunghezza dei file.
    Senza patching: ogni frame DAC è un token.

    Se i file sono da 5s (431 frame) e duration_s=5.0:
        → 1 chunk per file, ogni chunk = 430 frame

    Se i file sono da 30s (2584 frame) e duration_s=5.0:
        → 6 chunk per file, ogni chunk = 430 frame
    """

    def __init__(
        self,
        root_dir:   str,
        split:      str   = "train",
        duration_s: float = 5.0,
        normalizer: Optional[LatentNormalizer] = None,
        device:     str   = "cpu",
        preload:    bool  = True,
    ):
        self.root_dir   = Path(root_dir)
        self.split      = split
        self.normalizer = normalizer
        self.duration_s = duration_s
        self.preload    = preload

        # Numero di frame per chunk
        self.n_frames = int(duration_s * DAC_FRAMES_PER_S)

        # Ogni sample: (npy_path, start_frame, label_idx)
        self.samples: List[Tuple[Path, int, int]] = []
        self.label_to_idx: dict = {}
        self.idx_to_label: dict = {}
        self._actual_file_frames = None  # Auto-rilevato
        self._scan_directory()

        # Cache: {npy_path_str: tensor (1024, T)}
        self._cache: dict = {}
        if preload:
            self._preload_all()

        chunks_per_file = self._actual_file_frames // self.n_frames if self._actual_file_frames else "?"
        print(f"[Dataset/{split}] duration_s={duration_s}s → "
              f"n_frames={self.n_frames} (= sequenza token) | "
              f"token_dim={DAC_LATENT_DIM} | "
              f"file_frames={self._actual_file_frames} | "
              f"chunk per file={chunks_per_file} | "
              f"tot samples={len(self.samples)} | "
              f"preload={'ON' if preload else 'OFF'}")

    def _detect_file_frames(self, split_dir: Path) -> int:
        """Rileva il numero di frame dal primo file .npy trovato."""
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            for f in sorted(label_dir.iterdir()):
                if f.suffix.lower() in SUPPORTED_EXTS:
                    # Leggi solo lo shape senza caricare tutto
                    z = np.load(str(f), mmap_mode='r')
                    n_frames = z.shape[1]
                    print(f"[Dataset/{self.split}] Auto-rilevato: {n_frames} frame per file "
                          f"({n_frames / DAC_FRAMES_PER_S:.1f}s) da {f.name}")
                    return n_frames
        raise FileNotFoundError(f"Nessun file .npy trovato in {split_dir}")

    def _scan_directory(self):
        split_dir = self.root_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory non trovata: {split_dir}")

        # Auto-rileva lunghezza file (riferimento dal primo file)
        self._actual_file_frames = self._detect_file_frames(split_dir)

        label_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        self.label_to_idx = {d.name: i for i, d in enumerate(label_dirs)}
        self.idx_to_label = {i: d.name for i, d in enumerate(label_dirs)}

        # Controllo di sicurezza sul file di riferimento
        n_chunks_ref = self._actual_file_frames // self.n_frames
        if n_chunks_ref == 0:
            raise ValueError(
                f"I file hanno {self._actual_file_frames} frame "
                f"({self._actual_file_frames / DAC_FRAMES_PER_S:.1f}s) "
                f"ma duration_s={self.duration_s}s richiede {self.n_frames} frame. "
                f"I file sono troppo corti!"
            )

        # Per ogni file controlliamo la lunghezza reale:
        # - usa mmap_mode='r' per leggere solo lo shape senza caricare in RAM
        # - salta chunk che non hanno abbastanza frame
        n_files_total = 0
        n_files_short = 0
        n_chunks_skipped = 0

        for label_dir in label_dirs:
            label_idx = self.label_to_idx[label_dir.name]
            for f in sorted(label_dir.iterdir()):
                if f.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                n_files_total += 1

                # Leggi solo lo shape (senza caricare il file)
                try:
                    file_frames = np.load(str(f), mmap_mode='r').shape[1]
                except Exception as e:
                    print(f"[WARN] Impossibile leggere {f.name}: {e}")
                    continue

                # Quanti chunk completi entrano in QUESTO file specifico
                n_chunks_file = file_frames // self.n_frames

                if n_chunks_file == 0:
                    n_files_short += 1
                    continue

                for k in range(n_chunks_file):
                    start = k * self.n_frames
                    # Doppio controllo: il chunk deve avere esattamente n_frames
                    if start + self.n_frames <= file_frames:
                        self.samples.append((f, start, label_idx))
                    else:
                        n_chunks_skipped += 1

        if n_files_short > 0:
            print(f"[Dataset/{self.split}] ATTENZIONE: {n_files_short}/{n_files_total} "
                  f"file troppo corti per {self.n_frames} frame → skippati")
        if n_chunks_skipped > 0:
            print(f"[Dataset/{self.split}] ATTENZIONE: {n_chunks_skipped} chunk "
                  f"parziali skippati")

        print(f"[Dataset/{self.split}] {len(self.samples)} chunk totali | "
              f"Labels: {list(self.label_to_idx.keys())}")

    @staticmethod
    def _load_latent_static(npy_path: Path) -> torch.Tensor:
        z = np.load(str(npy_path)).astype(np.float32)
        return torch.from_numpy(z)

    @staticmethod
    def _load_latent_fp16(npy_path: Path) -> torch.Tensor:
        """Carica mantenendo float16 per la cache (metà RAM)."""
        z = np.load(str(npy_path))
        return torch.from_numpy(z.astype(np.float16))

    def _preload_all(self):
        """Carica tutti i file .npy unici in RAM in float16."""
        unique_paths = set(str(p) for p, _, _ in self.samples)
        print(f"[Dataset/{self.split}] Preloading {len(unique_paths)} file in RAM (float16)...")
        from tqdm import tqdm
        for path_str in tqdm(sorted(unique_paths), desc=f"Preload {self.split}"):
            self._cache[path_str] = self._load_latent_fp16(Path(path_str))
        size_gb = sum(t.nelement() * 2 for t in self._cache.values()) / 1e9
        print(f"[Dataset/{self.split}] Preloaded: {size_gb:.2f} GB in RAM")

    def get_chunks_for_normalizer(self) -> List[Tuple[Path, int]]:
        return [(path, start) for path, start, _ in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        npy_path, start, label_idx = self.samples[idx]

        key = str(npy_path)
        if key in self._cache:
            # Slice in fp16, poi converti solo il chunk a fp32
            z = self._cache[key][:, start : start + self.n_frames].float()
        else:
            z = self._load_latent_static(npy_path)
            z = z[:, start : start + self.n_frames]

        # Normalizza
        if self.normalizer is not None:
            z = self.normalizer.normalize(z)

        # Trasponi: (1024, n_frames) → (n_frames, 1024)
        z = z.T

        # Controllo di sicurezza sulla shape finale
        if z.shape[0] != self.n_frames:
            raise RuntimeError(
                f"Sample {npy_path.name} @ start={start}: "
                f"shape attesa ({self.n_frames}, 1024), ottenuta {tuple(z.shape)}. "
                f"Il file ha meno frame del previsto."
            )

        return z, label_idx


# ============================================================
# BUILD DATASETS
# ============================================================

def build_datasets(
    root_dir:        str,
    duration_s:      float = 5.0,
    device:          str   = "cpu",
    normalizer_path: Optional[str] = None,
    preload:         bool  = True,
) -> Tuple[AudioLatentDataset, AudioLatentDataset, LatentNormalizer, dict]:

    normalizer = LatentNormalizer()

    if normalizer_path and Path(normalizer_path).exists():
        normalizer.load(normalizer_path)
    else:
        print("[build_datasets] Calcolo normalizer sul training set...")

        train_raw = AudioLatentDataset(
            root_dir=root_dir,
            split="train",
            duration_s=duration_s,
            normalizer=None,
            device=device,
            preload=False,    # Non serve preload per calcolare il normalizer
        )

        chunks = train_raw.get_chunks_for_normalizer()
        normalizer.fit_from_chunks(chunks, n_frames=train_raw.n_frames)

    train_dataset = AudioLatentDataset(
        root_dir=root_dir,
        split="train",
        duration_s=duration_s,
        normalizer=normalizer,
        device=device,
        preload=preload,
    )

    val_dataset = AudioLatentDataset(
        root_dir=root_dir,
        split="val",
        duration_s=duration_s,
        normalizer=normalizer,
        device=device,
        preload=False,    # Val usato raramente, non serve in RAM
    )

    print(f"[build_datasets] Train: {len(train_dataset)} | Val: {len(val_dataset)} | "
          f"duration_s={duration_s}s → {train_dataset.n_frames} frame/token per chunk")

    return train_dataset, val_dataset, normalizer, train_dataset.label_to_idx


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    import sys

    root       = sys.argv[1] if len(sys.argv) > 1 else "./dataset_npy"
    duration_s = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    norm_path  = sys.argv[3] if len(sys.argv) > 3 else None

    print(f"Test AudioLatentDataset su: {root}")
    print(f"duration_s={duration_s}s | normalizer_path={norm_path}\n")

    train_dataset, val_dataset, normalizer, label_map = build_datasets(
        root_dir=root, duration_s=duration_s,
        normalizer_path=norm_path, preload=False,
    )

    if norm_path is None:
        import os
        os.makedirs("checkpoints_v2", exist_ok=True)
        normalizer.save("checkpoints_v2/normalizer.pt")

    sample, label = train_dataset[0]
    print(f"\nSingle sample:")
    print(f"  shape   : {sample.shape}  (n_frames, token_dim)")
    print(f"  label   : {label} ({train_dataset.idx_to_label[label]})")
    print(f"  Mean    : {sample.mean():.4f}")
    print(f"  Std     : {sample.std():.4f}")
