"""
preprocess_dataset.py  (v3)

Script unificato di preprocessing per il progetto Audio DiT.

Pipeline:
    Audio grezzi → Preprocessing qualitativo → Chunk → Encoding DAC → Latenti .npy

Ottimizzazioni:
    - I WAV di TRAIN non vengono salvati su disco (solo latenti .npy)
    - I WAV di VAL e TEST vengono salvati (servono per FAD e test)
    - Naming corto: ClassName_seg00001.wav  (niente nomi lunghi → no errori Windows)
    - I file temporanei vengono cancellati appena non servono più
    - Contatore globale per segmento per classe (non per file sorgente)

Output:
    output_dir/
        latents/
            train/ ClassName/ ClassName_seg00001.npy
            val/   ClassName/ ClassName_seg00001.npy
            test/  ClassName/ ClassName_seg00001.npy
        wav/
            val/   ClassName/ ClassName_seg00001.wav   ← solo val e test
            test/  ClassName/ ClassName_seg00001.wav
        metadata.csv
        class_mapping.json

Uso:
    python preprocess_dataset.py source_dir output_dir
    python preprocess_dataset.py source_dir output_dir --chunk_length 5
    python preprocess_dataset.py source_dir output_dir --chunk_length 10 --device cuda
    python preprocess_dataset.py source_dir output_dir --skip_dac
"""

import os
import re
import csv
import json
import math
import random
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter

import numpy as np
import torch


# ============================================================
# GLOBAL CONFIG — impostate da main() prima del multiprocessing
# ============================================================
CHUNK_LENGTH_SEC = 5
SR = 44100
OUTPUT_DIR = "/data/anasynth_nonbp/baione"
TEMP_DIR = ""

# Qualità audio
MIN_CHUNK_SEC = CHUNK_LENGTH_SEC   # accetta solo chunk di durata completa
SILENCE_THRESH_DB = -40.0
SILENCE_TRIM_DB = -35.0
TARGET_LUFS = -14.0
TARGET_TP = -1.0
TARGET_LRA = 11.0

MAX_WORKERS = 2

SUPPORTED_AUDIO_EXTS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".wma", ".mpc", ".oma", ".ape", ".aac",
}


# ============================================================
# SANITIZE
# ============================================================
def sanitize_class_name(name: str) -> str:
    """Sanitizza il nome della classe: solo ASCII alfanumerico + underscore."""
    name = re.sub(r"[^\w\s-]", "", name, flags=re.ASCII)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name if name else "unknown"


def sanitize_filename(name: str) -> str:
    """Sanitizza un nome file: solo ASCII lowercase."""
    name = Path(name).stem
    name = name.lower()
    name = re.sub(r"[^\w\s-]", "", name, flags=re.ASCII)
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name if name else "unknown"


# ============================================================
# FFMPEG / FFPROBE
# ============================================================
def get_audio_duration(file_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error",
             "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(file_path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_chunk_rms_db(file_path: str) -> float:
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(file_path),
             "-af", "volumedetect", "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in (result.stderr or "").splitlines():
            if "mean_volume" in line:
                return float(line.split("mean_volume:")[1].strip().replace(" dB", ""))
        return -100.0
    except Exception:
        return -100.0


# ============================================================
# DETECT SILENCE
# ============================================================
def detect_trim_points(file_path: str, threshold_db: float) -> tuple:
    duration = get_audio_duration(file_path)
    if duration == 0:
        return 0.0, 0.0

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(file_path),
             "-af", f"silencedetect=noise={threshold_db}dB:d=0.1",
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
    except Exception:
        return 0.0, duration

    stderr = result.stderr or ""
    silence_regions = []
    current_start = None

    for line in stderr.splitlines():
        if "silence_start:" in line:
            try:
                current_start = float(line.split("silence_start:")[1].strip().split()[0])
            except (ValueError, IndexError):
                current_start = None
        elif "silence_end:" in line and current_start is not None:
            try:
                end_val = float(line.split("silence_end:")[1].strip().split()[0])
                silence_regions.append((current_start, end_val))
            except (ValueError, IndexError):
                pass
            current_start = None

    if current_start is not None:
        silence_regions.append((current_start, duration))

    if not silence_regions:
        return 0.0, duration

    trim_start = 0.0
    if silence_regions[0][0] < 0.05:
        trim_start = silence_regions[0][1]

    trim_end = duration
    if silence_regions[-1][1] >= duration - 0.05:
        trim_end = silence_regions[-1][0]

    return trim_start, trim_end


# ============================================================
# LOUDNESS ANALYSIS (pass 1)
# ============================================================
def analyze_loudness(file_path: str) -> dict:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(file_path),
             "-af", (f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:"
                     f"LRA={TARGET_LRA}:print_format=json"),
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        stderr = result.stderr or ""
        json_start = stderr.rfind("{")
        json_end = stderr.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            return None
        data = json.loads(stderr[json_start:json_end])
        return {
            "measured_I": data.get("input_i", "-24.0"),
            "measured_TP": data.get("input_tp", "-1.0"),
            "measured_LRA": data.get("input_lra", "7.0"),
            "measured_thresh": data.get("input_thresh", "-34.0"),
        }
    except Exception as e:
        print(f"[WARN] Loudness analysis failed: {e}")
        return None


# ============================================================
# PREPROCESSING: TRIM + NORMALIZE
# ============================================================
def preprocess_file(args: tuple) -> dict:
    """
    args = (file_path, class_name, safe_class_name, file_index)
    Usa le globali TEMP_DIR e SR.
    """
    file_path, class_name, safe_class_name, file_index = args
    file_path = Path(file_path)

    trim_start, trim_end = detect_trim_points(str(file_path), SILENCE_TRIM_DB)
    trimmed_duration = trim_end - trim_start

    if trimmed_duration < MIN_CHUNK_SEC:
        print(f"[SKIP] Troppo corto dopo trim ({trimmed_duration:.1f}s): {file_path.name}")
        return None

    loudness = analyze_loudness(str(file_path))

    filters = []
    if loudness:
        filters.append(
            f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:"
            f"measured_I={loudness['measured_I']}:"
            f"measured_TP={loudness['measured_TP']}:"
            f"measured_LRA={loudness['measured_LRA']}:"
            f"measured_thresh={loudness['measured_thresh']}:"
            f"linear=true"
        )
    else:
        filters.append(f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}")

    # Nome temp corto: class_fileindex.wav
    temp_out = Path(TEMP_DIR) / f"{safe_class_name}_{file_index:05d}.wav"

    command = [
        "ffmpeg", "-y", "-hide_banner",
        "-ss", str(trim_start),
        "-i", str(file_path),
        "-t", str(trimmed_duration),
        "-af", ",".join(filters),
        "-ar", str(SR), "-ac", "1",
        "-loglevel", "error",
        str(temp_out),
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        stderr_msg = (e.stderr.decode("utf-8", errors="replace").strip()) if e.stderr else ""
        print(f"[ERROR] Preprocessing failed: {file_path.name}: {stderr_msg}")
        return None

    actual_duration = get_audio_duration(str(temp_out))
    if actual_duration < MIN_CHUNK_SEC:
        temp_out.unlink(missing_ok=True)
        return None

    return {
        "temp_path":       str(temp_out),
        "source_name":     file_path.name,
        "class_name":      class_name,
        "safe_class_name": safe_class_name,
        "duration":        actual_duration,
    }


# ============================================================
# RACCOLTA FILE SORGENTE
# ============================================================
def get_audio_files(source_dir: str) -> list:
    tasks = []
    file_index = 0
    for class_name in sorted(os.listdir(source_dir)):
        class_path = Path(source_dir) / class_name
        if not class_path.is_dir():
            continue
        safe_class = sanitize_class_name(class_name)
        for file_path in sorted(class_path.iterdir()):
            if file_path.suffix.lower() in SUPPORTED_AUDIO_EXTS:
                tasks.append((file_path, class_name, safe_class, file_index))
                file_index += 1
    return tasks


# ============================================================
# PIANIFICAZIONE CHUNK + SPLIT + NAMING
# ============================================================
def plan_and_assign_chunks(preprocessed_files: list, split_ratios: dict, seed: int) -> list:
    """
    Pianifica chunk, assegna split, genera nomi corti.
    Naming: ClassName_seg00001  (contatore globale per classe)
    """
    rng = random.Random(seed)
    thresholds = (split_ratios["train"], split_ratios["train"] + split_ratios["val"])

    # Contatore per classe per generare nomi unici
    class_counters = Counter()
    all_chunks = []

    for pf in preprocessed_files:
        n_chunks = math.ceil(pf["duration"] / CHUNK_LENGTH_SEC)
        safe_class = pf["safe_class_name"]

        for i in range(n_chunks):
            start_sec = i * CHUNK_LENGTH_SEC
            actual_duration = min(CHUNK_LENGTH_SEC, pf["duration"] - start_sec)

            if actual_duration < MIN_CHUNK_SEC:
                continue

            # Assegna split
            r = rng.random()
            if r < thresholds[0]:
                split = "train"
            elif r < thresholds[1]:
                split = "val"
            else:
                split = "test"

            # Nome corto con contatore globale per classe
            class_counters[safe_class] += 1
            seg_num = class_counters[safe_class]
            short_name = f"{safe_class}_seg{seg_num:05d}"

            # Calcola i path qui (processo principale) — non nei worker
            if split in ("val", "test"):
                wav_out = str(Path(OUTPUT_DIR) / "wav" / split / safe_class / f"{short_name}.wav")
            else:
                wav_out = str(Path(TEMP_DIR) / f"{short_name}.wav")

            all_chunks.append({
                "temp_path":       pf["temp_path"],
                "source_name":     pf["source_name"],
                "class_name":      pf["class_name"],
                "safe_class_name": safe_class,
                "seg_index":       i,
                "start_sec":       round(start_sec, 3),
                "duration_sec":    round(actual_duration, 3),
                "split":           split,
                "short_name":      short_name,
                "wav_path":        wav_out,
            })

    return all_chunks


# ============================================================
# EXTRACT CHUNK TO WAV (in memoria o su disco)
# ============================================================
def extract_chunk_to_file(chunk: dict, out_path: str) -> bool:
    """Estrae un chunk dal file preprocessato e lo salva come WAV."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg", "-y", "-hide_banner",
        "-i", chunk["temp_path"],
        "-ss", str(chunk["start_sec"]),
        "-t", str(CHUNK_LENGTH_SEC),
        "-c:a", "pcm_s16le",
        "-ar", str(SR), "-ac", "1",
        "-loglevel", "error",
        str(out_path),
    ]

    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        return False

    # Quality check
    rms_db = get_chunk_rms_db(str(out_path))
    if rms_db < SILENCE_THRESH_DB:
        Path(out_path).unlink(missing_ok=True)
        return False

    return True


# ============================================================
# PROCESS CHUNK: salva WAV (solo val/test) + encode DAC
# Questa funzione è chiamata dal Pool
# ============================================================
def process_chunk(chunk: dict) -> dict:
    """
    Per ogni chunk:
    - Estrae il WAV nel path già calcolato (wav_path nel dict)
    - Quality check (scarta chunk silenziosi)
    """
    wav_path = chunk["wav_path"]

    # Estrai chunk
    ok = extract_chunk_to_file(chunk, wav_path)
    if not ok:
        return None

    chunk["rms_db"] = round(get_chunk_rms_db(wav_path), 2)

    return chunk


# ============================================================
# ENCODING DAC (sequenziale, su GPU)
# ============================================================
def encode_chunks_dac(saved_chunks: list, device: str, output_dir: str):
    """Encoda chunk WAV → latenti .npy. Cancella WAV di train dopo encoding."""
    try:
        import dac
    except ImportError:
        print("[ERROR] DAC non installato. pip install descript-audio-codec")
        return

    import soundfile as sf

    print(f"\n[DAC] Caricamento modello su {device}...")
    model_path = dac.utils.download(model_type="44khz")
    dac_model = dac.DAC.load(model_path)
    dac_model.to(device)
    dac_model.eval()
    print(f"[DAC] Modello caricato.\n")

    n_ok = 0
    n_err = 0

    for chunk in tqdm(saved_chunks, desc="Encoding DAC"):
        wav_path = chunk["wav_path"]
        safe_class = chunk["safe_class_name"]
        short_name = chunk["short_name"]
        split = chunk["split"]

        # Path latente di output
        latent_path = Path(output_dir) / "latents" / split / safe_class / f"{short_name}.npy"
        latent_path.parent.mkdir(parents=True, exist_ok=True)

        if latent_path.exists():
            # Già encodato, pulisci WAV train se esiste
            if split == "train" and Path(wav_path).exists():
                Path(wav_path).unlink(missing_ok=True)
            n_ok += 1
            continue

        try:
            audio, sr = sf.read(str(wav_path), always_2d=True)
            waveform = torch.from_numpy(audio.T).float()

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            waveform = waveform.unsqueeze(0).to(device)

            with torch.no_grad():
                x = dac_model.preprocess(waveform, sr)
                z, codes, latents, _, _ = dac_model.encode(x)

            latents_np = z.squeeze(0).cpu().numpy().astype(np.float16)
            np.save(str(latent_path), latents_np)
            n_ok += 1

        except Exception as e:
            print(f"[DAC ERROR] {short_name}: {e}")
            n_err += 1

        # Cancella WAV di train (non serve più)
        if split == "train" and Path(wav_path).exists():
            Path(wav_path).unlink(missing_ok=True)

    del dac_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n[DAC] Completato: {n_ok} OK, {n_err} errori")


# ============================================================
# METADATA CSV
# ============================================================
def write_metadata(chunks: list, output_path: str):
    fieldnames = [
        "short_name", "split", "class_name", "safe_class_name",
        "source_name", "seg_index", "start_sec", "duration_sec", "rms_db",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in sorted(chunks, key=lambda x: (x["safe_class_name"], x["short_name"])):
            writer.writerow({k: c.get(k, "") for k in fieldnames})


# ============================================================
# MAIN
# ============================================================
def main():
    global CHUNK_LENGTH_SEC, SR, OUTPUT_DIR, TEMP_DIR, MAX_WORKERS, MIN_CHUNK_SEC

    parser = argparse.ArgumentParser(
        description="Preprocessing unificato: audio grezzi → latenti DAC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Esempi:
    python preprocess_dataset.py ./raw_music ./dataset_ready
    python preprocess_dataset.py ./raw_music ./dataset_ready --chunk_length 10
    python preprocess_dataset.py ./raw_music ./dataset_ready --device cuda
    python preprocess_dataset.py ./raw_music ./dataset_ready --skip_dac
        """,
    )
    parser.add_argument("source_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--chunk_length", type=float, default=5)
    parser.add_argument("--sr", type=int, default=44100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_dac", action="store_true")
    parser.add_argument("--max_workers", type=int, default=2)
    parser.add_argument("--split_train", type=float, default=0.8)
    parser.add_argument("--split_val", type=float, default=0.1)
    parser.add_argument("--split_test", type=float, default=0.1)

    args = parser.parse_args()

    # Imposta globali
    CHUNK_LENGTH_SEC = args.chunk_length
    MIN_CHUNK_SEC = CHUNK_LENGTH_SEC   # solo chunk completi
    SR = args.sr
    OUTPUT_DIR = args.output_dir
    TEMP_DIR = tempfile.mkdtemp(prefix="audio_preprocess_", dir="/data/anasynth_nonbp/baione")
    MAX_WORKERS = args.max_workers

    split_ratios = {"train": args.split_train, "val": args.split_val, "test": args.split_test}
    assert abs(sum(split_ratios.values()) - 1.0) < 1e-6

    n_cores = cpu_count()

    print(f"{'='*60}")
    print(f"PREPROCESSING DATASET AUDIO (v3)")
    print(f"{'='*60}")
    print(f"  Sorgente:       {args.source_dir}")
    print(f"  Output:         {OUTPUT_DIR}")
    print(f"  Chunk length:   {CHUNK_LENGTH_SEC}s")
    print(f"  Sample rate:    {SR} Hz")
    print(f"  Split:          {split_ratios}")
    print(f"  DAC:            {'SI' if not args.skip_dac else 'SKIP'} ({args.device})")
    print(f"  Workers:        {MAX_WORKERS} (preprocess), {n_cores} (chunks)")
    print(f"  WAV salvati:    solo val + test (train → solo latenti)")
    print(f"{'='*60}\n")

    # ── 1. SCANSIONE ──
    print("Fase 1/5 — Scansione file sorgente...")
    tasks = get_audio_files(args.source_dir)
    if not tasks:
        print(f"[ERROR] Nessun file audio trovato in {args.source_dir}")
        return

    class_mapping = {}
    for _, orig, safe, _ in tasks:
        if orig not in class_mapping:
            class_mapping[orig] = safe

    print(f"  {len(tasks)} file in {len(class_mapping)} classi:")
    for orig, safe in sorted(class_mapping.items()):
        print(f"    {orig} → {safe}")
    print()

    # ── 2. PREPROCESSING ──
    print(f"Fase 2/5 — Preprocessing con {MAX_WORKERS} worker...")
    preprocess_args = [(str(fp), cn, sc, fi) for fp, cn, sc, fi in tasks]

    preprocessed = []
    with Pool(MAX_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(preprocess_file, preprocess_args),
                           total=len(preprocess_args), desc="Preprocessing"):
            if result is not None:
                preprocessed.append(result)

    print(f"  Preprocessati: {len(preprocessed)}/{len(tasks)}\n")
    if not preprocessed:
        print("[ERROR] Nessun file preprocessato!")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return

    # ── 3. CHUNK + SPLIT ──
    print("Fase 3/5 — Pianificazione chunk + split...")
    all_chunks = plan_and_assign_chunks(preprocessed, split_ratios, args.seed)

    split_counts = Counter(c["split"] for c in all_chunks)
    print(f"  Chunk pianificati: {len(all_chunks)}")
    for s, n in sorted(split_counts.items()):
        print(f"    {s}: {n}")
    print()

    # ── 4. ESTRAZIONE CHUNK WAV ──
    print(f"Fase 4/5 — Estrazione chunk con {n_cores} worker...")
    saved_chunks = []
    skipped = 0

    with Pool(n_cores) as pool:
        for result in tqdm(pool.imap_unordered(process_chunk, all_chunks),
                           total=len(all_chunks), desc="Extracting chunks"):
            if result is not None:
                saved_chunks.append(result)
            else:
                skipped += 1

    # Ora possiamo cancellare i file preprocessati (non servono più)
    for pf in preprocessed:
        Path(pf["temp_path"]).unlink(missing_ok=True)

    # Metadata
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata_path = str(Path(OUTPUT_DIR) / "metadata.csv")
    write_metadata(saved_chunks, metadata_path)

    mapping_path = str(Path(OUTPUT_DIR) / "class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)

    print(f"  Chunk salvati: {len(saved_chunks)}, scartati: {skipped}\n")

    # ── 5. ENCODING DAC ──
    if not args.skip_dac:
        print("Fase 5/5 — Encoding DAC...")
        encode_chunks_dac(saved_chunks, args.device, args.output_dir)
    else:
        print("Fase 5/5 — Encoding DAC SALTATO\n")

    # Pulizia finale
    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # ── RIEPILOGO ──
    final_splits = Counter(c["split"] for c in saved_chunks)
    n_train_wav = sum(1 for c in saved_chunks if c["split"] == "train")
    n_valtest_wav = sum(1 for c in saved_chunks if c["split"] in ("val", "test"))

    print(f"\n{'='*60}")
    print(f"COMPLETATO!")
    print(f"{'='*60}")
    print(f"  File sorgente:    {len(tasks)}")
    print(f"  Preprocessati:    {len(preprocessed)}")
    print(f"  Chunk totali:     {len(saved_chunks)}")
    print(f"  Chunk scartati:   {skipped}")
    print(f"  Latenti .npy:     {len(saved_chunks)} (tutti gli split)")
    print(f"  WAV su disco:     {n_valtest_wav} (solo val+test)")
    print(f"  WAV NON salvati:  {n_train_wav} (train → solo latenti)")

    print(f"\n  Distribuzione:")
    for s, n in sorted(final_splits.items()):
        pct = 100 * n / len(saved_chunks) if saved_chunks else 0
        print(f"    {s}: {n:>6} ({pct:.1f}%)")

    print(f"\n  Output:")
    print(f"    {OUTPUT_DIR}/latents/train|val|test/<class>/*.npy")
    print(f"    {OUTPUT_DIR}/wav/val|test/<class>/*.wav")

    print(f"\n  Per il training:")
    print(f'    DATASET_ROOT = "{OUTPUT_DIR}/latents"')
    print(f'    WAV_ROOT     = "{OUTPUT_DIR}/wav"')


if __name__ == "__main__":
    main()
