"""
launch_training.py

Script for launching the training on IRCAM server.
Locks GPU before importing torch, then launches training_npy.py.

This launcher overrides the CONFIG values defined in training_npy.py.
To customize a run, modify the CONFIG block below — training_npy.py
itself does NOT need to be edited.

Naming suggestion: copy this file with descriptive names per run, e.g.
    launch_training_gottan.py
    launch_training_guqin_DiT_B.py
    launch_training_guqin_DiT_L.py
so you remember exactly what was launched where.

Use:
    python launch_training.py [num_gpus]

Default: 1 GPU.
"""

import os
import sys
import fcntl
import platform


# ============================================================
# CONFIG — modify these for the current run
# ============================================================
# Any value left as None will keep the default in training_npy.py.
# Set a value to override the corresponding constant.

# Paths
DATASET_ROOT      = None    # e.g. "/data/anasynth_nonbp/baione/dataset_ready/latents"
WAV_ROOT          = None    # e.g. "/data/anasynth_nonbp/baione/dataset_ready/wav"
NORMALIZER_PATH   = None    # e.g. "/data/anasynth_nonbp/baione/checkpoints_v2/normalizer.pt"
FD_DAC_CACHE_PATH = None
FAD_CACHE_DIR     = None
LOG_DIR           = None    # e.g. "/data/anasynth_nonbp/baione/runs/run_name"
CKPT_DIR          = None    # e.g. "/data/anasynth_nonbp/baione/ckpt_run_name/"

# Training hyperparameters
BATCH_SIZE        = 16    # e.g. 16
GRAD_ACCUM        = None    # e.g. 1
NUM_STEPS         = None    # e.g. 1000000
LR                = None    # e.g. 1e-4
MODEL_KIND        = 'B'    # 'S', 'B' or 'L'
DURATION_S        = None    # e.g. 5.0
USE_AMP           = None    # True / False

# Intervals
VAL_INTERVAL      = None
CKPT_INTERVAL     = None
AUDIO_INTERVAL    = None
METRICS_INTERVAL  = None
N_AUDIO_SAMPLES   = None
N_METRICS_SAMPLES = None
EULER_STEPS       = None

# EMA
EMA_DECAY         = None
EMA_START         = None

# Resume
RESUME_FROM       = None    # e.g. "/data/.../checkpoint_step550000.pt"


# ============================================================
# PARALLEL LOCK
# ============================================================
class ParallelLock:
    def __init__(self, path=None):
        if path is None:
            path = os.path.expanduser("~/.gpu_setup.lock")
        self.path = path
        self.fd = None

    def __enter__(self):
        self.fd = open(self.path, "w")
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        fcntl.flock(self.fd, fcntl.LOCK_UN)
        self.fd.close()


# ============================================================
# GPU LOCKING
# ============================================================
def acquire_gpu_locks(num_devices=1):
    """Gets num_devices GPU using IRCAM lock."""

    if 'torch' in sys.modules:
        raise RuntimeError(
            "torch has been imported BEFORE locking the GPU. "
            "Lock the GPUs before every torch import."
        )

    try:
        import manage_gpus as mgp
    except ImportError:
        raise RuntimeError(
            "did not find manage_gpus. "
            "Available only on IRCAM server."
        )

    if num_devices > 4:
        raise ValueError(
            f"At most 4 GPU supported for each node "
            f"(asked {num_devices})."
        )

    with ParallelLock():
        devices = mgp.retrieve_my_gpu_locks()

        if not devices:
            gpu_ids = mgp.board_ids()
            if gpu_ids is None or len(gpu_ids) == 0:
                raise RuntimeError(
                    f"No GPU available on {platform.node()}."
                )

            for _ in range(num_devices):
                locked_gpu_id = mgp.get_gpu_lock()
                if locked_gpu_id >= 0:
                    devices.append(locked_gpu_id)

            if not devices:
                raise RuntimeError(
                    f"Impossible to obtain a GPU on {platform.node()}."
                )

            if len(devices) < num_devices:
                print(
                    f"[WARN] asked {num_devices} GPU, "
                    f"obtained {len(devices)}.",
                    file=sys.stderr,
                )

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cuda_visible:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES not set after locking."
            )
        cuda_visible = sorted(cuda_visible.split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_visible)

        return devices


# ============================================================
# CONFIG INJECTION
# ============================================================
# We collect every uppercase variable in this module that is not None,
# and inject those values into training_npy.py right before the
# `if __name__ == "__main__":` block runs. The training script keeps
# all its other defaults intact.

def collect_overrides():
    """Collect uppercase non-None CONFIG variables from this module."""
    skip_names = {"CUDA_VISIBLE_DEVICES"}
    overrides = {}
    for name, value in list(globals().items()):
        if not name.isupper():
            continue
        if name.startswith("_"):
            continue
        if name in skip_names:
            continue
        if value is None:
            continue
        # Skip non-config items like classes etc.
        if callable(value) or isinstance(value, type):
            continue
        overrides[name] = value
    return overrides


def run_training_with_overrides(script_path: str, overrides: dict):
    """
    Run training_npy.py in the current process, injecting overrides
    right after the `if __name__ == "__main__":` line so that they
    replace the values defined at module level.
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Training script not found: {script_path}")

    with open(script_path, "r") as f:
        source = f.read()

    sentinel = 'if __name__ == "__main__":'
    if sentinel not in source:
        raise RuntimeError(
            f"Could not find sentinel `{sentinel}` in {script_path}."
        )

    override_lines = ["    # === overrides injected by launch_training.py ==="]
    for name, value in overrides.items():
        override_lines.append(f"    {name} = {value!r}")
    if "BATCH_SIZE" in overrides or "GRAD_ACCUM" in overrides:
        override_lines.append("    EFFECTIVE_BS = BATCH_SIZE * GRAD_ACCUM")
    override_lines.append("    # === end overrides ===\n")
    override_block = "\n".join(override_lines)

    # Insert the override block right AFTER the sentinel line
    new_source = source.replace(
        sentinel,
        sentinel + "\n" + override_block,
        1,
    )

    # Run as __main__ in a fresh globals dict
    code = compile(new_source, script_path, "exec")
    module_globals = {"__name__": "__main__", "__file__": script_path}
    exec(code, module_globals)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    num_gpus = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    overrides = collect_overrides()

    if overrides:
        print(f"[launcher] Overriding training_npy.py defaults with:")
        for k, v in overrides.items():
            print(f"    {k} = {v!r}")
    else:
        print(f"[launcher] No overrides — using all defaults from training_npy.py")
    print()

    print(f"[launcher] Getting {num_gpus} GPU...")
    devices = acquire_gpu_locks(num_devices=num_gpus)
    print(f"[launcher] GPU locked: {devices}")
    print(f"[launcher] CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}")

    print(f"[launcher] Running training...\n")

    run_training_with_overrides("training_npy.py", overrides)
