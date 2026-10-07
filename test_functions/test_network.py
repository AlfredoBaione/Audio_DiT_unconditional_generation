import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from network import AudioDiT, TOKEN_DIM

B, N = 2, 430


def test_output_shape_all_kinds():
    x = torch.randn(B, N, TOKEN_DIM)
    t = torch.rand(B)

    for kind in ['S', 'B', 'G', 'L', 'XL']:
        model = AudioDiT(kind=kind)
        out = model(x, t)
        print(f"  input {x.shape} -> output {out.shape}")
        assert out.shape == x.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(f"{name}...")
            fn()
            print("  OK\n")