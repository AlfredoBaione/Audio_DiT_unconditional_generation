# network.py
#
# Diffusion Transformer (DiT) per audio latenti DAC.
#
# Senza patching: ogni frame DAC è direttamente un token.
#   - Token dim: 1024 (= DAC_LATENT_DIM)
#   - Con DiT-L (hidden=1024): proiezione 1:1, zero compressione
#   - Sequenza per 5s: ~430 token
#
# Input:  (B, n_frames, 1024)  + timestep t
# Output: (B, n_frames, 1024)  → velocity field

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from audio_dataset_npy import DAC_LATENT_DIM, MAX_FRAMES


# ============================================================
# TOKEN DIM = DAC_LATENT_DIM direttamente
# ============================================================
TOKEN_DIM = DAC_LATENT_DIM   # 1024 — nessun patching


# ============================================================
# RoPE
# ============================================================

def precompute_rope_freqs(dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    return torch.outer(positions, freqs)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[1]
    freqs = freqs[:seq_len].to(x.device)
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ============================================================
# TIMESTEP EMBEDDING
# ============================================================

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        x = t[:, None] * freqs[None, :]
        return torch.cat([x.cos(), x.sin()], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))


# ============================================================
# ADAPTIVE LAYER NORM (AdaLN)
# ============================================================

class AdaLN(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, t_emb):
        params = self.adaLN_modulation(t_emb)
        s_a, sc_a, g_a, s_f, sc_f, g_f = params.chunk(6, dim=-1)

        def mod(x_norm, shift, scale):
            return x_norm * (1 + scale[:, None, :]) + shift[:, None, :]

        x_attn = mod(self.norm(x), s_a, sc_a)
        x_ffn  = mod(self.norm(x), s_f, sc_f)
        return x_attn, x_ffn, g_a[:, None, :], g_f[:, None, :]


# ============================================================
# SELF ATTENTION con RoPE
# ============================================================

class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_size // n_heads

        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        freqs = precompute_rope_freqs(self.head_dim, max_seq_len=max_seq_len)
        self.register_buffer("rope_freqs", freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = apply_rope(q, self.rope_freqs)
        k = apply_rope(k, self.rope_freqs)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, S, -1)
        return self.proj(x)


# ============================================================
# FFN (SwiGLU)
# ============================================================

class FFN(nn.Module):
    def __init__(self, hidden_size: int, expansion: int = 4):
        super().__init__()
        inner = hidden_size * expansion
        self.w1 = nn.Linear(hidden_size, inner, bias=False)
        self.w2 = nn.Linear(inner, hidden_size, bias=False)
        self.w3 = nn.Linear(hidden_size, inner, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================
# DIT BLOCK
# ============================================================

class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.adaLN = AdaLN(hidden_size)
        self.attn  = SelfAttention(hidden_size, n_heads, max_seq_len)
        self.ffn   = FFN(hidden_size)

    def forward(self, x, t_emb):
        x_attn, x_ffn, g_a, g_f = self.adaLN(x, t_emb)
        x = x + g_a * self.attn(x_attn)
        x = x + g_f * self.ffn(x_ffn)
        return x


# ============================================================
# AUDIO DIT
# ============================================================

class AudioDiT(nn.Module):
    """
    Diffusion Transformer per audio latenti DAC.
    Ogni frame DAC è un token — nessun patching.

    Configurazioni:
        'S': 6  layers, 512  hidden, 8  heads  → ~30M params
        'B': 12 layers, 768  hidden, 12 heads  → ~90M params
        'L': 24 layers, 1024 hidden, 16 heads  → ~340M params

    Con DiT-L, hidden_size = token_dim = 1024 → proiezione 1:1.
    """

    CONFIGS = {
        'S': dict(n_layers=6,  hidden_size=512,  n_heads=8),
        'B': dict(n_layers=12, hidden_size=768,  n_heads=12),
        'L': dict(n_layers=24, hidden_size=1024, n_heads=16),
    }

    def __init__(
        self,
        token_dim:   int = TOKEN_DIM,
        max_seq_len: int = MAX_FRAMES + 16,
        kind:        str = 'L',
    ):
        super().__init__()
        cfg = self.CONFIGS[kind]
        self.kind       = kind
        self.token_dim  = token_dim
        hidden_size     = cfg['hidden_size']
        n_layers        = cfg['n_layers']
        n_heads         = cfg['n_heads']

        # Proietta token_dim → hidden_size
        # Con DiT-L: 1024 → 1024, è una rotazione/proiezione lineare senza compressione
        self.input_proj = nn.Linear(token_dim, hidden_size, bias=True)

        # Timestep embedder
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, max_seq_len=max_seq_len)
            for _ in range(n_layers)
        ])

        # Final norm + proiezione → token_dim
        self.final_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.output_proj = nn.Linear(hidden_size, token_dim, bias=True)

        # Init output a zero per stabilità
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        n_params = sum(p.numel() for p in self.parameters())
        ratio = token_dim / hidden_size
        print(f"[AudioDiT-{kind}] {n_params/1e6:.1f}M params | "
              f"token_dim={token_dim} | hidden={hidden_size} | "
              f"layers={n_layers} | heads={n_heads} | "
              f"ratio={ratio:.1f}:1 {'(nessuna compressione!)' if ratio <= 1.0 else ''}")

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_frames, token_dim)   ← ogni frame è un token
            t: (B,)                        ← timestep in [0, 1]
        Returns:
            velocity: (B, n_frames, token_dim)
        """
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()

        x = self.input_proj(x)
        t_emb = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, t_emb)

        x = self.final_norm(x)
        x = self.output_proj(x)

        return x


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    B, N = 2, 430   # ~5 secondi
    x = torch.randn(B, N, TOKEN_DIM)
    t = torch.rand(B)

    for kind in ['S', 'B', 'L']:
        model = AudioDiT(kind=kind)
        out = model(x, t)
        print(f"  input {x.shape} → output {out.shape}")
        assert out.shape == x.shape
    print("Test superato!")