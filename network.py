# network.py
#
# Diffusion Transformer (DiT) for DAC audio latents.
#
# Block structure follows the official DiT implementation 1:1
# (facebookresearch/DiT - Peebles & Xie 2023):
#   - Pre-LN with TWO separate LayerNorms (norm1 / norm2)
#   - Sequential flow: x = x + gate * attn(modulate(norm1(x))),
#                      then x = x + gate * ffn(modulate(norm2(x)))
#   - AdaLN-Zero modulation (6-chunk: shift/scale/gate for attn + ffn)
#   - FinalLayer with its own AdaLN-Zero (2-chunk: shift/scale, no gate)
#   - Initialisation as in the official repo: xavier_uniform_ on Linear,
#     zero-out adaLN_modulation[-1] of every block AND of the final layer,
#     zero-out final_layer.linear, normal_(std=0.02) on the timestep MLP.
#
# Two local design choices intentionally kept from the previous version
# (and verified to be improvements over the 2023 DiT defaults):
#   - SwiGLU FFN instead of GELU MLP  (LLaMA-style, better expressivity)
#   - RoPE inside self-attention      (no additive sin/cos pos embedding)
#
# No patching: every DAC frame is directly a token (1024-dim).
#
# IMPORTANT: this is NOT backward-compatible with the previous network.py
# (no AdaLN class, no final_norm/output_proj, different parameter names).
# Existing checkpoints cannot be loaded into this model; re-training is
# required to take advantage of the fix.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_dataset_npy import DAC_LATENT_DIM, MAX_FRAMES


# ============================================================
# TOKEN DIM = DAC_LATENT_DIM directly
# ============================================================
TOKEN_DIM = DAC_LATENT_DIM   # 1024 - no patching


# ============================================================
# MODULATE
# ============================================================
def modulate(x, shift, scale):
    """
    AdaLN modulation. Identical to facebookresearch/DiT.
    x:     (B, T, D)
    shift: (B, D)
    scale: (B, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================
# RoPE (kept from the previous version)
# ============================================================
def precompute_rope_freqs(dim: int, max_seq_len: int = 4096,
                           theta: float = 10000.0) -> torch.Tensor:
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
# TIMESTEP EMBEDDING (kept from the previous version)
# ============================================================
class TimestepEmbedder(nn.Module):
    """
    Sinusoidal embedding of the (continuous) timestep followed by a
    2-layer MLP. Same structure as in the official DiT.
    """

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
# SELF ATTENTION (with RoPE - replaces the additive sin/cos pos embed)
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
# FFN (SwiGLU - replaces the official GELU MLP)
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
# DIT BLOCK (1:1 with facebookresearch/DiT - adaLN-Zero)
# ============================================================
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.

    Structurally identical to the official DiTBlock:
        x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * ffn (modulate(norm2(x), shift_mlp, scale_mlp))
    Note that norm2 is applied to the NEW x (post-attention), not to the
    input of the block. This is the standard Transformer flow.

    Local choices (kept from the previous codebase):
      - SwiGLU FFN instead of the official GELU MLP
      - RoPE inside SelfAttention instead of additive sin/cos
    """

    def __init__(self, hidden_size: int, n_heads: int,
                  max_seq_len: int = 4096, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = SelfAttention(hidden_size, n_heads, max_seq_len)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ffn   = FFN(hidden_size, expansion=int(mlp_ratio))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        c: (B, D)     conditioning vector (here: timestep embedding)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.ffn (modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================
# FINAL LAYER (1:1 with facebookresearch/DiT)
# ============================================================
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    AdaLN-Zero with only shift + scale (no gate, since there is no
    residual connection here), followed by a linear projection to token_dim.
    """

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear     = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ============================================================
# AUDIO DIT
# ============================================================
class AudioDiT(nn.Module):
    """
    Diffusion Transformer for DAC audio latents.
    Each DAC frame is a token (no patching).

    Configurations:
        'S':  6 layers,  512 hidden,  8 heads
        'B': 12 layers,  768 hidden, 12 heads
        'L': 24 layers, 1024 hidden, 16 heads

    With DiT-L, hidden_size = token_dim = 1024 -> 1:1 projection.
    """

    CONFIGS = {
        'S': dict(n_layers=6,  hidden_size=512,  n_heads=8),
        'B': dict(n_layers=12, hidden_size=768,  n_heads=12),
        'L': dict(n_layers=24, hidden_size=1024, n_heads=16),
    }

    def __init__(
        self,
        token_dim:   int   = TOKEN_DIM,
        max_seq_len: int   = MAX_FRAMES + 16,
        kind:        str   = 'L',
        mlp_ratio:   float = 4.0,
    ):
        super().__init__()
        cfg = self.CONFIGS[kind]
        self.kind      = kind
        self.token_dim = token_dim
        hidden_size    = cfg['hidden_size']
        n_layers       = cfg['n_layers']
        n_heads        = cfg['n_heads']

        # Token projection (no patching: every DAC frame is a token)
        self.input_proj = nn.Linear(token_dim, hidden_size, bias=True)

        # Timestep embedder (sinusoidal + MLP, as in the official DiT)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads,
                     max_seq_len=max_seq_len, mlp_ratio=mlp_ratio)
            for _ in range(n_layers)
        ])

        # Final layer (AdaLN-Zero modulated, as in the official DiT)
        self.final_layer = FinalLayer(hidden_size, token_dim)

        # Initialise weights as in the official DiT
        self.initialize_weights()

        n_params = sum(p.numel() for p in self.parameters())
        ratio = token_dim / hidden_size
        print(f"[AudioDiT-{kind}] {n_params/1e6:.1f}M params | "
              f"token_dim={token_dim} | hidden={hidden_size} | "
              f"layers={n_layers} | heads={n_heads} | "
              f"ratio={ratio:.1f}:1 {'(no compression!)' if ratio <= 1.0 else ''}")

    def initialize_weights(self):
        """
        Initialisation scheme identical to facebookresearch/DiT:
          - xavier_uniform_ on every nn.Linear, bias to 0
          - normal_(std=0.02) on the two layers of the timestep MLP
          - zero-out adaLN_modulation[-1] of every block        (adaLN-Zero)
          - zero-out adaLN_modulation[-1] of the final layer    (adaLN-Zero)
          - zero-out final_layer.linear weight + bias           (zero output)
        """
        # Basic init: xavier_uniform on every Linear, bias to 0
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Timestep MLP: normal init (std=0.02), as in the official DiT
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in every DiT block (adaLN-Zero)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out the final layer modulation + projection (zero output)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_frames, token_dim)   - one token per DAC frame
        t: (B,)                        - timestep in [0, 1]

        Returns:
            velocity field of shape (B, n_frames, token_dim)
        """
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()

        x = self.input_proj(x)
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    B, N = 2, 430   # ~5 seconds

    x = torch.randn(B, N, TOKEN_DIM)
    t = torch.rand(B)

    for kind in ['S', 'B', 'L']:
        model = AudioDiT(kind=kind)
        out = model(x, t)
        print(f"  input {x.shape} -> output {out.shape}")
        assert out.shape == x.shape
    print("Test passed!")
