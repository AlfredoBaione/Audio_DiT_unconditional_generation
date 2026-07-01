# network.py
#
# Diffusion Transformer (DiT) for DAC audio latents.
#
# DiT-style architecture (facebookresearch/DiT, Peebles & Xie 2023) kept 1:1
# with the official implementation EXCEPT for two intentional deviations:
#   - RoPE inside self-attention instead of the additive sin/cos positional
#     embedding                       (Su et al., RoFormer, 2021)
#   - SwiGLU FFN instead of the GELU MLP                 (Shazeer, 2020)
#
# Everything else is identical to the official DiT:
#   - Pre-LN with TWO separate LayerNorms (norm1 / norm2)
#   - Sequential flow:
#       x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
#       x = x + gate_mlp * mlp (modulate(norm2(x), shift_mlp, scale_mlp))
#   - AdaLN-Zero modulation (6-chunk for blocks, 2-chunk for final layer)
#   - Same TimestepEmbedder (sinusoidal + 2-layer MLP)
#   - Same FinalLayer (AdaLN-Zero, shift/scale only, then linear)
#   - Same initialisation scheme (xavier_uniform_ on Linear, adaLN-Zero,
#     zero-out final_layer.linear, normal_(std=0.02) on the timestep MLP)
#
# No patching: every DAC latent frame is directly a token (72-dim, pre-quantizer).
#
# NOTE: removing the additive pos_embed is a direct consequence of
# reintroducing RoPE (position is now encoded inside the attention).
# Keeping both would double-encode position. The get_1d_sincos_pos_embed
# helper and the numpy import are therefore no longer needed.


import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_dataset_npy import DAC_LATENT_DIM, MAX_FRAMES


# ============================================================
# TOKEN DIM = DAC_LATENT_DIM directly
# ============================================================
TOKEN_DIM = DAC_LATENT_DIM   # = DAC_LATENT_DIM (72) - no patching


# ============================================================
# MODULATE (identical to facebookresearch/DiT)
# ============================================================
def modulate(x, shift, scale):
    """
    AdaLN modulation.
    x:     (B, T, D)
    shift: (B, D)
    scale: (B, D)
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ============================================================
# RoPE  (copied 1:1 from transformers/models/llama/modeling_llama.py,
#        rotate-half convention; replaces additive sin/cos)
# ============================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input. Identical to
    transformers.models.llama.modeling_llama.rotate_half."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Identical to transformers.models.llama.modeling_llama.apply_rotary_pos_emb.
    q, k: (B, n_heads, S, head_dim)   cos, sin: (B|1, S, head_dim)
    unsqueeze_dim=1 broadcasts cos/sin over the head axis."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_default_rope_parameters(head_dim: int, theta: float = 10000.0) -> torch.Tensor:
    """inv_freq = 1 / theta^(2i/head_dim), i=0..head_dim/2-1.
    Same as transformers _compute_default_rope_parameters (rope_type='default')."""
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))


# ============================================================
# TIMESTEP EMBEDDING (standard DiT) - UNCHANGED
# ============================================================
class TimestepEmbedder(nn.Module):
    """
    Sinusoidal embedding of the (continuous) timestep followed by a
    2-layer MLP. Same structure as in the official DiT.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# ============================================================
# SELF ATTENTION  (RoPE applied exactly as in HF Llama)
# ============================================================
class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, max_seq_len: int = 4096,
                 theta: float = 10000.0):
        super().__init__()
        assert hidden_size % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = hidden_size // n_heads

        self.qkv  = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # inv_freq buffer, non-persistent like HF (deterministic -> not saved).
        inv_freq = compute_default_rope_parameters(self.head_dim, theta=theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _cos_sin(self, S: int, device, dtype):
        # Mirrors LlamaRotaryEmbedding.forward (rope_type='default'):
        # freqs = positions (outer) inv_freq ; emb = cat(freqs, freqs).
        t = torch.arange(S, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=torch.float32))
        emb = torch.cat((freqs, freqs), dim=-1)              # (S, head_dim)
        return emb.cos().to(dtype)[None], emb.sin().to(dtype)[None]  # (1, S, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        # -> (B, n_heads, S, head_dim), the same layout HF rotates in
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        cos, sin = self._cos_sin(S, x.device, x.dtype)       # (1, S, head_dim)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)          # identical to HF

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, S, -1)
        return self.proj(x)


# ============================================================
# FFN  (verbatim from network_old2.py - SwiGLU, replaces GELU MLP)
# ============================================================
class FFN(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0,
                 multiple_of: int = 256, dropout: float = 0.0):
        super().__init__()
        # SwiGLU ha 3 matrici (w1 gate, w3 up, w2 down) vs 2 di un FFN classico.
        # Per matchare params/FLOPs a un FFN standard a mlp_ratio*hidden, si scala
        # la larghezza di 2/3 (Shazeer 2020), poi si arrotonda a un multiplo per
        # efficienza sulle tensor core (convenzione LLaMA). Vecchio comportamento:
        # inner = hidden*4 (3 matrici a 4x) = +50% di params FFN rispetto al matchato.
        inner = int(2 * (mlp_ratio * hidden_size) / 3)
        inner = multiple_of * ((inner + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(hidden_size, inner, bias=False)   # gate
        self.w3 = nn.Linear(hidden_size, inner, bias=False)   # up
        self.w2 = nn.Linear(inner, hidden_size, bias=False)   # down
        # Two dropouts with the same p, mirroring timm Mlp (drop1 after the
        # activation/gating on the hidden tensor, drop2 after the output proj).
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        h = F.silu(self.w1(x)) * self.w3(x)   # gated hidden  (B, T, inner)
        h = self.drop1(h)
        h = self.w2(h)                         # output proj   (B, T, hidden)
        h = self.drop2(h)
        return h


# ============================================================
# DIT BLOCK (1:1 with facebookresearch/DiT)
# ============================================================
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Structurally identical to the official DiTBlock:
        x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * mlp (modulate(norm2(x), shift_mlp, scale_mlp))
    norm2 acts on the NEW x (post-attention), not on the input.

    Deviations from official DiT (see module docstring):
      - self.attn uses RoPE (SelfAttention) instead of timm Attention
      - self.mlp  is SwiGLU (FFN) instead of timm GELU Mlp
    `drop` is wired ONLY into the FFN (not the attention), matching where the
    previous timm-based network applied it (timm Mlp got `drop`; attention
    kept attn_drop=proj_drop=0). This keeps the same model.drop value
    comparable across the two architectures. Default 0.0 -> inert.
    """

    def __init__(self, hidden_size, num_heads, max_seq_len=4096,
                 mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = SelfAttention(hidden_size, num_heads, max_seq_len=max_seq_len)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp   = FFN(hidden_size, mlp_ratio=mlp_ratio, dropout=drop)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ============================================================
# FINAL LAYER (1:1 with facebookresearch/DiT) - UNCHANGED
# ============================================================
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
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

    Configurations (param counts are post-SwiGLU-2/3-fix, verified):
        'S':   6 layers,  512 hidden,  8 heads  (~30.9M params)   head_dim=64
        'B':  12 layers,  768 hidden, 12 heads  (~129.5M params)  head_dim=64
        'G':  18 layers, 1024 hidden, 16 heads  (~348.1M params)  head_dim=64  <- between B and L
        'L':  24 layers, 1024 hidden, 16 heads  (~463.0M params)  head_dim=64
        'XL': 28 layers, 1152 hidden, 16 heads  (~673.5M params)  head_dim=72  <- official DiT-XL

    The 'G' variant was added as an intermediate size between B and L: it is the
    largest model that still fits training at a small batch size on the 12 GB
    IRCAM GPUs (RTX 4070) in pure fp32, without AMP, gradient checkpointing or
    optimizer-state compression. It shares hidden=1024 / 16 heads with L (same
    head_dim=64, same RoPE setup) and sits at 18 layers, midway between B (12)
    and L (24).

    The 'XL' variant mirrors the official facebookresearch/DiT DiT-XL config
    (depth=28, hidden=1152, 16 heads) for the larger IRCAM GPUs (chichibu 24 GB,
    vacqueyras / A6000 49 GB). NOTE: unlike S/B/G/L it has head_dim=72 (1152/16),
    exactly as in the official DiT-XL — the RoPE setup adapts automatically since
    rope_freqs are built from head_dim. To instead keep the project-wide
    head_dim=64, use 18 heads (1152/18=64) rather than 16.
    """

    CONFIGS = {
        'S':  dict(n_layers=6,  hidden_size=512,  n_heads=8),
        'B':  dict(n_layers=12, hidden_size=768,  n_heads=12),
        'G':  dict(n_layers=18, hidden_size=1024, n_heads=16),
        'L':  dict(n_layers=24, hidden_size=1024, n_heads=16),
        'XL': dict(n_layers=28, hidden_size=1152, n_heads=16),
    }

    def __init__(
        self,
        token_dim:   int   = TOKEN_DIM,
        max_seq_len: int   = MAX_FRAMES + 16,
        kind:        str   = 'L',
        mlp_ratio:   float = 4.0,
        drop:        float = 0.0,
    ):
        super().__init__()
        cfg = self.CONFIGS[kind]
        self.kind        = kind
        self.token_dim   = token_dim
        self.max_seq_len = max_seq_len
        hidden_size      = cfg['hidden_size']
        n_layers         = cfg['n_layers']
        n_heads          = cfg['n_heads']

        # Token projection (no patching: every DAC frame is a token)
        self.input_proj = nn.Linear(token_dim, hidden_size, bias=True)

        # Timestep embedder (sinusoidal + MLP)
        self.t_embedder = TimestepEmbedder(hidden_size)

        # DiT blocks (RoPE handles position inside the attention; max_seq_len
        # is plumbed through so each block can build its rope_freqs buffer)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, max_seq_len=max_seq_len,
                     mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(n_layers)
        ])

        # Final layer (AdaLN-Zero modulated)
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
        Identical to facebookresearch/DiT:
          - xavier_uniform_ on every nn.Linear, bias to 0
          - normal_(std=0.02) on the two layers of the timestep MLP
          - zero-out adaLN_modulation[-1] of every block (adaLN-Zero)
          - zero-out adaLN_modulation[-1] and linear of the final layer
        (No pos_embed init: position is now handled by RoPE.)
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
        x: (B, n_frames, token_dim)   one token per DAC frame
        t: (B,)                        timestep in [0, 1]

        Returns:
            velocity field of shape (B, n_frames, token_dim)
        """
        x = x.to(torch.float32)
        t = t.to(torch.float32).flatten()

        # Token projection (position is injected by RoPE inside attention)
        x = self.input_proj(x)

        # Conditioning vector (only timestep in unconditional)
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

    for kind in ['S', 'B', 'G', 'L', 'XL']:
        model = AudioDiT(kind=kind)
        out = model(x, t)
        print(f"  input {x.shape} -> output {out.shape}")
        assert out.shape == x.shape
    print("Test passed!")
