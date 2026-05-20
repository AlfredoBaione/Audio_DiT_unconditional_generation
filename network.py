# network.py
#
# Diffusion Transformer (DiT) for DAC audio latents.
#
# Aligned 1:1 with the official DiT implementation
# (facebookresearch/DiT, Peebles & Xie 2023):
#   - Pre-LN with TWO separate LayerNorms (norm1 / norm2)
#   - Sequential flow:
#       x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
#       x = x + gate_mlp * mlp (modulate(norm2(x), shift_mlp, scale_mlp))
#   - AdaLN-Zero modulation (6-chunk: shift/scale/gate for attn + mlp)
#   - FinalLayer with its own AdaLN-Zero (2-chunk: shift/scale, no gate)
#   - Self-attention: qkv_bias=True, proj with bias (timm Attention defaults)
#   - MLP: Linear -> GELU(approximate="tanh") -> Linear (timm Mlp)
#   - Positional encoding: ADDITIVE sin/cos embedding (non-trainable),
#     same scheme as DiT 2D
#   - Initialisation as in the official repo:
#       xavier_uniform_ on every Linear, zero-out adaLN_modulation[-1]
#       of every block AND of the final layer, zero-out final_layer.linear,
#       normal_(std=0.02) on the timestep MLP, sin/cos init on pos_embed.
#
# No patching: every DAC frame is directly a token (1024-dim).




import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from audio_dataset_npy import DAC_LATENT_DIM, MAX_FRAMES
from timm.models.vision_transformer import Attention, Mlp


# ============================================================
# TOKEN DIM = DAC_LATENT_DIM directly
# ============================================================
TOKEN_DIM = DAC_LATENT_DIM   # 1024 - no patching


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
# TIMESTEP EMBEDDING (standard DiT)
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
# DIT BLOCK (1:1 with facebookresearch/DiT)
# ============================================================
class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Structurally identical to the official DiTBlock:
        x = x + gate_msa * attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * mlp (modulate(norm2(x), shift_mlp, scale_mlp))
    norm2 acts on the NEW x (post-attention), not on the input.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop=0.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=drop)
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
# FINAL LAYER (1:1 with facebookresearch/DiT)
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

    Configurations:
        'S':  6 layers,  512 hidden,  8 heads
        'B': 12 layers,  768 hidden, 12 heads
        'L': 24 layers, 1024 hidden, 16 heads
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

        # Additive sin/cos positional embedding, non-trainable. Length is
        # fixed at construction; the forward slices it to the current
        # sequence length. Same scheme as facebookresearch/DiT (which uses
        # 2D sin/cos on patch grids).
        self.pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, hidden_size),
            requires_grad=False,
        )

        # Timestep embedder (sinusoidal + MLP)
        self.t_embedder = TimestepEmbedder(hidden_size)
        

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, n_heads, mlp_ratio=mlp_ratio, drop=drop)
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
          - sin/cos init on pos_embed (frozen)
          - normal_(std=0.02) on the two layers of the timestep MLP
          - zero-out adaLN_modulation[-1] of every block (adaLN-Zero)
          - zero-out adaLN_modulation[-1] and linear of the final layer
        """
        # Basic init: xavier_uniform on every Linear, bias to 0
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Sin/cos init on pos_embed (frozen, no gradient flows back).
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.max_seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

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

        # Token projection + ADDITIVE positional embedding (sliced to T)
        x = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]

        # Conditioning vector (only timestep in unconditional)
        c = self.t_embedder(t)

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        return x
    

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    """
    1D sin/cos positional embedding (the 1D version of the function used in
    facebookresearch/DiT). The DiT repo builds a 2D grid embedding by
    splitting it into two 1D embeddings and concatenating them; for our
    1D audio sequence we use the same primitive directly.

    Args:
        embed_dim: total embedding dimension (must be even).
        length:    number of positions.
    Returns:
        np.ndarray of shape (length, embed_dim), dtype float32.
    """
    assert embed_dim % 2 == 0, "embed_dim must be even"
    pos = np.arange(length, dtype=np.float32)             # (L,)
    omega = np.arange(embed_dim // 2, dtype=np.float32)   # (D/2,)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega                           # (D/2,)
    out = np.einsum('m,d->md', pos, omega)                # (L, D/2)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)     # (L, D)

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