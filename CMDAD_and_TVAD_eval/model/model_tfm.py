"""
Transformer part modified from OpenAI's CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
Caption module modified from ClipCap: https://colab.research.google.com/drive/1tuoAC5F4sC7qid56Z0ap-stR3rwdk0ZV?usp=sharing#scrollTo=OArDkm_24w4L 

Designed for short video captioning. 
"""

import torch
from torch import nn
from typing import Tuple, List, Union, Optional
from collections import OrderedDict
from torch.nn import LayerNorm
from einops import rearrange, repeat
from einops_exts import rearrange_many
from torch import einsum
import math

def exists(val):
    return val is not None

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock_Step(nn.Module):
    def __init__(self, d_model: int, n_head: int, if_causal: bool):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.if_causal = if_causal

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        return tensor if pos is None else tensor + pos

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, pos: torch.Tensor = None):
        key_padding_mask = key_padding_mask.to(device=x.device) if key_padding_mask is not None else None
        q = k = self.with_pos_embed(x, pos)
        if self.if_causal:
            return self.attn(q, k, x, need_weights=False, key_padding_mask=key_padding_mask, is_causal=self.if_causal, attn_mask=torch.randn(1,1))[0]
        else:
            return self.attn(q, k, x, need_weights=False, key_padding_mask=key_padding_mask, is_causal=self.if_causal)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, pos: torch.Tensor = None):
        x_norm = self.ln_1(x)
        x = x + self.attention(x_norm, key_padding_mask=key_padding_mask, pos=pos)
        x = x + self.mlp(self.ln_2(x))
        return x, x_norm


class TemporalEncoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, if_causal: bool):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock_Step(width, heads, if_causal) for _ in range(layers)])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, pos: torch.Tensor = None):
        intermediate = []
        for block in self.resblocks:
            x, x_norm = block(x, key_padding_mask, pos)
            intermediate.append(x_norm)
        intermediate.pop(0)
        intermediate.append(x)
        return intermediate


class PerceiverEncoder(nn.Module):
    """Perceiver-like module, with TransformerEncoder([latent; features])"""
    def __init__(self, num_latents=16, d_latents=768, nhead=12, num_layers=2):
        super().__init__()
        self.num_latents = num_latents
        self.latent = nn.Parameter(torch.randn(num_latents, d_latents))
        self.temporal_pos_embed = nn.Parameter(torch.randn(58, d_latents))
        self.encoder = TemporalEncoder(width=d_latents, layers=num_layers, heads=nhead, if_causal=False)
        self.visual_prenorm = LayerNorm(d_latents)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.latent, mean=0, std=1)
        nn.init.normal_(self.temporal_pos_embed, mean=0, std=1.0)
        proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
        attn_std = self.encoder.width ** -0.5
        fc_std = (2 * self.encoder.width) ** -0.5
        for block in self.encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, visual_feature, key_padding_mask=None):
        B, T, *_ = visual_feature.shape
        visual_feature = rearrange(visual_feature, 'b t c -> t b c')
        temp_pos = self.temporal_pos_embed[0:T, None, :]
        visual_feature = self.visual_prenorm(visual_feature) + temp_pos
        latent = self.latent[:,None,:].repeat(1,B,1)  # k,b,c
        concat = torch.cat((latent, visual_feature), dim=0)
        enc_out = self.encoder(concat, key_padding_mask, pos=None)[-1]  # last layer output

        latent_out = enc_out[0:self.num_latents, :]
        return rearrange(latent_out, 'k b c -> b k c')
    
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(self, x, media, media_locations=None, use_cached_media=False):
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            use_cached_media: bool
                If true, treat all of x as if they occur after the last media
                registered in media_locations. T_txt does not need to exactly
                equal media_locations.shape[1] in this case
        """

        # if not use_cached_media:
        #     assert (
        #         media_locations.shape[1] == x.shape[1]
        #     ), f"media_location.shape is {media_locations.shape} but x.shape is {x.shape}"

        T_txt = x.shape[1]
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            assert 0
            media_time = torch.arange(T_img, device=x.device) + 1

            if use_cached_media:
                # text time is set to the last cached media location
                text_time = repeat(
                    torch.count_nonzero(media_locations, dim=1),
                    "b -> b i",
                    i=T_txt,
                )
            else:
                # at each boolean of True, increment the time counter (relative to media time)
                text_time = media_locations.cumsum(dim=-1)

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            assert 0
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(
                text_without_media_mask, "b i -> b 1 i 1"
            )
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_visual,
        dim_head=64,
        heads=8,
        ff_mult=4,
        only_attend_immediate_media=True,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))

        self.ff = FeedForward(dim, mult=ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
        self,
        x,
        media,
        media_locations=None,
        use_cached_media=False,
    ):
        x = (
            self.attn(
                x,
                media,
                media_locations=media_locations,
                use_cached_media=use_cached_media,
            )
            * self.attn_gate.tanh()
            + x
        )
        x = self.ff(x) * self.ff_gate.tanh() + x

        return x

# class PositionalEncoding(nn.Module):    # torch PositionalEncoding

#     def __init__(self, d_model: int, max_len: int = 5000):
#         super().__init__()

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(1, max_len, d_model)
#         pe[0, :, 0::2] = torch.sin(position * div_term)
#         pe[0, :, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x, video_length):
#         """
#         Arguments:
#             x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
#         """
#         x[:, :video_length] = x[:, :video_length] + self.pe[0, :video_length]
#         return x


class TFMEncoder(nn.Module):
    """TransformerEncoder([features])"""
    def __init__(self, d_latents=768, nhead=12, num_layers=2, if_causal=False):
        super().__init__()
        self.temporal_pos_embed = nn.Parameter(torch.randn(58, d_latents))
        self.encoder = TemporalEncoder(width=d_latents, layers=num_layers, heads=nhead, if_causal=if_causal)
        self.visual_prenorm = LayerNorm(d_latents)
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.temporal_pos_embed, mean=0, std=1.0)
        proj_std = (self.encoder.width ** -0.5) * ((2 * self.encoder.layers) ** -0.5)
        attn_std = self.encoder.width ** -0.5
        fc_std = (2 * self.encoder.width) ** -0.5
        for block in self.encoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, visual_feature, key_padding_mask=None):
        B, T, *_ = visual_feature.shape
        visual_feature = rearrange(visual_feature, 'b t c -> t b c')
        temp_pos = self.temporal_pos_embed[0:T, None, :]
        visual_feature = self.visual_prenorm(visual_feature) + temp_pos
        enc_out = self.encoder(visual_feature, key_padding_mask, pos=None)[-1]  # last layer output
        return rearrange(enc_out, 'k b c -> b k c')
    

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): visual features
                shape (b, T, D)
            latent (torch.Tensor): latent features
                shape (b, n, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=6,
        dim_head=64,
        heads=8,
        num_latents=64,
        ff_mult=4,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.media_time_embs = nn.Parameter(torch.randn(58, dim))
        self.visual_prenorm = LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, D)
        Returns:
            shape (b, n, D) where n is self.num_latents
        """
        b, T = x.shape[:2]

        # if exists(self.media_time_embs):
        # frame and media time embeddings
        x = self.visual_prenorm(x) + self.media_time_embs[None, :T, :]

        # blocks
        latents = repeat(self.latents, "n d -> b n d", b=b)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        return self.norm(latents)