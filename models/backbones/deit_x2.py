import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from functools import partial
from einops import rearrange

from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
try:
    from timm.layers import DropPath, Mlp, trunc_normal_
except ImportError:  # timm < 1.0
    from timm.models.layers import DropPath, Mlp, trunc_normal_


# ============================================================
# Utils
# ============================================================

def resize_pos_embed_distilled(weight, img_size, patch_size=16):
    """
    weight: [1, 2+old_grid, D]  (CLS + DIST + patches)
    img_size: (H,W)
    """
    token_embed = weight[:, :2, :]
    grid_embed = weight[:, 2:, :]

    old_grid = int(np.sqrt(grid_embed.shape[1]))
    new_grid = (img_size[0] // patch_size, img_size[1] // patch_size)

    grid_embed = grid_embed.reshape(1, old_grid, old_grid, weight.shape[-1]).permute(0, 3, 1, 2)
    resize = torchvision.transforms.Resize(new_grid)
    grid_embed = resize(grid_embed)
    grid_embed = grid_embed.permute(0, 2, 3, 1).reshape(1, -1, weight.shape[-1])

    return torch.cat([token_embed, grid_embed], dim=1)


def load_pl_ckpt(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def extract_submodule_state(state_dict, prefix):
    """
    Extract all keys starting with prefix and strip prefix.
    Example:
      prefix="encoder_a."
      key="encoder_a.model.pos_embed" -> "model.pos_embed"
    """
    out = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            nk = k[len(prefix):]
            out[nk] = v
    return out


def strip_prefixes(state_dict, prefixes):
    out = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        out[nk] = v
    return out


# ============================================================
# Stage1 DeiT backbone (Distilled)
# ============================================================

class DistilledVisionTransformer(VisionTransformer):
    """
    GAReT style DeiT distilled:
    output embedding = (head(cls)+head_dist(dist))/2
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))

        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x, **kwargs):
        x_cls, x_dist = self.forward_features(x)
        x_cls = self.head(x_cls)
        x_dist = self.head_dist(x_dist)
        return (x_cls + x_dist) / 2


# ============================================================
# GeoAdapter building blocks (Stage2)
# ============================================================

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.inv_act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden)
        self.D_inv_fcb1 = nn.Parameter(torch.zeros(D_features))
        self.D_fc2 = nn.Linear(D_hidden, D_features)
        self.D_inv_fcb2 = nn.Parameter(torch.zeros(D_hidden))
        
        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.D_fc2.weight, 0)
        nn.init.constant_(self.D_fc2.bias, 0)

    def forward(self, x):
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        return xs

    def forward_inv(self, x):
        xs = F.linear(x, self.D_fc2.weight.T, self.D_inv_fcb2)
        xs = self.inv_act(xs)
        xs = F.linear(xs, self.D_fc1.weight.T, self.D_inv_fcb1)
        
        return xs


class Attention(nn.Module):
    def __init__(
        self, dim, num_heads=8, qkv_bias=False,
        attn_drop=0., proj_drop=0., qk_scale=None
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        if mask is not None:
            attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GeoAdapterBlock(nn.Module):
    def __init__(
        self, dim, num_frames, num_heads, mlp_ratio=4.,
        qkv_bias=True, drop=0., attn_drop=0.,
        drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_frames = num_frames

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.scale = 0.5
        self.T_Adapter = Adapter(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden, act_layer=act_layer, drop=drop)

    def forward(self, x, temporal_embed, mask, T):
        # x: (BT, N, D)
        bt, n, d = x.shape
        B = bt // T

        # spatial attention
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # temporal attention (token-wise)
        xt = rearrange(x, "(b t) n d -> (b n) t d", b=B, t=T)
        xt = xt + temporal_embed

        xt = self.T_Adapter(xt)
        xt = self.T_Adapter.forward_inv(self.attn(self.norm1(xt), mask=mask))

        xt = rearrange(xt, "(b n) t d -> (b t) n d", b=B, t=T, n=n)
        x = x + self.drop_path(xt)

        # MLP
        xn = self.norm2(x)
        x = x + self.mlp(xn)
        return x

class DeitGeoAdapter(nn.Module):
    """
    GAReT Stage2 model (DeitAdapter).
    Input: (B,T,C,H,W)
    Output: (B, dim)
    """
    def __init__(
        self,
        num_frames=8,
        aerial=False,
        img_size=(224, 224),
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_classes=512,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_frames = num_frames
        self.aerial = aerial
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.temporal_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            GeoAdapterBlock(
                dim=embed_dim,
                num_frames=num_frames,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer
            )
            for i in range(depth)
        ])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.dist_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.temporal_embedding, std=0.02)

        # 仿照第二段代码的 init_weights 逻辑
        for n, m in self.named_modules():
            if 'T_Adapter' in n:
                for n2, m2 in m.named_modules():
                    if 'D_fc2' in n2:
                        if isinstance(m2, nn.Linear):
                            nn.init.constant_(m2.weight, 0)
                            nn.init.constant_(m2.bias, 0)
                            
    def forward(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        temporal_embed = self.temporal_embedding[:, :T, :]

        x = rearrange(x, "b t c h w -> (b t) c h w")

        x = self.patch_embed(x)  # (BT, HW, D)

        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        dist_tokens = self.dist_token.expand(B * T, -1, -1)

        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)  # (BT, HW+2, D)
        x = x + self.pos_embed.to(x.dtype)
        x = self.pos_drop(x)


        n = x.shape[1]
        mask = None

        if self.aerial:
            m = ~torch.eye(T, dtype=torch.bool, device=x.device)
            m = m.repeat(n, 1, 1).repeat(B, 1, 1, 1)
            m[:, 0:2, :, :] = False
            m = rearrange(m, "b n t1 t2 -> (b n) t1 t2")
            mask = m.unsqueeze(1)

        for blk in self.blocks:
            x = blk(x, temporal_embed, mask, T)

        x = self.norm(x)

        x_cls = self.head(x[:, 0])
        x_dist = self.head_dist(x[:, 1])
        x = (x_cls + x_dist) / 2

        x = rearrange(x, "(b t) c -> b t c", b=B, t=T)

        x_g = x.mean(dim=1)

        return x_g, x

    def freeze_layers(self, missing_keys):
        for n, p in self.named_parameters():
            if n not in set(missing_keys):
                p.requires_grad = False


# ============================================================
# Factory
# ============================================================

def build_deit_small_garet(
    adapter=False,
    aerial=False,
    num_frames=8,
    pretrained=True,
    img_size=(224, 224),
    dim=512,
    resume=None,
    ckpt_tower=None,          # "a" or "b"
    freeze_backbone=True,
):
    """
    ckpt_tower:
      - None: normal pretrained init
      - "a": load encoder_a from resume checkpoint
      - "b": load encoder_b from resume checkpoint
    """
    if adapter:
        model = DeitGeoAdapter(
            num_frames=num_frames,
            aerial=aerial,
            img_size=img_size,
            num_classes=dim
        )
    else:
        model = DistilledVisionTransformer(
            img_size=img_size,
            patch_size=16,
            embed_dim=384,
            num_classes=dim,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        model.default_cfg = _cfg()

    missing_keys = []

    # --- ImageNet DeiT init ---
    if pretrained:
        ckpt = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu",
            check_hash=True
        )
        state = ckpt["model"]

        # resize pos_embed
        state["pos_embed"] = resize_pos_embed_distilled(state["pos_embed"], img_size=img_size, patch_size=16)

        # resize heads (GAReT trick)
        if dim != 1000:
            state["head.weight"] = state["head.weight"].repeat(5, 1)[:dim, :]
            state["head.bias"] = state["head.bias"].repeat(5)[:dim]
            state["head_dist.weight"] = state["head.weight"].repeat(1, 1)[:dim, :]
            state["head_dist.bias"] = state["head.bias"].repeat(1)[:dim]

        msg = model.load_state_dict(state, strict=False)
        assert len(msg.unexpected_keys) == 0, f"unexpected keys: {msg.unexpected_keys}"
        missing_keys = list(msg.missing_keys)

    # --- Step1 checkpoint init (two-tower) ---
    if resume is not None and os.path.isfile(resume):
        full_state = load_pl_ckpt(resume)

        # Lightning checkpoint will contain something like:
        # encoder_a.xxx or encoder_b.xxx
        if ckpt_tower == "a":
            tower_state = extract_submodule_state(full_state, "encoder_a.")
        elif ckpt_tower == "b":
            tower_state = extract_submodule_state(full_state, "encoder_b.")
        else:
            raise ValueError("ckpt_tower must be 'a' or 'b' when resume is provided")

        # In your code, encoder_a is a wrapper; might be encoder_a.model.xxx
        tower_state = strip_prefixes(tower_state, prefixes=[
            "model.",
            "backbone.",
        ])

        msg2 = model.load_state_dict(tower_state, strict=False)
        missing_keys = sorted(set(missing_keys + list(msg2.missing_keys)))

        if freeze_backbone and hasattr(model, "freeze_layers"):
            model.freeze_layers(missing_keys)

        print(f"[GAReT] Loaded Step1 ckpt tower={ckpt_tower} from {resume}")
        print(f"[GAReT] Missing keys after load: {len(missing_keys)}")

    return model, missing_keys


# ============================================================
# Wrapper class for helper.get_backbone
# ============================================================

class DeiTWithX2Agg(nn.Module):
    """
    Usage for Stage2:
      DeiT(adapter=True, ckpt_path=step1.ckpt, ckpt_tower="a"/"b")
    """
    def __init__(
        self,
        model_name="deit_small",
        img_size=(224, 224),
        dim=512,
        adapter=False,
        aerial=False,
        num_frames=8,
        pretrained=True,
        ckpt_path=None,
        ckpt_tower=None,   # "a" or "b"
        freeze_backbone=True,
    ):
        super().__init__()
        assert model_name == "deit_small"

        self.model, self.missing_keys = build_deit_small_garet(
            adapter=adapter,
            aerial=aerial,
            num_frames=num_frames,
            pretrained=pretrained,
            img_size=img_size,
            dim=dim,
            resume=ckpt_path,
            ckpt_tower=ckpt_tower,
            freeze_backbone=freeze_backbone,
        )
        self.num_channels = dim

    def forward(self, x):
        return self.model(x)
