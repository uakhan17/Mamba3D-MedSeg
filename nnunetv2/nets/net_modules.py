import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint
import logging
from einops import rearrange
from nnunetv2.nets.utils import trunc_normal_
from timm.models.layers import DropPath
from timm.models.vision_transformer import _load_weights
import math
import torch.nn.functional as F
from nnunetv2.nets.mamba_MS import Mamba_MS
from mamba_ssm.modules.mamba_simple import Mamba
logger = logging.getLogger(__name__)
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def create_block_MS(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    mamba_ms=True,
    device=None,
    dtype=None
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba_MS, layer_idx=layer_idx, mamba_ms=mamba_ms, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = BlockMa(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
    rand = True
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, rand = rand, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = BlockMa(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32
    )
    block.layer_idx = layer_idx
    return block

def conv3x3x3(in_planes, out_planes, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False, groups=1):
    "3x3x3 convolution with padding"
    if weight_std:
        return Conv3d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, groups=groups)


def Norm_layer(norm_cfg, inplanes):
    if norm_cfg == 'BN':
        out = nn.BatchNorm3d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm3d(inplanes, affine=True)
    elif norm_cfg == 'LN':
        out = nn.LayerNorm(inplanes, eps=1e-6)

    return out


def Activation_layer(activation_cfg, inplace=True):
    if activation_cfg == 'ReLU':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)

    return out

    
def extract_and_switch_patches(volume, patch_size):

    batch_size, channels, D, H, W = volume.shape
    d_step, h_step, w_step = patch_size

    new_shape = (batch_size, channels, D // d_step, d_step, H // h_step, h_step, W // w_step, w_step)
    volume_reshaped = volume.reshape(new_shape)

    permute_order = (0, 1, 2, 4, 5, 3, 6, 7)  
    volume_permuted = volume_reshaped.permute(permute_order)

    final_shape = (batch_size, channels, D // d_step * h_step, H // h_step * d_step, W)
    switched_volume = volume_permuted.reshape(final_shape)

    return switched_volume

def reverse_switch_patches(switched_volume, original_shape, patch_size):
    batch_size, channels, D, H, W = original_shape
    d_step, h_step, w_step = patch_size

    D_patches = D // d_step
    H_patches = H // h_step

    temp_shape = (batch_size, channels, D_patches, H_patches, h_step, d_step, W)
    volume_reshaped = switched_volume.reshape(temp_shape)

    permute_order = (0, 1, 2, 5, 3, 4, 6) 
    volume_permuted_back = volume_reshaped.permute(permute_order)

    final_shape = (batch_size, channels, D, H, W)
    volume_original = volume_permuted_back.reshape(final_shape)

    return volume_original


class Conv3d_wd(nn.Conv3d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1,1), padding=(0,0,0), dilation=(1,1,1), groups=1, bias=False):
        super(Conv3d_wd, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3, 4], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv3d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_cfg, activation_cfg, kernel_size, stride=(1, 1, 1),
                 padding=(0, 0, 0), dilation=(1, 1, 1), bias=False, weight_std=False, groups=1):
        super(Conv3dBlock, self).__init__()
        self.conv = conv3x3x3(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, bias=bias, weight_std=weight_std, groups=groups)
        self.norm = Norm_layer(norm_cfg, out_channels)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.norm(x)
        x = self.nonlin(x)

        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, dim):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, dim):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, dim[0], dim[1], dim[2])
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class BlockTr(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, dim):
        x = x + self.drop_path(self.attn(self.norm1(x), dim))
        x = x + self.drop_path(self.mlp(self.norm2(x), dim))

        return x


class BlockMa(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )# hidden_states torch.Size([2, 9409, 192]) residual.shape torch.Size([2, 9409, 192])
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual # hidden_states.shape torch.Size([2, 9409, 192])

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


# class PatchEmbed_unet(nn.Module):
#     """ Image to Patch Embedding
#     """

#     def __init__(self, norm_cfg='BN', activation_cfg='ReLU', weight_std=False, img_size=[16, 96, 96], patch_size=[16, 16, 16], kernel_size=3, stride=2, padding=1, in_chans=1, embed_dim=768, groups=1, more_conv_en=False):
#         super().__init__()
#         num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1]) * (img_size[2] // patch_size[2])

#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         self.more_conv = more_conv_en

#         self.proj = Conv3dBlock(in_chans, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

#         # if self.more_conv:
#         #     self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=1, padding=padding)
#         if self.more_conv:
#             if groups == in_chans:
#                 self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=1, padding=padding, groups=embed_dim)
#             else:
#                 self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=1, padding=padding)

#     def forward(self, x):
#         B, C, D, H, W = x.shape #torch.Size([2, 32, 48, 96, 96])
        
#         if self.more_conv:
#             x = self.proj1(self.proj(x)).flatten(2).transpose(1, 2)
#         else:
#             x = self.proj(x).flatten(2).transpose(1, 2)

#         return x, (D//self.patch_size[0], H//self.patch_size[1], W//self.patch_size[1])

class PatchEmbed_unet(nn.Module):
    """ Image to Patch Embedding - CORRECTED
    """

    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', weight_std=False, img_size=[16, 96, 96], patch_size=[16, 16, 16], kernel_size=3, stride=2, padding=1, in_chans=1, embed_dim=768, groups=1, more_conv_en=False):
        super().__init__()

        # --- CORRECTED num_patches CALCULATION ---
        # This formula precisely calculates the output shape of the conv layer below
        D_out = math.floor((img_size[0] - kernel_size + 2 * padding) / stride) + 1
        H_out = math.floor((img_size[1] - kernel_size + 2 * padding) / stride) + 1
        W_out = math.floor((img_size[2] - kernel_size + 2 * padding) / stride) + 1
        num_patches = D_out * H_out * W_out
        # --- END OF CORRECTION ---

        self.img_size = img_size
        self.patch_size = patch_size # Note: This patch_size variable is now misleading and not used in the corrected calculation
        self.num_patches = num_patches

        self.more_conv = more_conv_en

        self.proj = Conv3dBlock(in_chans, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        
        if self.more_conv:
            if groups == in_chans:
                self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=1, padding=padding, groups=embed_dim)
            else:
                self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        # The forward pass logic itself was correct, only the init was wrong.
        B, C, D, H, W = x.shape
        
        if self.more_conv:
            x_proj = self.proj1(self.proj(x))
        else:
            x_proj = self.proj(x)
            
        x = x_proj.flatten(2).transpose(1, 2)
        
        # The second return value is not used in your main forward loop, but we'll keep it consistent
        # It's better to return the actual shape from the projected feature map
        out_D, out_H, out_W = x_proj.shape[2:]

        return x, (out_D, out_H, out_W)

class PatchEmbed_merge(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', weight_std=False, kernel_size=3, stride=2, padding=1, in_chans=1, embed_dim=768, groups=1, more_conv_m=False):
        super().__init__()

        self.more_conv = more_conv_m

        self.proj = Conv3dBlock(in_chans, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)

        if self.more_conv:
            self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        B, C, D, H, W = x.shape #torch.Size([2, 32, 48, 96, 96])

        if self.more_conv:
            x = self.proj1(self.proj(x)).flatten(2).transpose(1, 2)
        else:
            x = self.proj(x).flatten(2).transpose(1, 2)

        return x


class PatchEmbed_dec(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, norm_cfg='IN', activation_cfg='LeakyReLU', img_size=[16, 96, 96], patch_size=[16, 16, 16], in_chans=1, embed_dim=768, more_conv_de=False):
        super().__init__()
        num_patches = (img_size[0] * patch_size[0]) * (img_size[1] * patch_size[1]) * (img_size[2] * patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.more_conv = more_conv_de

        self.proj = nn.ConvTranspose3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if self.more_conv:
            self.proj1 = Conv3dBlock(embed_dim, embed_dim, norm_cfg=norm_cfg, activation_cfg=activation_cfg, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        
        if self.more_conv:
            x = self.proj1(self.proj(x)).flatten(2).transpose(1, 2)
        else:
            x = self.proj(x).flatten(2).transpose(1, 2)

        return x, (D*self.patch_size[0], H*self.patch_size[1], W*self.patch_size[1])

class DecoderUpsampleBlock(nn.Module):
    """
    A truly robust decoder block that:
    1. Upsamples using deterministic interpolation.
    2. Matches the channel dimension of the skip connection.
    3. Fuses the two tensors via addition.
    4. Applies a final convolution to learn from the fused features.
    """
    def __init__(self, in_chans, skip_chans, out_chans, norm_cfg='IN', activation_cfg='LeakyReLU', weight_std=False):
        super().__init__()
        
        # Upsampling layer to handle the spatial resizing
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        
        # A 1x1 convolution is the standard way to change the number of channels
        # without affecting spatial dimensions. This will match the skip connection's channels.
        self.conv_channel_matcher = nn.Conv3d(in_chans, skip_chans, kernel_size=1, stride=1, padding=0)
        
        # The final convolution processes the fused tensor. Its input channel size
        # is `skip_chans` because we've added two tensors of that size.
        self.conv_out = Conv3dBlock(skip_chans, out_chans, norm_cfg=norm_cfg, activation_cfg=activation_cfg, 
                                    kernel_size=3, stride=1, padding=1, weight_std=weight_std)

    def forward(self, x, skip_tensor):
        """
        Upsamples x, matches channels, and fuses with the skip_tensor.
        """
        # 1. Upsample the input from the deeper layer
        x = self.upsample(x)
        
        # 2. Match the channel dimension of 'x' to that of 'skip_tensor'
        x = self.conv_channel_matcher(x)
        
        # 3. Ensure spatial dimensions match perfectly (handles any "off-by-one" errors)
        if x.shape[2:] != skip_tensor.shape[2:]:
            x = F.interpolate(x, size=skip_tensor.shape[2:], mode='trilinear', align_corners=False)
            
        # 4. Fuse the tensors. They now have identical shapes.
        x = x + skip_tensor
        
        # 5. Process the fused tensor to get the final output for this stage
        x = self.conv_out(x)
        return x

class VisionMamba_MS(nn.Module):
    def __init__(
            self, 
            depth=24, 
            embed_dim=192,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            mamba_ms=True,
            # video 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block_MS(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    mamba_ms = mamba_ms,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        
        # mamba impl
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params # None None
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)
        return x


class VisionMamba(nn.Module):
    def __init__(
            self, 
            depth=24, 
            embed_dim=192,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            # video 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            rand = True
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba = bimamba,
                    rand = rand,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        
        # mamba impl
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params # None None
                )

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)
        return x


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

