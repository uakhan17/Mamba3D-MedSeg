import torch
import torch.nn as nn
from functools import partial
import logging
from timm.models.layers import DropPath
from timm.models.vision_transformer import _load_weights
from nnunetv2.nets.utils import trunc_normal_
logger = logging.getLogger(__name__)

import nnunetv2.nets.net_modules as net_modules
from nnunetv2.nets.mambaBlock.mamba_3d_ms_tri import MambaConfig_ms, Mamba_ms
from nnunetv2.nets.mambaBlock.mamba_3d_tri import MambaConfig, Mamba


class Mamba_encoder(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', weight_std=False, img_size=[48, 192, 192], in_chans=1, 
                drop_rate=0., num_classes=0, embed_dims=[64,192,384,384],bimamba = False, rand = False, depths=[2,3,4,3],
                attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, 
                qk_scale=None, sr_ratios=[6, 4, 2, 1], sc=['M','M','M','M']):

        super().__init__()

        self.embed_dims = embed_dims
        self.stage_change = sc

        # Encoder patchEmbed
        self.patch_embed0 = net_modules.Conv3dBlock(in_chans, 32, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=7, stride=(1, 2, 2), padding=3)
        self.patch_embed1 = net_modules.PatchEmbed_unet(norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, img_size=[img_size[0]//1, img_size[1]//2, img_size[2]//2], patch_size=[2, 2, 2], in_chans=32, embed_dim=embed_dims[0])
        self.patch_embed2 = net_modules.PatchEmbed_unet(norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, img_size=[img_size[0]//2, img_size[1]//4, img_size[2]//4], patch_size=[2, 2, 2], in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = net_modules.PatchEmbed_unet(norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, img_size=[img_size[0]//4, img_size[1]//8, img_size[2]//8], patch_size=[2, 2, 2], in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = net_modules.PatchEmbed_unet(norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, img_size=[img_size[0]//8, img_size[1]//16, img_size[2]//16], patch_size=[2, 2, 2], in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # pos_embed
        pos_embeds = []
        pos_drops = []
        for i in range(4):
            pos_embed = nn.Parameter(torch.zeros(1, getattr(self, f'patch_embed{i+1}').num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)
            pos_embeds.append(pos_embed)
            pos_drops.append(pos_drop)

        self.pos_embed1, self.pos_embed2, self.pos_embed3, self.pos_embed4 = pos_embeds
        self.pos_drop1, self.pos_drop2, self.pos_drop3, self.pos_drop4 = pos_drops


        # define blocks
        common_params_transformer = {
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'drop': drop_rate,
            'attn_drop': attn_drop_rate,
            'norm_layer': norm_layer,
        }

        common_params_mamba = {
            'use_cuda': True,
            'tridirectional': True,
        }

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        blocks_all = []

        for i in range(4):
            if sc[i] == 'T':
                block = nn.ModuleList([
                    net_modules.BlockTr(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        sr_ratio=sr_ratios[i],
                        **common_params_transformer
                    )
                    for j in range(depths[i])
                ])

            else:
                block = Mamba_ms(MambaConfig_ms(d_model=embed_dims[i], n_layers=depths[i], **common_params_mamba))

            blocks_all.append(block)
            cur += depths[i]

        self.block1, self.block2, self.block3, self.block4 = blocks_all

        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv3d, net_modules.Conv3d_wd)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        out = []
        B = x.shape[0] 
        x = self.patch_embed0(x)
        out.append(x)

        patch_embeds = [self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4]
        pos_embeds = [self.pos_embed1, self.pos_embed2, self.pos_embed3, self.pos_embed4]
        pos_drops = [self.pos_drop1, self.pos_drop2, self.pos_drop3, self.pos_drop4]
        blocks = [self.block1, self.block2, self.block3, self.block4]

        for i in range(4):
            x, (D, H, W) = patch_embeds[i](x)
            x = x + pos_embeds[i]
            # print("shape of input",x.shape)
            # print("\n ************************ \n")
            # print("shape of pos_embed", pos_embeds[i].shape)
            x = pos_drops[i](x)
            if self.stage_change[i]=='T':
                for blk in blocks[i]:
                    x = blk(x, (D, H, W))
            else:
                x = blocks[i](x, (D, H, W))
            if i < 3:  # Only reshape and permute for stages 1, 2, and 3
                x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
            out.append(x)

        x = self.head(x)
        out[-1] = x  # Replace the last appended item with the head output

        return out, (D, H, W)


class vm_seg(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', weight_std=False, img_size=[48, 192, 192], num_classes=None, embed_dims=[192,64,32],
                 norm_layer=nn.LayerNorm,in_chans =1, drop_rate=0., bimamba = False, rand = False, debi = False, 
                 derand = False, deep_supervision=False, depths=[3, 4, 3], depths_en=[2,3,4,3],
                 attn_drop_rate=0., drop_path_rate=0., num_heads=[8,4,2], mlp_ratios=[4, 4, 4], qkv_bias=False, 
                 qk_scale=None, sr_ratios=[2, 4, 6], sc_en=['M','M','M','M'], sc=['M','M','M']):


        super().__init__()
        self.MODEL_NUM_CLASSES = num_classes
        self.embed_dims = embed_dims
        self.deep_supervision = deep_supervision
        self.stage_change = sc

        # Encoder
        self.transformer = Mamba_encoder(img_size=img_size, norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std,in_chans=in_chans, 
                                        bimamba=bimamba,rand=rand, embed_dims=[48,128,256,512], depths=depths_en, 
                                        num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], sr_ratios=[6, 4, 2, 1], qkv_bias=True, 
                                        norm_layer=partial(nn.LayerNorm, eps=1e-6), sc=sc_en)

        total = sum([param.nelement() for param in self.transformer.parameters()])
        print('  + Number of Transformer Params: %.2f(e6)' % (total / 1e6))

        # upsampling
        self.upsamplex122 = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        # self.DecEmbed0 = net_modules.PatchEmbed_dec(img_size=[img_size[0]//8, img_size[1]//16, img_size[2]//16], patch_size=[2, 2, 2], in_chans=512, embed_dim=embed_dims[0])
        # self.DecEmbed1 = net_modules.PatchEmbed_dec(img_size=[img_size[0]//4, img_size[1]//8, img_size[2]//8], patch_size=[2, 2, 2], in_chans=embed_dims[0], embed_dim=embed_dims[1])
        # self.DecEmbed2 = net_modules.PatchEmbed_dec(img_size=[img_size[0]//2, img_size[1]//4, img_size[2]//4], patch_size=[2, 2, 2], in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.DecEmbed0 = net_modules.DecoderUpsampleBlock(in_chans=512, skip_chans=embed_dims[0],out_chans=embed_dims[0])
        self.DecEmbed1 = net_modules.DecoderUpsampleBlock(in_chans=embed_dims[0], skip_chans=embed_dims[1],out_chans=embed_dims[1])
        self.DecEmbed2 = net_modules.DecoderUpsampleBlock(in_chans=embed_dims[1], skip_chans=embed_dims[2],out_chans=embed_dims[2])
        # Decoder patchEmbed
        dec_pos_embeds = []
        dec_pos_drops = []
        patch_embed_order = [3, 2, 1]  # The order in which the patch embeds are accessed

        for i in range(3):
            dec_pos_embed = nn.Parameter(torch.zeros(1, getattr(self.transformer, f'patch_embed{patch_embed_order[i]}').num_patches, embed_dims[i]))
            dec_pos_drop = nn.Dropout(p=drop_rate)
            dec_pos_embeds.append(dec_pos_embed)
            dec_pos_drops.append(dec_pos_drop)

        self.DecPosEmbed0, self.DecPosEmbed1, self.DecPosEmbed2 = dec_pos_embeds
        self.DecPosDrop0, self.DecPosDrop1, self.DecPosDrop2 = dec_pos_drops

        self.pos_drop = nn.Dropout(p=0.1)

        # Decoder transformer
        common_params_transformer = {
            'qkv_bias': qkv_bias,
            'qk_scale': qk_scale,
            'drop': drop_rate,
            'attn_drop': attn_drop_rate,
            'norm_layer': norm_layer,
        }

        common_params_mamba = {
            'use_cuda': True,
            'tridirectional': True,
        }

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        blocks_all = []

        for i in range(3):
            if sc[i] == 'T':
                block = nn.ModuleList([
                    net_modules.BlockTr(
                        dim=embed_dims[i],
                        num_heads=num_heads[i],
                        mlp_ratio=mlp_ratios[i],
                        drop_path=dpr[cur + j],
                        sr_ratio=sr_ratios[i],
                        **common_params_transformer
                    )
                    for j in range(depths[i])
                ])

            else:
                block = Mamba(MambaConfig(d_model=embed_dims[i], n_layers=depths[i], **common_params_mamba))

            blocks_all.append(block)
            cur += depths[i]

        self.Decblock0, self.Decblock1, self.Decblock2 = blocks_all

        self.norm = norm_layer(embed_dims[2])

        self.transposeconv_stage3 = nn.ConvTranspose3d(embed_dims[2], embed_dims[3], kernel_size=2, stride=2)
        self.stage3_de = net_modules.Conv3dBlock(embed_dims[3], embed_dims[3], norm_cfg=norm_cfg, activation_cfg=activation_cfg, weight_std=weight_std, kernel_size=3, stride=1, padding=1)

        # Seg head
        self.ds0_cls_conv = nn.Conv3d(embed_dims[0], self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds1_cls_conv = nn.Conv3d(embed_dims[1], self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds2_cls_conv = nn.Conv3d(embed_dims[2], self.MODEL_NUM_CLASSES, kernel_size=1)
        self.ds3_cls_conv = nn.Conv3d(embed_dims[3], self.MODEL_NUM_CLASSES, kernel_size=1)

        self.cls_conv = nn.Conv3d(embed_dims[3], self.MODEL_NUM_CLASSES, kernel_size=1)

        trunc_normal_(self.DecPosEmbed0, std=.02)
        trunc_normal_(self.DecPosEmbed1, std=.02)
        trunc_normal_(self.DecPosEmbed2, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv3d, net_modules.Conv3d_wd)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        B = inputs.shape[0]
        
        ####### encoder
        x_encoder, (D, H, W) = self.transformer(inputs) 
        x_trans = x_encoder[-1].reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        
        ####### decoder
        dec_embeds = [self.DecEmbed0, self.DecEmbed1, self.DecEmbed2]
        dec_pos_embeds = [self.DecPosEmbed0, self.DecPosEmbed1, self.DecPosEmbed2]
        dec_pos_drops = [self.DecPosDrop0, self.DecPosDrop1, self.DecPosDrop2]
        dec_blocks = [self.Decblock0, self.Decblock1, self.Decblock2]
        skips = [-2, -3, -4]
        ds_cls_convs = [self.ds0_cls_conv, self.ds1_cls_conv, self.ds2_cls_conv]

        x = x_trans
        ds_outputs = []
        
        # for i in range(3):
        #     x, (D, H, W) = dec_embeds[i](x)
        #     skip = x_encoder[skips[i]]
        #     x = x + skip.flatten(2).transpose(1, 2)
        #     x = x + dec_pos_embeds[i]
        #     x = dec_pos_drops[i](x)

        #     if self.stage_change[i]=='T':
        #         for blk in dec_blocks[i]:
        #             x = blk(x, (D, H, W))
        #     else:
        #         x = dec_blocks[i](x, (D, H, W))

        #     if i == 2:
        #         x = self.norm(x)
        #     x = x.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        #     ds_outputs.append(ds_cls_convs[i](x))
        for i in range(3):
            skip = x_encoder[skips[i]]
            x = dec_embeds[i](x, skip)
            D, H, W = x.shape[2:]
            x_flat = x.flatten(2).transpose(1, 2)

            x_flat = x_flat + dec_pos_embeds[i]
            x_flat = dec_pos_drops[i](x_flat)

            if self.stage_change[i]=='T':
                for blk in dec_blocks[i]:
                    x_flat = blk(x_flat, (D, H, W))
            else:
                x_flat = dec_blocks[i](x_flat, (D, H, W))

            # 6. Reshape back to a 5D tensor for the next decoder stage.
            if i == 2:
                x_flat = self.norm(x_flat)
            x = x_flat.reshape(B, D, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
            ds_outputs.append(ds_cls_convs[i](x))
        # stage 3
        x = self.transposeconv_stage3(x)
        skip3 = x_encoder[-5]
        x = x + skip3
        x = self.stage3_de(x)
        ds3 = self.ds3_cls_conv(x)
        
        x = self.upsamplex122(x)
        result = self.cls_conv(x)

        if not self.deep_supervision:
            r = result
        else:
            r = [result, ds3] + ds_outputs[::-1]

        return r


class Mamba_segnet(nn.Module):
    def __init__(self, norm_cfg='BN', activation_cfg='ReLU', weight_std=False, img_size=None, num_classes=None, in_chans=1, deep_supervision=False, 
                bimamba=False, rand = False, debi = False, derand = False, 
                sc_en=['M','M','M','M'], sc=['M','M','M'], depths=[3,4,3], depths_en=[2, 3, 4, 3]):
        super().__init__()
        
        self.model = vm_seg(norm_cfg, activation_cfg, weight_std, img_size, num_classes, embed_dims=[256,128,48,32], in_chans=in_chans, bimamba=bimamba, rand=rand, debi=debi, derand=derand, deep_supervision=deep_supervision, sc_en=sc_en, sc=sc, depths=depths, depths_en=depths_en)

        total = sum([param.nelement() for param in self.model.parameters()])
        print('  + Number of Network Params: %.2f(e6)' % (total / 1e6))

        if weight_std==False:
            self.conv_op = nn.Conv3d
        else:
            self.conv_op = net_modules.Conv3d_wd
        if norm_cfg == 'BN':
            self.norm_op = nn.BatchNorm3d
        if norm_cfg == 'SyncBN':
            self.norm_op = nn.SyncBatchNorm
        if norm_cfg == 'GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg == 'IN':
            self.norm_op = nn.InstanceNorm3d
        self.dropout_op = nn.Dropout3d
        self.num_classes = num_classes

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv3d, net_modules.Conv3d_wd)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d, nn.SyncBatchNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        seg_output = self.model(x)
        return seg_output

