import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
class LocalizationNetwork(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_clip_224.laion2b',pretrained=True, crop_size=(224,224)):
        super().__init__()
        self.backbone = timm.create_model(model_name=backbone_name, pretrained=pretrained, num_classes=0)
        if hasattr(self.backbone, 'embed_dim'):
            self.embed_dim = self.backbone.embed_dim
        else:
            print(f"Model {backbone_name} does not have a direct 'embed_dim' attribute.")
        print(f"[{self.__class__.__name__}] Initialize ViT ({backbone_name} with embedding dim of {self.embed_dim})")
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 4) #  sx, sy, tx, ty
        )

        self.mlp[2].weight.data.zero_()
        self.mlp[2].bias.data.copy_(torch.Tensor([1,1,0,0]))

    def forward(self, x):
        features = self.backbone(x) # x: [B, 3, 224, 224] -> features: [B, 197, 768]?
        theta_params = self.mlp(features)
        return theta_params # [s_x, s_y, t_x, t_y]

class AFC(nn.Module):
    def __init__(self, crop_size=(224,224)):
        super().__init__()
        self.localization_networks = LocalizationNetwork()
        self.crop_size = crop_size

    def forward(self, x_wide):
        B = x_wide.size(0) # x_wide: [B, C, H, W]
        C = x_wide.size(1)
        grid_size = torch.Size((B, C, self.crop_size[0], self.crop_size[1]))
        theta_params = self.localization_networks(x_wide)

        sx = theta_params[:, 0]
        sy = theta_params[:, 1]
        tx = theta_params[:, 2]
        ty = theta_params[:, 3]

        theta = torch.zeros(B, 2, 3, device=x_wide.device)

        theta[:, 0, 0] = sx
        theta[:, 1, 1] = sy
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty

        grid = F.affine_grid(theta=theta, size=grid_size, align_corners=False)

        x_crop = F.grid_sample(grid=grid, input=x_wide, mode='bilinear', padding_mode='border', align_corners=False)
        return x_crop, theta

class CascadedCrossScaleInterrogation(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        
        # STAGE 1: LOCAL INTERROGATES MICRO 
        # LayerNorm
        self.norm1_q = nn.LayerNorm(embed_dim)
        self.norm1_kv = nn.LayerNorm(embed_dim)
        self.cross_attn_1 = nn.MultiheadAttention(embed_dim=embed_dim, 
                                                  num_heads=num_heads, 
                                                  batch_first=True)
        
        # STAGE 2: GLOBAL INTERROGATES ENHANCED LOCAL
        self.norm2_q = nn.LayerNorm(embed_dim)
        self.norm2_kv = nn.LayerNorm(embed_dim)
        self.cross_attn_2 = nn.MultiheadAttention(embed_dim=embed_dim, 
                                                  num_heads=num_heads, 
                                                  batch_first=True)

    def forward(self, global_cls, local_patches, micro_patches):
        # Stage 1: Local <- cross Micro
        q1 = self.norm1_q(local_patches)
        k1v = self.norm1_kv(micro_patches)          # share norm cho k & v
        attn_out_1, _ = self.cross_attn_1(q1, k1v, k1v)   # để tự học Wv riêng

        enhanced_local = local_patches + attn_out_1

        # Stage 2: Global <- cross Enhanced Local
        q2 = self.norm2_q(global_cls)
        k2v = self.norm2_kv(enhanced_local)
        attn_out_2, attn_weights_2 = self.cross_attn_2(q2, k2v, k2v)

        z_final = global_cls + attn_out_2

        return z_final, attn_weights_2