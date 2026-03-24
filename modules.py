import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstrainedHighPassFilter(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, 1, 3, 3))

    def forward(self, x):
        weight = self.weight.clone()
        weight[:, :, 1, 1] = 0
        sum_weight = weight.sum(dim=(2, 3), keepdim=True)
        weight = weight / (sum_weight + 1e-8)
        weight[:, :, 1, 1] = -1.0
        return F.conv2d(x, weight, padding=1, groups=x.size(1))
class LocalizationNetwork(nn.Module):
    def __init__(self, backbone_name='vit_base_patch16_clip_224.laion2b',pretrained=True, crop_size=(224,224)):
        super().__init__()
        from model import BaselineViT
        self.high_pass_filter = ConstrainedHighPassFilter(3, 3)
        self.fusion_conv = nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)
        self.backbone = BaselineViT(model_name=backbone_name, pretrained=pretrained, num_classes=0)
        if hasattr(self.backbone, 'embed_dim'):
            self.embed_dim = self.backbone.embed_dim
        else:
            print(f"Model {backbone_name} does not have a direct 'embed_dim' attribute.")
        print(f"[{self.__class__.__name__}] Initialize ViT ({backbone_name} with embedding dim of {self.embed_dim})")
        self.stn_head = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 4) #  sx, sy, tx, ty
        )

        self.stn_head[2].weight.data.zero_()
        self.stn_head[2].bias.data.copy_(torch.Tensor([1,1,0,0]))
        self._freeze_base_weights()
        
    def _freeze_base_weights(self):

        for name, param in self.named_parameters():
            if 'lora' in name or 'stn_head' in name or 'high_pass_filter' in name or 'fusion_conv' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] Trainable parameters (LoRA + MLP): {trainable_params:,}")
    def forward(self, x):
        x_freq = self.high_pass_filter(x)
        x_fused = torch.cat([x, x_freq], dim=1)
        x_input_vit = self.fusion_conv(x_fused)
        features = self.backbone(x_input_vit) # x: [B, 3, 224, 224] -> features: [B, 768]?
        theta_params = self.stn_head(features)
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
        global_cls = global_cls.unsqueeze(1) if global_cls.dim()==2 else global_cls
        # Stage 1: Local <- cross Micro
        q1 = self.norm1_q(local_patches)
        k1v = self.norm1_kv(micro_patches)          # share norm cho k & v
        attn_out_1, _ = self.cross_attn_1(q1, k1v, k1v)   # để tự học Wv riêng

        enhanced_local = local_patches + attn_out_1

        # Stage 2: Global <- cross Enhanced Local
        q2 = self.norm2_q(global_cls)
        k2v = self.norm2_kv(enhanced_local)
        attn_out_2, attn_weights_2 = self.cross_attn_2(q2, k2v, k2v)

        z_final = (global_cls + attn_out_2).squeeze(1)

        return z_final, attn_weights_2




class DualEarlyStopping:
    def __init__(self, patience=7, delta=0.0, save_dir='./checkpoints', model_name='GranIT'):
        self.patience = patience
        self.delta = delta
        self.save_dir = save_dir
        self.model_name = model_name
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.counter = 0
        self.best_val_loss = np.Inf    
        self.best_val_auc = -np.Inf    
        self.early_stop = False

    def __call__(self, val_loss, val_auc, model):
        if val_auc > self.best_val_auc:
            print(f"Val AUC new record ({self.best_val_auc:.4f} --> {val_auc:.4f}). Save model BEST_AUC...")
            self.best_val_auc = val_auc
            save_path_auc = os.path.join(self.save_dir, f"{self.model_name}_BEST_AUC.pth")
            torch.save(model.state_dict(), save_path_auc)

        if val_loss < self.best_val_loss - self.delta:
            print(f"Val Loss reduce perfectly ({self.best_val_loss:.4f} --> {val_loss:.4f}). SAve model BEST_VAL_LOSS...")
            self.best_val_loss = val_loss
            save_path_loss = os.path.join(self.save_dir, f"{self.model_name}_BEST_VAL_LOSS.pth")
            torch.save(model.state_dict(), save_path_loss)
            
            self.counter = 0 
        else:
            self.counter += 1
            print(f"Val Loss DOES NOT DECREASE! Warn Overfitting: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True 


