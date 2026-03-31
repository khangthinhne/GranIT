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
    def __init__(self, backbone_name='vit_base_patch16_clip_224.laion2b',pretrained=True, crop_size=(224,224), use_fg_afc=False, use_lhpf=False):
        super().__init__()
        self.use_fg_afc = use_fg_afc
        self.use_lhpf = use_lhpf
        if self.use_fg_afc:
            if use_lhpf:
                self.high_pass_filter = ConstrainedHighPassFilter(3, 3)
            else:
                self.high_pass_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
                kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
                self.high_pass_filter.weight.data = kernel
                self.high_pass_filter.weight.requires_grad = False
            self.fusion_conv = nn.Conv2d(6, 3, kernel_size=1)

        from modules.model import BaselineViT
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
            if 'lora' in name or 'stn_head' in name or 'fusion_conv' in name:
                param.requires_grad = True
            elif 'high_pass_filter' in name:
                param.requires_grad = self.use_lhpf
            else:
                param.requires_grad = False
                
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[{self.__class__.__name__}] Trainable parameters (LoRA + MLP): {trainable_params:,}")
    def forward(self, x):
        if self.use_fg_afc:
            x_freq = self.high_pass_filter(x)
            x_fused = torch.cat([x, x_freq], dim=1)
            x_input = self.fusion_conv(x_fused)
        else:
            x_input = x
        features = self.backbone(x_input) # x: [B, 3, 224, 224] -> features: [B, 768]?
        theta_params = self.stn_head(features)
        return theta_params # [s_x, s_y, t_x, t_y]

class AFC(nn.Module):
    def __init__(self, crop_size=(224,224), use_fg_afc=False, use_lhpf=False):
        super().__init__()
        self.localization_networks = LocalizationNetwork(use_fg_afc=use_fg_afc, use_lhpf=use_lhpf)
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
        enhanced_local = local_patches
        if micro_patches is not None:
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
            print(f"Val Loss reduce perfectly ({self.best_val_loss:.4f} --> {val_loss:.4f})...")
            self.best_val_loss = val_loss
            # save_path_loss = os.path.join(self.save_dir, f"{self.model_name}_BEST_VAL_LOSS.pth")
            # torch.save(model.state_dict(), save_path_loss)
            
            self.counter = 0 
        else:
            self.counter += 1
            print(f"Val Loss DOES NOT DECREASE! Warn Overfitting: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True 


# ==========================================
# Variant 1: Simple Concatenation
# ==========================================
class SimpleConcatFusion(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        # 3 nhánh -> 3 cái [CLS] token -> tổng dim là 3 * 768
        self.proj = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, cls_global, patch_local, patch_micro):
        cls_global = cls_global.squeeze(1) if cls_global.dim() == 3 else cls_global
        # Tính Global Average Pooling (hoặc lấy CLS) cho nhánh Local và Micro
        # patch_local: [B, 196, 768] -> [B, 768]
        local_pool = patch_local.mean(dim=1) 
        micro_pool = patch_micro.mean(dim=1)
        
        # Nối lại: [B, 768 * 3]
        fused = torch.cat([cls_global, local_pool, micro_pool], dim=-1)
        
        # Chiếu về lại embed_dim để nhét vào MLP cuối
        z_final = self.proj(fused)
        return z_final, None # Return None cho attn_weights vì không xài Attention

# ==========================================
# Variant 2: Standard Self-Attention
# ==========================================
class StandardSelfAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        # Xài đúng 1 block Encoder của Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, 
                                                   dim_feedforward=embed_dim * 4, 
                                                   batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
    def forward(self, cls_global, patch_local, patch_micro):
        B = cls_global.size(0)
        cls_global = cls_global.unsqueeze(1) # [B, 1, 768]
        
        # Nối tất cả thành 1 sequence siêu dài: 1 + 196 + 196 = 393 tokens
        # sequence: [B, 393, 768]
        sequence = torch.cat([cls_global, patch_local, patch_micro], dim=1)
        
        # Đẩy qua Self-Attention
        out_sequence = self.transformer(sequence)
        
        # Lấy lại cái token đầu tiên (đại diện cho CLS) làm output
        z_final = out_sequence[:, 0, :]
        return z_final, None

# ==========================================
# Variant 3: Parallel Cross-Attention
# ==========================================
class ParallelCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        # Khởi tạo 2 bộ Cross-Attention độc lập
        self.cross_attn_local = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_micro = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, cls_global, patch_local, patch_micro):
        cls_global_q = cls_global.unsqueeze(1) # [B, 1, 768]
        
        # Nhánh 1: Global soi Local
        z_local, _ = self.cross_attn_local(query=cls_global_q, key=patch_local, value=patch_local)
        
        # Nhánh 2: Global soi Micro (Làm song song, không phụ thuộc kết quả nhánh 1)
        z_micro, _ = self.cross_attn_micro(query=cls_global_q, key=patch_micro, value=patch_micro)
        
        # Cộng 2 kết quả lại (hoặc concat tùy ông)
        z_final = z_local.squeeze(1) + z_micro.squeeze(1)
        z_final = self.norm(z_final)
        
        return z_final, None
    

class SequentialCrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.norm_final = nn.LayerNorm(embed_dim)

    def forward(self, cls_global, patch_local, patch_micro):
        # Đảm bảo shape là [B, 1, 768]
        cls_q = cls_global.unsqueeze(1) if cls_global.dim() == 2 else cls_global

        # Step 1: Global -> Micro (Cộng Residual)
        q1 = self.norm1(cls_q)
        z1, _ = self.cross_attn1(q1, patch_micro, patch_micro)
        cls_enhanced_1 = cls_q + z1

        # Step 2: Global (đã có thông tin Micro) -> Local (Cộng Residual)
        q2 = self.norm2(cls_enhanced_1)
        z2, _ = self.cross_attn2(q2, patch_local, patch_local)
        cls_enhanced_2 = cls_enhanced_1 + z2

        # Squeeze lại và Normalize lần cuối
        z_final = self.norm_final(cls_enhanced_2.squeeze(1))
        return z_final, None
    

fusion_list = {
    'CCSIM': CascadedCrossScaleInterrogation, 
    'SCF': SimpleConcatFusion, 
    'SSAF': StandardSelfAttentionFusion, 
    'PCAF': ParallelCrossAttentionFusion, 
    'SCAF':SequentialCrossAttentionFusion}
