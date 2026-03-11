import torch
import torch.nn as nn
import timm
from LoRA import merge_lora
from modules import TransformerSTN
'''
GranIT: A 3-Branch Framework
'''

        
class BaselineViT(nn.Module):
    def __init__(self, model_name='', pretrained=True, num_class=2):
        super().__init__() 
        self.backbone: nn.Module = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_class)
        self.pretrained = pretrained
        self.embed_dim = self.backbone.num_features
        merge_lora(backbone=self.backbone)
        self.classifier = nn.Linear(in_features=self.embed_dim, out_features=num_class)

    def forward(self, x, return_tokens=False):
        features = self.backbone.forward_features(x)

        cls_token = features[:, 0]
        logits = self.classifier(cls_token)

        if return_tokens:
            patch_tokens = features[:, 1:]
            return logits, cls_token, patch_tokens
        
        return logits

class MicroBranchViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', use_lora=True):
        super().__init__()
        
        #  ViT backbone 
        self.freq_encoder = timm.create_model(
            model_name, 
            pretrained=True, 
            num_classes=0,
            global_pool=''          # return [B, 197, 768]
        )
        if use_lora:
            merge_lora(backbone=self.freq_encoder)

    def extract_frequency(self, x):
        B, C, H, W = x.shape
        
        # FFT 
        fft_x = torch.fft.fft2(x, norm='ortho')
        fft_shift = torch.fft.fftshift(fft_x, dim=(-2, -1))
        
        # High-pass mask circle
        Y, X = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing='ij')
        center_y, center_x = H // 2, W // 2
        radius = 15.0
        mask = ((X - center_x)**2 + (Y - center_y)**2 > radius**2).float()
        mask = mask[None, None, :, :]  # [1,1,H,W] → broadcast to [B,3,H,W]
        
        fft_filtered = fft_shift * mask
        magnitude = torch.abs(fft_filtered)
        magnitude_log = torch.log1p(magnitude)  # log(1 + |F|)
        
        # Normalize per image 
        magnitude_log = (magnitude_log - magnitude_log.mean(dim=[1,2,3], keepdim=True)) / \
                        (magnitude_log.std(dim=[1,2,3], keepdim=True) + 1e-5)
        
        return magnitude_log  #  [B,3,H,W]

    def forward(self, x_cropped):
        freq_img = self.extract_frequency(x_cropped)  # [B,3,224,224]
        
        # Feed ViT
        tokens = self.freq_encoder(freq_img)          # [B, 197, 768] (cls + patches)
        
        patch_tokens = tokens[:, 1:, :]               # [B, 196, 768] - bỏ [cls]
        
        return patch_tokens


class CrossScaleInterrogation(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super().__init__()
        
        # GLobal -> Local
        self.attn_local = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        
        # Global -> Micro 
        self.attn_freq = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.1)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, q_global, kv_local, kv_freq):
        # query of global
        query = q_global.unsqueeze(1)
        
        # cross-attn: global -> local
        local_report, _ = self.attn_local(query, kv_local, kv_local)
        local_report = self.norm1(local_report).squeeze(1) # Trở về [Batch, 768]
        
        # cross-attn: global -> micro : Cần chỉnh lại local -> micro trước rồi tới global -> local 
        freq_report, _ = self.attn_freq(query, kv_freq, kv_freq)
        freq_report = self.norm2(freq_report).squeeze(1) # Trở về [Batch, 768]
        
        # combination
        combined_features = torch.cat([q_global, local_report, freq_report], dim=-1) # [Batch, 768 * 3]
        
        # mlp -> combine infor
        final_feature = self.fusion_mlp(combined_features) # [Batch, 768]
        
        return final_feature
class GranIT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768):
        super().__init__()
        
        
        # GLOBAL & LOCAL - share backbone
        self.shared_vit = BaselineViT(model_name='vit_base_patch16_clip_224.laion2b', pretrained=True, num_class=0)
        
        # STN
        self.stn = TransformerSTN(crop_size=(224, 224))
        
        # MICRO (TẦN SỐ)
        self.micro_branch = MicroBranchViT(model_name='vit_base_patch16_clip_224.laion2b')
        
        # cascaded cross-scale interrogate
        self.interrogator = CrossScaleInterrogation(embed_dim=embed_dim)

        self.classifier = nn.Linear(embed_dim, num_classes)
        

    def forward(self, x_wide):
        # GLOBAL
        _, cls_global, _ = self.shared_vit(x_wide, return_tokens=True)
        
        # LOCAL
        ## STN -> cropped region
        x_cropped, theta = self.stn(x_wide)
        
        _, _, patch_local = self.shared_vit(x_cropped, return_tokens=True)
        
        # Micro: High pass frequency mask
        patch_freq = self.micro_branch(x_cropped)

        fused_feature = self.interrogator(q_global=cls_global, 
                                          kv_local=patch_local, 
                                          kv_freq=patch_freq)
        
        logits = self.classifier(fused_feature)
        
        # theta -> loss stn
        return logits, theta
