import torch
import torch.nn as nn
import timm
from modules.LoRA import merge_lora
from modules.modules import AFC, CascadedCrossScaleInterrogation, ConstrainedHighPassFilter
from config import BACKBONE_NAME
'''
GranIT: A 3-Branch Framework
'''

# Baseline Vision Transformer: vit_base_patch16_clip_224.laion2b
class BaselineViT(nn.Module):
    def __init__(self, model_name=BACKBONE_NAME, pretrained=True, num_classes=2):
        super().__init__() 
        self.backbone: nn.Module = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=num_classes)
        self.pretrained = pretrained
        self.embed_dim = self.backbone.num_features
        merge_lora(backbone=self.backbone)
        if num_classes > 0:
            self.classifier = nn.Linear(in_features=self.embed_dim, out_features=num_classes)
        else:
            self.classifier = nn.Identity() #remove head

    def forward(self, x, return_tokens=False):
        features = self.backbone.forward_features(x)

        cls_token = features[:, 0]
        logits = self.classifier(cls_token)

        if return_tokens:
            patch_tokens = features[:, 1:]
            return logits, cls_token, patch_tokens
        
        return logits

class MicroBranch(nn.Module):
    def __init__(self, model_name=BACKBONE_NAME, pretrained=True, use_lhpf=False):
        super().__init__()
        self.model = BaselineViT(model_name, pretrained=True, num_classes=0)
        
        if use_lhpf:
            self.high_pass_filter = ConstrainedHighPassFilter(3, 3)
        else:
            # Fixed Laplacian Filter
            self.high_pass_filter = nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3, bias=False)
            kernel = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
            self.high_pass_filter.weight.data = kernel
            self.high_pass_filter.weight.requires_grad = False
        self.model.head = nn.Identity() # remove classification block (head)

    def forward(self, x_crop):
        x_freq = self.high_pass_filter(x_crop)
        _, _, patch_tokens = self.model(x_freq, return_tokens=True)
        return patch_tokens

class LocalBranch(nn.Module):
    def __init__(self, model_name=BACKBONE_NAME, pretrained=True):
        super().__init__()
        self.model = BaselineViT(model_name, pretrained=True, num_classes=0)
    
    def forward(self, x_crop):
        _, cls_tokens, patch_tokens = self.model(x_crop)
        return cls_tokens, patch_tokens 
        

class GranIT(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512, has_global=True):
        super().__init__()
        
        
        # GLOBAL & LOCAL - share backbone
        self.global_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.local_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
    
        # STN
        self.afc_module = AFC(crop_size=(224, 224), use_fg_afc=True, use_lhpf=True)
        
        # MICRO - Frequecy
        self.micro_branch = MicroBranch(model_name=BACKBONE_NAME, use_lhpf=True)
        
        # cascaded cross-scale interrogate
        self.interrogator = CascadedCrossScaleInterrogation(embed_dim=embed_dim)

        self.theta_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        B = x_wide.size(0)
        # GLOBAL BRANCH
        _, cls_global, _ = self.global_vit(x_wide, return_tokens=True)
        # cls_global = cls_global.unsqueeze(1)
        # LOCAL BRANCH
        # AFC cropped the 224x224 region
        x_crop, theta = self.afc_module(x_wide)
        #Take the patch tokens for query section
        _, _, patch_tokens_local = self.local_vit(x_crop, return_tokens=True) 

        # MICRO BRANCH
        patch_tokens_micro = self.micro_branch(x_crop)

        theta_flat = theta.view(B, 6)
        theta_emb = self.theta_proj(theta_flat) # [B, 768]
        
        # Add position to [CLS] token Global
        cls_global_aware = cls_global + theta_emb
        #Cross-Attention Mechanism
        z_final, attn_weights = self.interrogator(cls_global_aware, patch_tokens_local, patch_tokens_micro)
        # z_final = z_final.squeeze(1)
        logits = self.mlp(z_final)
        
        return logits, theta, attn_weights
