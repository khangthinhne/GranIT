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
    def __init__(self, ):
        super().__init__()
        self.localization_networks = LocalizationNetwork()

    def forward(self, x_wide):
        B = x_wide.size(0) # x_wide: [B, C, H, W]
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

        grid = F.affine_grid(theta=theta, size=x_wide.size(), align_corners=False)

        x_crop = F.grid_sample(grid=grid, input=x_wide, mode='bilinear', padding_mode='border', align_corners=False)
        return x_crop, theta
        