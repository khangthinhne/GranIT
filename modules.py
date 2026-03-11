import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerSTN(nn.Module):
    def __init__(self, in_channels=3, embed_dim=128, crop_size=(224, 224)):
        super().__init__()
        self.crop_size = crop_size
        
        # 1. TRẠM TIỀN PHƯƠNG (CNN siêu nhẹ để băm ảnh thành Patch)
        # Giữ lại một chút CNN để giảm size ảnh cực nhanh, cứu VRAM cho ông
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(True), nn.MaxPool2d(2, 2)
        ) 
        # Output của trạm này sẽ là [Batch, 128, 14, 14]
        
        # 2. ĐIỆP VIÊN [LOC] (Learnable Token)
        # Tạo ra 1 vector có chiều dài bằng embed_dim (128)
        self.loc_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.01)
        
        # 3. PHÒNG THẨM VẤN (Cross-Attention)
        # num_heads=4 giúp [LOC] có thể nhìn vào 4 vùng lỗi khác nhau cùng lúc
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True) 
        
        # 4. CHUYỂN ĐỔI TOẠ ĐỘ (Regressor)
        self.fc_loc = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2) # Xuất ra ma trận 2x3 (6 con số)
        )
        
        # BÍ THUẬT: Vẫn phải khởi tạo Ma trận Identity để tránh mù ở Epoch 0
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        B = x.size(0)
        
        # --- BƯỚC 1: Băm ảnh ---
        features = self.patch_embed(x) # [B, 128, 14, 14]
        C, H, W = features.size(1), features.size(2), features.size(3)
        
        # Ép phẳng 2D thành dải 1D (Tokens) -> Shape: [B, 196, 128]
        patch_tokens = features.view(B, C, H * W).transpose(1, 2)
        
        # --- BƯỚC 2: Gọi Điệp viên ---
        # Nhân bản cái token [LOC] lên cho đủ số lượng ảnh trong Batch
        query = self.loc_token.expand(B, -1, -1) # Shape: [B, 1, 128]
        
        # --- BƯỚC 3: Thẩm vấn (Cross-Attention) ---
        # Query đi hỏi Key và Value (chính là patch_tokens)
        attn_output, _ = self.cross_attn(query, patch_tokens, patch_tokens)
        
        # Lấy vector [LOC] sau khi hút thông tin (Bỏ chiều dư thừa đi)
        loc_features = attn_output.squeeze(1) # Shape: [B, 128]
        
        # --- BƯỚC 4: Ra tọa độ và Cắt ảnh ---
        theta = self.fc_loc(loc_features)
        theta = theta.view(-1, 2, 3) # Shape: [B, 2, 3]
        
        output_size = torch.Size((B, x.size(1), self.crop_size[0], self.crop_size[1]))
        grid = F.affine_grid(theta, output_size, align_corners=False)
        x_cropped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return x_cropped, theta



class STNCropper(nn.Module):
    def __init__(self, in_channels=3, crop_size=(224, 224)):
        super().__init__()
        self.crop_size = crop_size
        
        # 1. Localization Network (CNN siêu nhẹ để tìm toạ độ zoom)
        # Giảm size ảnh dần dần: 224 -> 112 -> 56 -> 28 -> 14
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=7, padding=3, stride=2), # [8, 112, 112]
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # [8, 56, 56]
            
            nn.Conv2d(8, 10, kernel_size=5, padding=2, stride=2), # [10, 28, 28]
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2), # [10, 14, 14]
        )
        
        # Tính kích thước vector sau khi duỗi thẳng (flatten) = 10 channels * 14 * 14
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 14 * 14, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2) # Đầu ra bắt buộc là 6 con số (Ma trận Affine 2x3)
        )
        
        # [BÍ THUẬT QUAN TRỌNG]: Khởi tạo ma trận Affine mặc định là Ma trận Đơn vị (Identity)
        # Nghĩa là ở Epoch 0, nó sẽ cắt nguyên xi cái ảnh ban đầu, không zoom không lệch.
        # Nếu không làm bước này, số random sẽ làm ảnh bị lật ngược hoặc méo mó ngay từ đầu -> văng Loss.
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        
        # Bước 1: Cho CNN dòm ảnh để đoán 6 con số Affine
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 14 * 14)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3) # Shape: [Batch, 2, 3]
        
        # Bước 2: Tạo lưới cắt ảnh (Grid Generator)
        # Canh chuẩn output size là (Batch, Channels, 224, 224)
        output_size = torch.Size((x.size(0), x.size(1), self.crop_size[0], self.crop_size[1]))
        grid = F.affine_grid(theta, output_size, align_corners=False)
        
        # Bước 3: Cắt ảnh và phóng to (Grid Sampler)
        # Dùng nội suy song tuyến tính (bilinear) để cắt mà không bị vỡ hạt
        x_cropped = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=False)
        
        return x_cropped, theta # Trả về ảnh đã crop và ma trận theta (để debug)

# Test thử
if __name__ == "__main__":
    stn_modern = TransformerSTN()
    dummy_input = torch.randn(2, 3, 224, 224)
    cropped_img, theta = stn_modern(dummy_input)
    print("Shape ảnh Crop:", cropped_img.shape) # Kì vọng: [2, 3, 224, 224]