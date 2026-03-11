import torch
import torch.nn as nn

class MiniDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- CÔNG NHÂN TRÍCH XUẤT (Feature Extraction) ---
        # Công thức chuẩn: Conv -> BatchNorm -> ReLU -> MaxPool
        
        # Block 1: Quét sương sương
        self.block1 = nn.Sequential(
            # Nhận 3 kênh màu (RGB), xuất ra 16 kênh đặc trưng. 
            # kernel_size=3, padding=1 giúp giữ nguyên chiều dài x rộng của ảnh
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), # Cảnh sát giao thông chốt chặn cho 16 kênh
            nn.ReLU(),          # Bẻ cong dữ liệu
            nn.MaxPool2d(kernel_size=2, stride=2) # Ép size giảm một nửa
        )
        
        # Block 2: Quét chi tiết sâu hơn
        self.block2 = nn.Sequential(
            # Nhận 16 kênh từ Block 1, xuất ra 32 kênh tinh hoa hơn
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # Ép size giảm thêm một nửa nữa
        )
        
        # --- BỘ NÃO SUY LUẬN (Decision Block) ---
        # Ảnh 64x64 đi qua 2 lần MaxPool (chia 2 rồi lại chia 2) -> còn 16x16.
        # Chiều sâu lúc này là 32 kênh. 
        # Vậy tổng số lượng con số khi duỗi thẳng (flatten) = 32 * 16 * 16 = 8192
        
        self.classifier = nn.Sequential(
            nn.Flatten(), # Máy duỗi thẳng ma trận 2D thành vector 1D
            nn.Linear(in_features=8192, out_features=128), # Bóp từ 8192 thông tin về 128
            nn.ReLU(),
            nn.Dropout(p=0.5), # Cho ngất 50% nơ-ron để chống học vẹt
            nn.Linear(in_features=128, out_features=2) # Chốt hạ: Phán 2 class (Real/Fake)
        )

    def forward(self, x):
        print(f"📦 1. Ảnh gốc đi vào        : {x.shape}")
        
        x = self.block1(x)
        print(f"🛠️ 2. Sau khi qua Block 1   : {x.shape} (Dày hơn, nhưng Size bị chia 2)")
        
        x = self.block2(x)
        print(f"🛠️ 3. Sau khi qua Block 2   : {x.shape} (Dày gấp đôi nữa, Size chia 2 tiếp)")
        
        x = self.classifier(x)
        print(f"🧠 4. Kết quả phán đoán cuối: {x.shape} (2 class Real/Fake)")
        
        return x

# ==========================================
# CHẠY THỬ NGHIỆM
# ==========================================
if __name__ == "__main__":
    # Tạo mô hình
    model = MiniDetector()
    
    # Giả lập 1 batch gồm 4 bức ảnh, 3 kênh màu RGB, kích thước 64x64
    dummy_input = torch.randn(4, 3, 64, 64)
    
    print("🚀 BẮT ĐẦU CHẠY MÔ HÌNH 🚀")
    print("-" * 50)
    
    # Đưa ảnh qua mạng
    output = model(dummy_input)
    
    print("-" * 50)
    print("✅ XONG! Dữ liệu đã đi trót lọt từ đầu đến cuối.")