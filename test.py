import os
import argparse
import config
from PIL import Image

# Giả sử tui dùng lại logic Center Crop tỷ lệ nãy tui với ông viết
def crop_beta_on_image(image, target_beta):
    # Load ảnh hiện tại (đang là margin 1.5)
    
    if target_beta < 1.5:
        # 1. Tính tỷ lệ cần cắt
        ratio = target_beta / 1.5
        
        # 2. Lấy kích thước ảnh gốc (của mức 1.5)
        w, h = image.size
        
        # 3. Tính kích thước khung mới
        new_w, new_h = w * ratio, h * ratio
        
        # 4. Tính tọa độ để cắt đúng tâm (Center Crop)
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = (w + new_w) / 2
        bottom = (h + new_h) / 2
        
        # 5. Thực hiện cắt ảnh
        image = image.crop((left, top, right, bottom))
        
    return image

if __name__ == "__main__":
    # === ÔNG SỬA ĐƯỜNG DẪN ẢNH VÀO ĐÂY ===
    # Nhớ dùng ảnh đang ở margin 1.5 nha!
    input_image_path = "data/faces_processed_split/test/FaceSwap/001_870_wide_f184.jpg"
    image_name = "visualize_beta"
    # =====================================
    
    image = Image.open(input_image_path).convert('RGB')
    
    margins = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    
    # Tạo thư mục đầu ra
    output_dir = "visualize_beta_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[*] Đang xử lý ảnh: {input_image_path}...")
    
    for beta in margins:
        cropped_image = crop_beta_on_image(image, beta)
        # Lưu ảnh với tên bao gồm giá trị beta
        cropped_image.save(os.path.join(output_dir, f"{image_name}_{beta:.1f}.jpg"))
        
    print(f"[*] HOÀN TẤT! Đã tạo 6 trạng thái beta trong thư mục '{output_dir}'.")