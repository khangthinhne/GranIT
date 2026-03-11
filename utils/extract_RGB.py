import cv2
import numpy as np
import os

def save_rgb_channels_separately(image_path, output_dir="./output_channels"):
    """
    Hàm bóc tách ảnh và lưu thành 3 file ảnh Đỏ, Xanh lá, Xanh dương riêng biệt.
    """
    # 1. Tạo thư mục chứa ảnh nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Đọc ảnh (OpenCV đọc mặc định là hệ BGR)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"❌ Khóc thét: Không tìm thấy ảnh tại {image_path}")
        return

    # 3. Tách 3 kênh màu nguyên thủy của OpenCV (B, G, R)
    b, g, r = cv2.split(img_bgr)

    # 4. Tạo "màn đêm" (ma trận toàn số 0)
    zeros = np.zeros_like(b)

    # 5. Gộp lại theo ĐÚNG CHUẨN BGR để hàm imwrite không bị ngáo màu
    # Ảnh chỉ có màu Blue (B giữ nguyên, G=0, R=0)
    img_blue_only = cv2.merge([b, zeros, zeros])
    
    # Ảnh chỉ có màu Green (B=0, G giữ nguyên, R=0)
    img_green_only = cv2.merge([zeros, g, zeros])
    
    # Ảnh chỉ có màu Red (B=0, G=0, R giữ nguyên)
    img_red_only = cv2.merge([zeros, zeros, r])

    # 6. Lấy tên gốc của bức ảnh để lưu cho ngầu
    # Ví dụ: "000_wide_f0.jpg" -> base_name = "000_wide_f0"
    base_name = os.path.basename(image_path).split('.')[0]
    
    path_r = os.path.join(output_dir, f"{base_name}_RED.jpg")
    path_g = os.path.join(output_dir, f"{base_name}_GREEN.jpg")
    path_b = os.path.join(output_dir, f"{base_name}_BLUE.jpg")
    
    # 7. Lưu thẳng xuống ổ cứng với chất lượng cao nhất (100%)
    cv2.imwrite(path_r, img_red_only, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(path_g, img_green_only, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(path_b, img_blue_only, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    
    print("=" * 40)
    print(f"✅ Đã lưu thành công 3 kênh màu riêng biệt:")
    print(f" 🔴 Kênh Đỏ       -> {path_r}")
    print(f" 🟢 Kênh Xanh Lá  -> {path_g}")
    print(f" 🔵 Kênh Xanh Dương -> {path_b}")
    print("=" * 40)

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # Đổi đường dẫn này thành đường dẫn thật tới ảnh của ông
    test_img = "data/faces_processed/original/159_wide_f64.jpg"
    save_rgb_channels_separately(test_img)