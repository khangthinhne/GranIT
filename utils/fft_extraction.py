import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_spatial_highpass_filter(image_path, save_path=None):
    """
    Hàm lọc ảnh để lấy tần số cao nhưng vẫn giữ nguyên cấu trúc không gian (mắt mũi miệng).
    Sử dụng bộ lọc Laplacian 3x3 kinh điển.
    """
    # 1. Đọc ảnh và chuyển sang ảnh xám (Grayscale)
    # Tần số cao thường được phân tích trên kênh độ sáng (luminance)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        print(f"❌ Không tìm thấy ảnh tại: {image_path}")
        return

    # 2. Định nghĩa Nhân Chập (Convolution Kernel) - Bộ lọc thông cao 3x3
    # Tổng các phần tử trong ma trận này bằng 0. 
    # Vùng nào màu phẳng (tần số thấp) đi qua sẽ thành 0 (đen/xám).
    # Vùng nào có cạnh/nhiễu (tần số cao) sẽ bật sáng lên.
    kernel_3x3 = np.array([[-1, -1, -1],
                           [-1,  8, -1],
                           [-1, -1, -1]], dtype=np.float32)

    # 3. Thực hiện phép nhân chập (Convolution)
    # Dùng cv2.CV_32F vì kết quả có thể là số âm
    filtered_image_float = cv2.filter2D(img_gray, cv2.CV_32F, kernel_3x3)

    # 4. Xử lý hậu kỳ để hiển thị đẹp như ảnh Lena
    # Kết quả đang dao động quanh số 0. Để nhìn được màu xám trung tính, ta cộng thêm 128.
    # Những chỗ có cạnh sẽ sáng hơn hoặc tối hơn mức xám 128 này.
    visualization_img = filtered_image_float + 128.0
    
    # Kẹp giá trị lại trong khoảng 0-255 và chuyển về dạng ảnh không dấu (uint8)
    visualization_img = np.clip(visualization_img, 0, 255)
    visualization_img = visualization_img.astype(np.uint8)

    # --- Hiển thị và Lưu ---
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title("Ảnh gốc RGB (Local Input)", fontweight='bold')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # Hiển thị ảnh xám (cmap='gray')
    plt.imshow(visualization_img, cmap='gray', vmin=0, vmax=255)
    plt.title("Ảnh Tần số cao Không gian\n(Micro Input)", fontweight='bold', color='purple')
    plt.axis('off')

    plt.tight_layout()
    
    # Lưu ảnh kết quả riêng ra để ông nhét vào draw.io
    if save_path:
        # Tạo thư mục nếu chưa có
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Lưu ảnh xám
        cv2.imwrite(save_path, visualization_img)
        print(f"✅ Đã lưu ảnh tần số cao tại: {save_path}")

    plt.show()

# --- CHẠY THỬ NGAY ---
if __name__ == "__main__":
    # Lấy cái ảnh crop mặt người của ông đưa vào đây
    sample_crop = "data/faces_wide/original/001_wide_f114.jpg" # Ví dụ ảnh đã crop
    output_file = "output_images/spatial_highpass_example.jpg"
    
    generate_spatial_highpass_filter(sample_crop, save_path=output_file)