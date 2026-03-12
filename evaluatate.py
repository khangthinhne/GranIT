import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
import cv2
import numpy as np
import os
from tqdm import tqdm

# Import từ các file của ông
from dataset import get_dataloaders
from model import GranIT

def evaluate_and_visualize(model_path="models/GranIT_vit_lora_1.pth", data_dir="./data/faces_processed", vis_dir="./visualizations"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Đang đánh giá t                                                                     rên {device}...")

    # 1. Load data validation
    _, val_loader = get_dataloaders(data_dir, batch_size=16)

    # 2. Khởi tạo và Load Model
    model = GranIT() 
    
    # Load trọng số BEST mà ông đã train được
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Đã load weights con GranIT thành công!k")
    except Exception as e:
        print(f" Lỗi load weights: {e}")
        return


    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    # Tạo thư mục lưu ảnh vẽ box
    os.makedirs(vis_dir, exist_ok=True)
    images_saved = 0
    FIRST_IMAGES = 20 # Chỉ lưu 20 ảnh đầu cho lẹ

    print("Đang chạy Inference trên tập Validation...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Chạy qua model (Đầu ra là logits và theta)
            logits, theta = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1] # Xác suất Fake
            _, preds = torch.max(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # --- TRỰC QUAN HÓA (Vẽ Bounding Box) ---
            if images_saved < FIRST_IMAGES:
                for j in range(images.size(0)):
                    if images_saved >= FIRST_IMAGES: break
                    
                    # 1. Lấy thông số Affine Matrix (theta) [2, 3]
                    # [ s_x,  0,   t_x ]
                    # [ 0,    s_y, t_y ]
                    t = theta[j].cpu().numpy()
                    s_x = t[0, 0]
                    s_y = t[1, 1]
                    t_x = t[0, 2]
                    t_y = t[1, 2]

                    # 2. Denormalize ảnh từ Tensor về numpy [0, 255]
                    img_np = images[j].cpu().numpy().transpose(1, 2, 0)
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_np = std * img_np + mean
                    img_np = np.clip(img_np, 0, 1) * 255.0
                    img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

                    # 3. Tính toán Bounding Box chuẩn xác theo hệ trục PyTorch [-1, 1]
                    H, W, _ = img_bgr.shape
                    
                    # CỰC KỲ QUAN TRỌNG: Công thức giải mã STN Pytorch sang Pixel
                    # Tâm cx, cy dịch chuyển từ [-1, 1] sang [0, W]
                    cx = int((t_x + 1.0) / 2.0 * W)
                    cy = int((t_y + 1.0) / 2.0 * H)
                    
                    # Chiều dài, rộng thực tế trên ảnh pixel
                    box_w = int(s_x * W)
                    box_h = int(s_y * H)

                    # Tọa độ 2 góc (x1, y1) và (x2, y2)
                    x1 = max(0, cx - box_w // 2)
                    y1 = max(0, cy - box_h // 2)
                    x2 = min(W, cx + box_w // 2)
                    y2 = min(H, cy + box_h // 2)

                    # 4. Vẽ vời
                    color = (0, 255, 0) if labels[j].item() == preds[j].item() else (0, 0, 255)
                    
                    # Vẽ Khung STN
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
                    
                    # Ghi Text
                    true_txt = "Real" if labels[j].item() == 0 else "Fake"
                    pred_txt = "Real" if preds[j].item() == 0 else "Fake"
                    cv2.putText(img_bgr, f"T:{true_txt}|P:{pred_txt}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    # Lưu ảnh
                    cv2.imwrite(os.path.join(vis_dir, f"vis_epoch_best_{images_saved}.jpg"), img_bgr)
                    images_saved += 1

    # Tính toán các chỉ số
    acc = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    print("\n" + "="*40)
    print("🔥 KẾT QUẢ INFERENCE TẬP VALIDATION 🔥")
    print("="*40)
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"AUC Score: {auc:.4f}")
    print("="*40)
    print(f"📸 Đã lưu 20 ảnh crop vào thư mục: {vis_dir}")

if __name__ == "__main__":
    evaluate_and_visualize()