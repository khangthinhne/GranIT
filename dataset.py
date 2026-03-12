import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

class FaceForensicsDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        """
        Args:
            image_paths (list): Danh sách đường dẫn tới các file ảnh (.jpg)
            transform (callable, optional): Các phép biến đổi ảnh (Augmentation)
        """
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 1. Đọc ảnh bằng PIL (chuẩn của torchvision)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Lỗi đọc ảnh {img_path}: {e}")
            # Nếu lỗi, đọc tạm ảnh đầu tiên để không bị crash list
            image = Image.open(self.image_paths[0]).convert('RGB')

        # 2. Áp dụng Augmentation & chuyển thành Tensor
        if self.transform:
            image = self.transform(image)

        # 3. Gán nhãn (Labeling)
        # Nếu thư mục chứa chữ 'original' -> Real (0)
        # Còn lại (Face2Face, Deepfakes...) -> Fake (1)
        if "original" in img_path:
            label = 0  # Real
        else:
            label = 1  # Fake

        return image, label


def get_dataloaders(data_dir, batch_size=32, val_split=0.2, mode='afc'):
    """
    DataLoader vs Video-level split và Weighted Sampling.
    """
    all_images = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True)
    
    if len(all_images) == 0:
        raise ValueError(f"No images in {data_dir}")

    # ---------------------------------------------------------
    # BƯỚC 1: VIDEO-LEVEL SPLIT (Cực kỳ quan trọng)
    # ---------------------------------------------------------
    # Gom nhóm các frame thuộc cùng 1 video lại với nhau.
    # Tên file của ông đang có dạng: "000_wide_f0.jpg" -> Tên video là "000"
    video_dict = {}
    for img_path in all_images:
        # Lấy tên folder chứa ảnh + prefix tên file để làm ID duy nhất cho video
        # Ví dụ: "Deepfakes/c23/videos/000"
        folder_path = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        video_id = filename.split('_')[0] 
        
        unique_vid_key = os.path.join(folder_path, video_id)
        
        if unique_vid_key not in video_dict:
            video_dict[unique_vid_key] = []
        video_dict[unique_vid_key].append(img_path)

    # Chia Train/Val dựa trên danh sách Video (không phải danh sách frame)
    unique_videos = list(video_dict.keys())
    random.shuffle(unique_videos)
    
    split_idx = int(len(unique_videos) * (1 - val_split))
    train_vids = unique_videos[:split_idx]
    val_vids = unique_videos[split_idx:]

    train_paths = []
    for vid in train_vids:
        train_paths.extend(video_dict[vid])
        
    val_paths = []
    for vid in val_vids:
        val_paths.extend(video_dict[vid])

    print(f"Total Train Images: {len(train_paths)} | Val: {len(val_paths)}")

    # ---------------------------------------------------------
    # BƯỚC 2: ĐỊNH NGHĨA DATA AUGMENTATION
    # ---------------------------------------------------------
    # LƯU Ý: Không resize ở đây vì ảnh ông cắt đã là 384x384. 
    # Mạng Selector sẽ nhận 384x384 và tự STN crop xuống 224x224.
    if mode == 'tight':
        # TIGHT: Cắt chính giữa 224x224 (Bỏ hết background/tóc)
        train_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif mode == 'wide':
        # WIDE: Ép toàn bộ ảnh 384x384 xuống 224x224 (Giữ nền nhưng mờ mặt)
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # AFC (Mặc định): Giữ nguyên 384x384 để mạng STN tự xử
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    train_dataset = FaceForensicsDataset(train_paths, transform=train_transforms)
    val_dataset = FaceForensicsDataset(val_paths, transform=val_transforms)

    # ---------------------------------------------------------
    # BƯỚC 3: WEIGHTED SAMPLER CHO TẬP TRAIN (Giải quyết 1 Real vs 4 Fake)
    # ---------------------------------------------------------
    train_labels = [0 if "original" in path else 1 for path in train_paths]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    print(f"Ratio of Train - Real (0): {class_counts[0]}, Fake (1): {class_counts[1]}")

    # Trọng số của mỗi class = Tổng số mẫu / Số mẫu của class đó
    class_weights = [1.0 / class_counts[0], 1.0 / class_counts[1]]
    
    # Gán trọng số cho TỪNG ẢNH trong tập train
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    # ---------------------------------------------------------
    # BƯỚC 4: TẠO DATALOADER
    # ---------------------------------------------------------
    # Lưu ý: Khi dùng sampler thì không được set shuffle=True nữa
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler, 
        num_workers=4, # Tăng lên 4 hoặc 8 nếu CPU mạnh để load ảnh nhanh hơn
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


# ==========================================
# TEST THỬ CODE (Chỉ chạy khi ông gõ python dataset.py)
# ==========================================
if __name__ == "__main__":
    DATA_ROOT = "./data/faces_processed" # Đổi lại đường dẫn nếu cần
    
    try:
        train_loader, val_loader = get_dataloaders(DATA_ROOT, batch_size=32)
        
        # Rút thử 1 batch ra xem thế nào
        for images, labels in train_loader:
            print("\n--- TEST LOAD 1 BATCH ---")
            print(f"Shape của 1 batch ảnh: {images.shape}") # Kỳ vọng: [32, 3, 384, 384]
            print(f"Nhãn của batch này: \n{labels}")
            
            # Đếm xem Sampler hoạt động tốt không (Kỳ vọng số lượng 0 và 1 xấp xỉ bằng nhau)
            num_real = (labels == 0).sum().item()
            num_fake = (labels == 1).sum().item()
            print(f"Trong batch 32 ảnh có: {num_real} Real và {num_fake} Fake.")
            break # Chạy 1 batch rồi dừng
            
    except Exception as e:
        print("Lỗi:", e)