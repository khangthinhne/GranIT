import os
from torch.utils.data import Dataset
from PIL import Image
import config
class BaseDeepfakeDataset(Dataset):
    def __init__(self, image_paths, transform=None, crop_margin=1.5):
        self.image_paths = image_paths
        self.transform = transform
        self.crop_margin = crop_margin

    def __len__(self):
        return len(self.image_paths)
     
    def get_label(self, img_path):
        raise NotImplementedError("Derive this funcc")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Fail {img_path}: {e}")
            image = Image.open(self.image_paths[0]).convert('RGB')

        target_beta = self.crop_margin

        # print("FF++: BETA: ", target_beta)
        if target_beta < 1.5:
            ratio = target_beta / 1.5
            w, h = image.size
            
            new_w, new_h = w * ratio, h * ratio
            left = (w - new_w) / 2
            top = (h - new_h) / 2
            right = (w + new_w) / 2
            bottom = (h + new_h) / 2
    
            image = image.crop((left, top, right, bottom))

        if self.transform:
            image = self.transform(image)

        label = self.get_label(img_path)

        return image, label

class FaceForensicsDataset(BaseDeepfakeDataset):
    def get_label(self, img_path):
        label = 0 if "original" in img_path else 1
        # print(label)
        return label

    @staticmethod
    def extract_video_id(img_path):
        folder_path = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        video_id = filename.split('_')[0]
        return os.path.join(folder_path, video_id)

class CelebDFDataset(BaseDeepfakeDataset):
    def get_label(self, img_path):
        return 0 if "real" in img_path.lower() else 1

    @staticmethod
    def extract_video_id(img_path):
        # = "id0_id16_0000.jpg" -> video_id "id0_id16"
        folder_path = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        video_id = "_".join(parts[:-1]) if len(parts) > 1 else filename
        return os.path.join(folder_path, video_id)

class WildDeepfakeDataset(BaseDeepfakeDataset):
    def get_label(self, img_path):
        return 0 if "real" in img_path.lower() else 1

    @staticmethod
    def extract_video_id(img_path):
        folder_path = os.path.dirname(img_path)
        return folder_path 
    

class DFDCDataset(BaseDeepfakeDataset):
    def get_label(self, img_path):
        # Giữ đúng logic ông vừa viết
        return 0 if "0_Real" in img_path else 1

    @staticmethod
    def extract_video_id(img_path):
        # Tương tự như FaceForensics: Lấy tên thư mục + phần đầu của tên file
        # VD: .../0_Real/atkdltyyen_265_0.jpg -> .../0_Real/atkdltyyen
        folder_path = os.path.dirname(img_path)
        filename = os.path.basename(img_path)
        video_id = filename.split('_')[0]
        
        return os.path.join(folder_path, video_id)