import os
import glob
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from data_preparation.dataset_models import FaceForensicsDataset, CelebDFDataset, WildDeepfakeDataset

DATASET_REGISTRY = {
    'faceforensic++': FaceForensicsDataset,
    'celebdf': CelebDFDataset,
    'wilddf': WildDeepfakeDataset
}

DATA_DIR = {
    'faceforensic++': 'data/faces_processed',
    'celebdf': 'data/celebdf_processed' ,
    'wilddf': 'data/test/wilddf_processed'        
}

VAL_SPLIT = 0.2
def get_dataloaders(mode='training', batch_size=8, dataset_model='faceforensic++'):
    """
    mode: 'training' -> (train_loader, val_loader),  'testing' -> (test_loader)
    """
    if dataset_model not in DATASET_REGISTRY:
        raise ValueError(f"Need to check dataset registry: {list(DATASET_REGISTRY.keys())}")

    dir_path = DATA_DIR[dataset_model]
    DatasetClass = DATASET_REGISTRY[dataset_model]
    exts = ["*.jpg", "*.jpeg", "*.png"]
    all_images = []
    for ext in exts:
        all_images.extend(
            glob.glob(os.path.join(dir_path, "**", ext), recursive=True)
        )
    if not all_images:
        raise ValueError(f"No images in {dir_path}")

    # VIDEO-LEVEL SPLIT 
    video_dict = {}
    for img_path in all_images:
        unique_vid_key = DatasetClass.extract_video_id(img_path)
        
        if unique_vid_key not in video_dict:
            video_dict[unique_vid_key] = []
        video_dict[unique_vid_key].append(img_path)

    unique_videos = list(video_dict.keys())
    random.shuffle(unique_videos)
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if mode == 'training':
        split_idx = int(len(unique_videos) * (1 - VAL_SPLIT))
        train_vids = unique_videos[:split_idx]
        val_vids = unique_videos[split_idx:]

        train_paths, val_paths = [], []
        for vid in train_vids: train_paths.extend(video_dict[vid])
        for vid in val_vids: val_paths.extend(video_dict[vid])

        print(f"[{dataset_model.upper()}] Training: {len(train_paths)} ảnh | Validation: {len(val_paths)} ảnh")

        train_dataset = DatasetClass(train_paths, transform=train_transforms)
        val_dataset = DatasetClass(val_paths, transform=val_test_transforms)

        train_labels = [train_dataset.get_label(p) for p in train_paths]
        class_counts = [train_labels.count(0), train_labels.count(1)]
        
        class_weights = [1.0 / (c if c > 0 else 1) for c in class_counts]
        sample_weights = [class_weights[label] for label in train_labels]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        return train_loader, val_loader

    elif mode == 'testing':
        print(f"[{dataset_model.upper()}] Full Testing: {len(all_images)} images")
        test_dataset = DatasetClass(all_images, transform=val_test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
        return test_loader