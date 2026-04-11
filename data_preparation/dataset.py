import os
import glob
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

# Assuming these are imported correctly from your actual project
from data_preparation.dataset_models import (
    FaceForensicsDataset, CelebDFDataset, WildDeepfakeDataset, 
    DFDCDataset, FFDeepfakesDataset, FFFace2FaceDataset, 
    FFFaceSwapDataset, FFNeuralTexturesDataset
)

DATASET_REGISTRY = {
    'faceforensic++': FaceForensicsDataset,
    'celebdf': CelebDFDataset,
    'wilddf': WildDeepfakeDataset,
    'dfdc': DFDCDataset,
    'ff_df': FFDeepfakesDataset,
    'ff_f2f': FFFace2FaceDataset,
    'ff_fs': FFFaceSwapDataset,
    'ff_nt': FFNeuralTexturesDataset
}

DATA_DIR = {
    'celebdf': 'data/celebdf_processed',
    'wilddf': 'data/wilddf_processed',
    'dfdc': 'data/DFDC'
}

VAL_SPLIT = 0.2

def get_dataloaders(mode='training', batch_size=8, dataset_model='faceforensic++', crop_margin=1.5, num_workers=4, seed=42):
    """
    mode: 'training' -> (train_loader, val_loader),  'testing' -> (test_loader)
    """
    if dataset_model not in DATASET_REGISTRY:
        raise ValueError(f"Need to check dataset registry: {list(DATASET_REGISTRY.keys())}")

    ff_variants = ['faceforensic++', 'ff_df', 'ff_f2f', 'ff_fs', 'ff_nt']
    
    # [FIX 1] Safely determine dir_path without triggering KeyError
    if dataset_model in ff_variants:
        if mode == 'testing':
            print(f"Testing on {dataset_model.upper()}")
            dir_path = 'data/faces_processed_split/test'
        else:
            dir_path = 'data/faces_processed_split/train'
    else:
        dir_path = DATA_DIR[dataset_model]
        # [FIX 2] Warn about testing on unsplit data
        if mode == 'testing':
            print(f"WARNING: No explicit test directory for {dataset_model.upper()}. Loading ALL images from {dir_path}!")

    DatasetClass = DATASET_REGISTRY[dataset_model]
    exts = ["*.jpg", "*.jpeg", "*.png"]
    all_images = []
    for ext in exts:
        all_images.extend(
            glob.glob(os.path.join(dir_path, "**", ext), recursive=True)
        )
        
    if not all_images:
        raise ValueError(f"No images found in {dir_path}")

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if mode == 'training':
        # VIDEO-LEVEL SPLIT 
        video_dict = {}
        for img_path in all_images:
            unique_vid_key = DatasetClass.extract_video_id(img_path)
            if unique_vid_key not in video_dict:
                video_dict[unique_vid_key] = []
            video_dict[unique_vid_key].append(img_path)

        unique_videos = list(video_dict.keys())
        
        # [FIX 3] Add seed for reproducible splits
        random.seed(seed)
        random.shuffle(unique_videos)

        split_idx = int(len(unique_videos) * (1 - VAL_SPLIT))
        train_vids = unique_videos[:split_idx]
        val_vids = unique_videos[split_idx:]
        # Thêm vào get_dataloaders sau khi split
        train_vids_set = set(train_vids)
        val_vids_set = set(val_vids)
        overlap = train_vids_set & val_vids_set
        print(f"Overlap videos: {len(overlap)}")  # Phải = 0
        
        train_paths, val_paths = [], []
        for vid in train_vids: train_paths.extend(video_dict[vid])
        for vid in val_vids: val_paths.extend(video_dict[vid])

        print(f"[{dataset_model.upper()}] Training: {len(train_paths)} images | Validation: {len(val_paths)} images")

        train_dataset = DatasetClass(train_paths, transform=train_transforms, crop_margin=crop_margin)
        val_dataset = DatasetClass(val_paths, transform=val_test_transforms, crop_margin=crop_margin)

        # [FIX 4] Optimize class weighting calculation
        train_labels = [train_dataset.get_label(p) for p in train_paths]
        counts = Counter(train_labels)
        class_counts = [counts.get(0, 0), counts.get(1, 0)]
        print(f"Class counts - Real: {class_counts[0]}, Fake: {class_counts[1]}")
        class_weights = [1.0 / (c if c > 0 else 1) for c in class_counts]
        sample_weights = [class_weights[label] for label in train_labels]
        
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        return train_loader, val_loader

    elif mode == 'testing':
        print(f"[{dataset_model.upper()}] Full Testing: {len(all_images)} images")
        test_dataset = DatasetClass(all_images, transform=val_test_transforms, crop_margin=crop_margin)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        return test_loader
