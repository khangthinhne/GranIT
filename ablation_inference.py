import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse
from datetime import datetime
import csv
import re
from data_preparation.dataset import get_dataloaders
from model import GranIT
import config
from ablation_training import GranIT_Global_Local, GranIT_GlobalOnly, GranIT_Local_Micro, GranIT_LocalOnly, GranIT_MicroOnly, GranIT_Margin
def get_args():
    parser = argparse.ArgumentParser(description='Evaluate GranIT Model on cross datasets')
    parser.add_argument('--ablation_model', type=str, choices=['only_global', 'only_local', 'only_micro', 'local_micro', 'global_local', 'margin'])
    parser.add_argument('--crop_margin', type=float, default=1.5)
    parser.add_argument('--dataset', type=str, default='faceforensic++', choices=['faceforensic++', 'celebdf', 'wilddf', 'dfdc'], help='Target dataset for evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights (.pth)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--vis_dir', type=str, default='./visualizations/ablation', help='Directory to save visualization images')
    parser.add_argument('--log_file', type=str, default='evaluation_ablation_results.txt', help='Path to save evaluation metrics')
    parser.add_argument('--num_vis', type=int, default=10, help='Number of images to visualize and save')
    return parser.parse_args()

def denormalize_image(tensor):
    img_np = tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1) * 255.0
    return cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)

def get_stn_box(t_x, t_y, s_x, s_y, W, H):
    cx = int((t_x + 1.0) / 2.0 * W)
    cy = int((t_y + 1.0) / 2.0 * H)
    box_w = int(s_x * W)
    box_h = int(s_y * H)

    x1 = max(0, cx - box_w // 2)
    y1 = max(0, cy - box_h // 2)
    x2 = min(W, cx + box_w // 2)
    y2 = min(H, cy + box_h // 2)
    return x1, y1, x2, y2

def log_results(log_file, dataset_name, model_name, metrics):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"\n[{timestamp}] Model: {model_name} | Dataset: {dataset_name}\n | Crop Margin (Beta): {config.BETA}" 
    log_entry += "-" * 50 + "\n"
    for key, value in metrics.items():
        if isinstance(value, float):
            log_entry += f"{key:<15}: {value:.4f}\n"
        else:
            log_entry += f"{key:<15}: {value}\n"
    log_entry += "-" * 50 + "\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)

def log_results_csv(log_file, model_name, test_beta, metrics):
    csv_file = log_file.replace('.txt', '.csv')
    
    match = re.search(r'Margin_([0-9\.]+)', model_name)
    train_beta = match.group(1) if match else "Unknown"
    
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Train_Beta', 'Test_Beta', 'AUC', 'Accuracy'])
        
        writer.writerow([train_beta, test_beta, f"{metrics['AUC Score']:.4f}", f"{metrics['Accuracy']:.4f}"])
        
def get_model(args):
    if args.ablation_model == 'only_global':
        model = GranIT_GlobalOnly()
    elif args.ablation_model == 'only_local':
        model = GranIT_LocalOnly()
    elif args.ablation_model == 'only_micro':
        model = GranIT_MicroOnly()
    elif args.ablation_model == 'local_micro':
        model = GranIT_Local_Micro()
    elif args.ablation_model == 'global_local':
        model = GranIT_Global_Local()
    elif args.ablation_model == 'margin':
        model = GranIT_Margin()
    else:
        raise ValueError(f'There is no ablation models named {args.ablation_model}!')
    
    print(f"Init [{model.__class__.__name__}] model")
    return model
def evaluate_and_visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device.type.upper()}...")

    test_loader = get_dataloaders(mode='testing', batch_size=args.batch_size, dataset_model=args.dataset, crop_margin=args.crop_margin)
    # Model init
    model = get_model(args=args)
    model = model.to(device)
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded weights from {args.model_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    os.makedirs(args.vis_dir, exist_ok=True)
    images_saved = 0

    print(f"Running inference on {args.dataset.upper()} dataset...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            with torch.cuda.amp.autocast():
                logits, theta, attn_weight = model(images)
            
            probs = torch.softmax(logits.float(), dim=1)[:, 1] 
            _, preds = torch.max(logits, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if images_saved < args.num_vis:
                for j in range(images.size(0)):
                    if images_saved >= args.num_vis: 
                        break
                    img_bgr = denormalize_image(images[j])
                    is_correct = labels[j].item() == preds[j].item()
                    color = (0, 255, 0) if is_correct else (0, 0, 255)
                    
                    if theta is not None:
                        t = theta[j].cpu().numpy()
                        s_x, s_y = t[0, 0], t[1, 1]
                        t_x, t_y = t[0, 2], t[1, 2]

                        H, W, _ = img_bgr.shape
                        
                        x1, y1, x2, y2 = get_stn_box(t_x, t_y, s_x, s_y, W, H)

                        
                        
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 3)
                    
                    true_txt = "Real" if labels[j].item() == 0 else "Fake"
                    pred_txt = "Real" if preds[j].item() == 0 else "Fake"
                    cv2.putText(img_bgr, f"T:{true_txt}|P:{pred_txt}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    filename = f"{args.dataset}_vis_{images_saved:03d}_{'correct' if is_correct else 'wrong'}.jpg"
                    cv2.imwrite(os.path.join(args.vis_dir, filename), img_bgr)
                    images_saved += 1

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    metrics = {
        "Total Samples": len(all_labels),
        "Accuracy": acc,
        "AUC Score": auc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    print("\n" + "="*50)
    print(f"EVALUATION RESULTS: {args.dataset.upper()}")
    print("="*50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<15}: {v:.4f}")
        else:
            print(f"{k:<15}: {v}")
    print("="*50)
    
    log_results(args.log_file, args.dataset, os.path.basename(args.model_path), metrics)
    log_results_csv(args.log_file, os.path.basename(args.model_path), config.BETA, metrics)
    print(f"Metrics saved to {args.log_file}")
    print(f"Saved {images_saved} visualization images to {args.vis_dir}")

if __name__ == "__main__":
    args = get_args()
    config.BETA = args.crop_margin
    config.BATCH_SIZE = args.batch_size
    evaluate_and_visualize(args)