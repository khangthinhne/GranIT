import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
import torch.backends.cudnn as cudnn

from data_preparation.dataset import get_dataloaders
from model import GranIT 
from modules import DualEarlyStopping
import config
import argparse
 
def get_args():
    parser = argparse.ArgumentParser(description='GranIT - Granularity-Adaptive Interrogation Transformer')
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR)
    parser.add_argument('--save_name', type=str, default=config.MODEL_NAME)
    return parser.parse_args()

def loss_function(logits, labels, theta, criterion_ce):
    loss_ce = criterion_ce(logits, labels)
    
    s_x = theta[:, 0, 0]
    s_y = theta[:, 1, 1]
    t_x = theta[:, 0, 2]
    t_y = theta[:, 1, 2]
    
    # loss trans
    loss_trans = (t_x**2 + t_y**2).mean() # cho ngay xung quanh tâm thoi
    
    # loss scale (0.4, 0.9)
    loss_scale = (torch.relu(config.SCALE_MIN - s_x) + torch.relu(s_x - config.SCALE_MAX) +
                  torch.relu(config.SCALE_MIN - s_y) + torch.relu(s_y - config.SCALE_MAX)).mean()
    
    total_loss = loss_ce + config.LAMBDA_TRANS * loss_trans + config.LAMBDA_SCALE * loss_scale
    
    return total_loss, loss_ce, loss_trans, loss_scale

def train_model():
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    args = get_args()
    # Data preparation
    data_dir = args.data_dir 
    train_loader, val_loader = get_dataloaders(
        mode='training', 
        batch_size=config.BATCH_SIZE, 
        dataset_model='faceforensic++'
    )

    # Model init
    model = GranIT()
    model = model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=config.LEARNING_RATE, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss() 
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = DualEarlyStopping(
        patience=5, 
        delta=0.0001, 
        save_dir=config.SAVE_MODEL_DIR, 
        model_name=config.MODEL_NAME
    )
    # Warm up
    warmup_epochs = 2 
    total_epochs = config.EPOCHS
    if total_epochs <= warmup_epochs: warmup_epochs = 0

    def warmup_lambda(current_epoch: int):
        if current_epoch < warmup_epochs: 
            return float(current_epoch + 1) /   warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    if total_epochs > warmup_epochs:
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = warmup_scheduler

    # file log
    log_file = os.path.join(config.LOG_DIR,f"training_log_{config.MODEL_NAME}.csv")
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'Val_AUC'])

    best_auc = 0.0 

    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"TRAIN Epoch {epoch+1}/{config.EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Ép kiểu Float16 
            with torch.cuda.amp.autocast():
                logits, theta, att_weight = model(images)
                loss, _, _, _ = loss_function(logits, labels, theta, criterion_ce)
            
            # Backprop qua GradScaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Thống kê
            running_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({'T_Loss': f"{loss.item():.3f}", 'T_Acc': f"{100.*correct/total:.1f}%"})

        scheduler.step() 
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct / total
        
        torch.cuda.empty_cache() 

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        y_true, y_scores = [], []

        print("Validation...")
        with torch.no_grad(): 
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast():
                    logits, theta, att_weight = model(images)
                    loss, _, _, _ = loss_function(logits, labels, theta, criterion_ce)
                    
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1)[:, 1] # Probability
                _, predicted = torch.max(logits, 1)
                
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                y_true.extend(labels.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        try: 
            epoch_val_auc = roc_auc_score(y_true, y_scores)
        except ValueError: 
            epoch_val_auc = 0.0 

        print(f"End EPOCH {epoch+1}:")
        print(f"TRAIN: Acc {epoch_train_acc:.2f}% | Loss {epoch_train_loss:.4f}")
        print(f"VAL  : Acc {epoch_val_acc:.2f}% | Loss {epoch_val_loss:.4f} | AUC: {epoch_val_auc:.4f}")

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{epoch_train_loss:.4f}", f"{epoch_train_acc:.2f}", 
                              f"{epoch_val_loss:.4f}", f"{epoch_val_acc:.2f}", f"{epoch_val_auc:.4f}"])

        early_stopping(epoch_val_loss, epoch_val_auc, model)

        if early_stopping.early_stop:
            print(f"ACTIVATED EARLY STOPPING ở Epoch {epoch+1}!")
            print(f"Reason: Validation loss does not decrease in {early_stopping.patience} epoch liên tiếp.")
            break
            
        print("-" * 50)

    print(f"Done training phase, best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_model()