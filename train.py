import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

from dataset import get_dataloaders
from model import GranIT 
BATCH_SIZE = 16     
EPOCHS = 25
LEARNING_RATE = 2e-5

LAMBDA_TRANS = 0.1    
LAMBDA_SCALE = 0.1    

def loss_function(logits, labels, theta, criterion_ce):
    loss_ce = criterion_ce(logits, labels)
    
    s_x = theta[:, 0, 0]
    s_y = theta[:, 1, 1]
    t_x = theta[:, 0, 2]
    t_y = theta[:, 1, 2]
    
    # loss trans
    loss_trans = (t_x**2 + t_y**2).mean()
    
    # loss scale (0.4, 1)
    loss_scale = (torch.relu(0.4 - s_x) + torch.relu(s_x - 1) +
                  torch.relu(0.4 - s_y) + torch.relu(s_y - 1)).mean()
    
    total_loss = loss_ce + LAMBDA_TRANS * loss_trans + LAMBDA_SCALE * loss_scale
    
    return total_loss, loss_ce, loss_trans, loss_scale

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data preparation
    data_dir = "./data/faces_processed" 
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=BATCH_SIZE)

    # Model init
    model = GranIT()
    model = model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler()
    
    # Warm up
    warmup_epochs = 2 
    total_epochs = EPOCHS
    if total_epochs <= warmup_epochs: warmup_epochs = 0

    def warmup_lambda(current_epoch: int):
        if current_epoch < warmup_epochs: 
            return float(current_epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)

    if total_epochs > warmup_epochs:
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    else:
        scheduler = warmup_scheduler

    # file log
    log_file = "training_log_GranIT_vit.csv"
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'Val_AUC'])

    best_auc = 0.0 

    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"TRAIN Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            # Ép kiểu Float16 
            with torch.cuda.amp.autocast():
                logits, theta = model(images)
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

        print("Đang đánh giá trên tập Validation...")
        with torch.no_grad(): 
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.cuda.amp.autocast():
                    logits, theta = model(images)
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
        print(f"  -> TRAIN: Acc {epoch_train_acc:.2f}% | Loss {epoch_train_loss:.4f}")
        print(f"  -> VAL  : Acc {epoch_val_acc:.2f}% | Loss {epoch_val_loss:.4f} | AUC: {epoch_val_auc:.4f}")

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{epoch_train_loss:.4f}", f"{epoch_train_acc:.2f}", 
                             f"{epoch_val_loss:.4f}", f"{epoch_val_acc:.2f}", f"{epoch_val_auc:.4f}"])

        if epoch_val_auc > best_auc:
            best_auc = epoch_val_auc
            best_model_name = "GranIT_vit_lora_BEST.pth"
            torch.save(model.state_dict(), best_model_name)
            print(f"Save new model with best AUC: {best_auc:.4f} -> {best_model_name}")
            
        print("-" * 50)

    print(f"Done training phase, best AUC: {best_auc:.4f}")

if __name__ == "__main__":
    train_model()