import torch
import torch.nn as nn
from model import BaselineViT, MicroBranch
from config import BACKBONE_NAME
from modules import AFC, CascadedCrossScaleInterrogation
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
 

class GranIT_GlobalOnly(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.global_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        # Chỉ lấy [CLS] token của ảnh gốc
        _, cls_global, _ = self.global_vit(x_wide, return_tokens=True)
        logits = self.mlp(cls_global)
        
        return logits, None, None


class GranIT_LocalOnly(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.afc_module = AFC(crop_size=(224, 224))
        self.local_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        x_crop, theta = self.afc_module(x_wide)
        # Bỏ qua patch_tokens, chỉ lấy [CLS] token của vùng mặt cục bộ để phân loại
        _, cls_local, _ = self.local_vit(x_crop, return_tokens=True)
        
        logits = self.mlp(cls_local)
        return logits, theta, None


class GranIT_MicroOnly(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.afc_module = AFC(crop_size=(224, 224))
        self.micro_branch = MicroBranch(model_name=BACKBONE_NAME)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        x_crop, theta = self.afc_module(x_wide)
        
        # micro_branch hiện tại trả về patch_tokens (shape: B, N, Embed_dim)
        patch_tokens_micro = self.micro_branch(x_crop)
        
        # Global Average Pooling (GAP) 
        # lên các patch tokens để tạo ra 1 vector đại diện cho toàn bộ ảnh tần số
        micro_features = patch_tokens_micro.mean(dim=1) # Shape: (B, Embed_dim)
        
        logits = self.mlp(micro_features)
        return logits, theta, None



class GranIT_Local_Micro(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.afc_module = AFC(crop_size=(224, 224))
        
        self.local_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.micro_branch = MicroBranch(model_name=BACKBONE_NAME)
        
        # Vẫn giữ Interrogator để Cross-Attention giữa Local và Micro
        self.interrogator = CascadedCrossScaleInterrogation(embed_dim=embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        x_crop, theta = self.afc_module(x_wide)
        
        # Lấy cả [CLS] và patch_tokens từ Local
        _, cls_local, patch_tokens_local = self.local_vit(x_crop, return_tokens=True)
        patch_tokens_micro = self.micro_branch(x_crop)
        
        # Thay vì dùng cls_global làm Query cho Interrogator, ta dùng cls_local
        cls_local = cls_local.unsqueeze(1) 
        
        # Interrogate giữa Local và Micro
        z_final, attn_weights = self.interrogator(cls_local, patch_tokens_local, patch_tokens_micro)
        z_final_flat = z_final.squeeze(1)
        
        logits = self.mlp(z_final_flat)
        return logits, theta, attn_weights
    
class GranIT_Global_Local(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.global_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, batch_first=True)

        self.global_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.local_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        # STN
        self.afc_module = AFC(crop_size=(224, 224))

    def forward(self, x_wide):
            # GLOBAL BRANCH
        _, cls_global, _ = self.global_vit(x_wide, return_tokens=True)
        cls_global = cls_global.unsqueeze(1)
        # LOCAL BRANCH
        # AFC cropped the 224x224 region
        x_crop, theta = self.afc_module(x_wide)
        #Take the patch tokens for query section
        _, _, patch_tokens_local = self.local_vit(x_crop, return_tokens=True) 
        z_final, attn_weights = self.cross_attn(query=cls_global, 
                                            key=patch_tokens_local, 
                                            value=patch_tokens_local)
        z_final_flat = z_final.squeeze(1) # Xóa chiều sequence thừa: [B, embed_dim]
        logits = self.mlp(z_final_flat)
        
        return logits, theta, attn_weights

class GranIT_Margin(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        _, cls, _ = self.vit(x, return_tokens=True)
        logits = self.mlp(cls)
        return logits, None, None

def get_args():
    parser = argparse.ArgumentParser(description='GranIT - Granularity-Adaptive Interrogation Transformer')
    parser.add_argument('--ablation_model', type=str, choices=['only_global', 'only_local', 'only_micro', 'local_micro', 'global_local', 'margin'])
    parser.add_argument('--crop_margin', type=float, default=1.5)
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE)
    parser.add_argument('--data_dir', type=str, default=config.DATA_DIR)
    parser.add_argument('--save_name', type=str, default=config.MODEL_NAME)
    return parser.parse_args()

def loss_function(logits, labels, theta, criterion_ce):
    loss_ce = criterion_ce(logits, labels)
    if theta is None:
        return loss_ce, loss_ce, torch.tensor(0.0), torch.tensor(0.0)
    
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

def train_model():
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    args = get_args()
    # Data preparation
    config.BETA = args.crop_margin
    config.BATCH_SIZE = args.batch_size
    print(f"BETA = {config.BETA}")
    print(f"BATCH SIZE: {config.BATCH_SIZE}")
    data_dir = args.data_dir 
    train_loader, val_loader = get_dataloaders(
        mode='training', 
        batch_size=config.BATCH_SIZE, 
        dataset_model='faceforensic++',
        crop_margin=args.crop_margin
    )

    # Model init
    model = get_model(args=args)
    
    model = model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=config.LEARNING_RATE, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss() 
    scaler = torch.cuda.amp.GradScaler()
    if args.ablation_model == 'margin':
        model_name = f"{model.__class__.__name__}_{config.BETA}"
    else:
        model_name = f"{model.__class__.__name__}"
    early_stopping = DualEarlyStopping(
        patience=5, 
        delta=0.0001, 
        save_dir=config.SAVE_MODEL_DIR, 
        model_name=model_name
    )
    print(f"\nStart training on {model.__class__.__name__} branch...")
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
    log_path = f"training_log_{args.ablation_model}_model.csv"
    if args.ablation_model == 'margin':
        log_path = f"training_log_{args.ablation_model}_{config.BETA}_model.csv"
    log_file = os.path.join(config.LOG_DIR, log_path)
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
                probs = torch.softmax(logits.float(), dim=1)[:, 1] # Probability
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
        if epoch_val_auc > best_auc: 
            best_auc = epoch_val_auc
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