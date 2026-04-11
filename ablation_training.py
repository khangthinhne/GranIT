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
from collections import Counter

from data_preparation.dataset import get_dataloaders
from model import GranIT 
from modules import DualEarlyStopping, fusion_list
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
        self.afc_module = AFC(crop_size=(224, 224), use_fg_afc=True, use_lhpf=True) 
        self.local_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        x_crop, theta = self.afc_module(x_wide)
        _, cls_local, _ = self.local_vit(x_crop, return_tokens=True)
        logits = self.mlp(cls_local)
        return logits, theta, None


class GranIT_MicroOnly(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512):
        super().__init__()
        self.afc_module = AFC(crop_size=(224, 224), use_fg_afc=True, use_lhpf=True)
        self.micro_branch = MicroBranch(model_name=BACKBONE_NAME, use_lhpf=True) 
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        x_crop, theta = self.afc_module(x_wide)
        patch_tokens_micro = self.micro_branch(x_crop)
        micro_features = patch_tokens_micro.mean(dim=1) # GAP
        logits = self.mlp(micro_features)
        return logits, theta, None


class GranIT_Local_Micro(nn.Module): # (Chính là w/o G-Branch)
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512):
        super().__init__()
        # Cập nhật đồ xịn
        self.afc_module = AFC(crop_size=(224, 224), use_fg_afc=True, use_lhpf=True)
        self.local_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.micro_branch = MicroBranch(model_name=BACKBONE_NAME, use_lhpf=True)
        
        self.interrogator = CascadedCrossScaleInterrogation(embed_dim=embed_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(p=0.2), nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        x_crop, theta = self.afc_module(x_wide)
        _, cls_local, patch_tokens_local = self.local_vit(x_crop, return_tokens=True)
        patch_tokens_micro = self.micro_branch(x_crop)
        
        cls_local = cls_local.unsqueeze(1) 
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
class GranIT_Ablation(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512, 
                 use_lhpf=False, 
                 use_fg_afc=False, 
                 use_coord_inj=False, 
                 use_m_branch=True):
        super().__init__()
        
        # Lưu lại các cờ để dùng trong forward
        self.use_coord_inj = use_coord_inj
        self.use_m_branch = use_m_branch
        
        print(f"--- INIT GRANIT ABLATION ---")
        print(f"M-Branch: {use_m_branch} | LHPF: {use_lhpf} | FG-AFC: {use_fg_afc} | Coord-Inj: {use_coord_inj}")
        
        # 1.  SPATIAL BACKBONE
        self.global_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.local_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)

        # 2. STN (Truyền cờ vào để AFC biết dùng RGB hay Fusion)
        self.afc_module = AFC(crop_size=(224, 224), use_fg_afc=use_fg_afc, use_lhpf=use_lhpf)
        
        # 3. MICRO BRANCH (Chỉ khởi tạo nếu bật)
        if self.use_m_branch:
            self.micro_branch = MicroBranch(model_name=BACKBONE_NAME, use_lhpf=use_lhpf)
        
        # 4. CASCADED CROSS-SCALE INTERROGATOR
        self.interrogator = CascadedCrossScaleInterrogation(embed_dim=embed_dim)

        # 5. COORDINATE INJECTION (Chỉ khởi tạo nếu bật)
        if self.use_coord_inj:
            self.theta_proj = nn.Sequential(
                nn.Linear(6, 64),
                nn.GELU(),
                nn.Linear(64, embed_dim)
            )

        # 6. CLASSIFIER HEAD
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        nn.init.zeros_(self.theta_proj[-1].weight)
        nn.init.zeros_(self.theta_proj[-1].bias)

    def forward(self, x_wide):
        B = x_wide.size(0)

        # ---------------- GLOBAL BRANCH ----------------
        _, cls_global, _ = self.global_vit(x_wide, return_tokens=True) 

        # ---------------- LOCAL BRANCH ----------------
        x_crop, theta = self.afc_module(x_wide)
        _, _, patch_tokens_local = self.local_vit(x_crop, return_tokens=True) 

        # ---------------- MICRO BRANCH ----------------
        patch_tokens_micro = None
        if self.use_m_branch:
            patch_tokens_micro = self.micro_branch(x_crop)

        # ---------------- COORDINATE INJECTION ----------------
        if self.use_coord_inj:
            theta_flat = theta.reshape(B, 6) # Dùng reshape cho an toàn
            theta_emb = self.theta_proj(theta_flat)
            cls_global_aware = cls_global + theta_emb
        else:
            cls_global_aware = cls_global # Bypass, giữ nguyên

        # ---------------- CROSS-SCALE INTERROGATION ----------------
        # Nếu micro_patches=None, module sẽ tự động bypass Stage 1
        z_final, attn_weights = self.interrogator(cls_global_aware, patch_tokens_local, micro_patches=patch_tokens_micro)
        
        logits = self.mlp(z_final)
        return logits, theta, attn_weights


class GranIT_Fusion(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, hidden_dim=512, module='CCSIM'): 
        super().__init__()
            
        # GLOBAL & LOCAL - share backbone
        self.global_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
        self.local_vit = BaselineViT(model_name=BACKBONE_NAME, pretrained=True, num_classes=0)
    
        # STN
        self.afc_module = AFC(crop_size=(224, 224))
        
        # MICRO - Frequecy
        self.micro_branch = MicroBranch(model_name=BACKBONE_NAME)
        
        # cascaded cross-scale interrogate
        self.interrogator = fusion_list[module](embed_dim=embed_dim)

        self.theta_proj = nn.Sequential(
            nn.Linear(6, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim)
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), 
            nn.GELU(),                   
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x_wide):
        B = x_wide.size(0)
        # GLOBAL BRANCH
        _, cls_global, _ = self.global_vit(x_wide, return_tokens=True)
        # cls_global = cls_global.unsqueeze(1)
        # LOCAL BRANCH
        # AFC cropped the 224x224 region
        x_crop, theta = self.afc_module(x_wide)
        #Take the patch tokens for query section
        _, _, patch_tokens_local = self.local_vit(x_crop, return_tokens=True) 

        # MICRO BRANCH
        patch_tokens_micro = self.micro_branch(x_crop)

        theta_flat = theta.view(B, 6)
        theta_emb = self.theta_proj(theta_flat) # [B, 768]
        
        # Add position to [CLS] token Global
        cls_global_aware = cls_global + theta_emb
        #Cross-Attention Mechanism
        z_final, attn_weights = self.interrogator(cls_global_aware, patch_tokens_local, patch_tokens_micro)
        # z_final = z_final.squeeze(1)
        logits = self.mlp(z_final)
        
        return logits, theta, attn_weights

def get_args():
    parser = argparse.ArgumentParser(description='GranIT - Granularity-Adaptive Interrogation Transformer')
    parser.add_argument('--ablation_model', type=str, choices=[
        'only_global', 'only_local', 'only_micro', 'local_micro', 'global_local', 'margin',
        'v2_baseline', 'v2_lhpf', 'v2_fgafc', 'v2_full', 'v2_no_m', 'CCSIM', 'SCF', 'SSAF', 
        'PCAF', 'SCAF'
    ])
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

    elif args.ablation_model == 'v2_baseline':
        model = GranIT_Ablation(use_lhpf=False, use_fg_afc=False, use_coord_inj=False, use_m_branch=True)
    elif args.ablation_model == 'v2_lhpf':
        model = GranIT_Ablation(use_lhpf=True, use_fg_afc=False, use_coord_inj=False, use_m_branch=True)
    elif args.ablation_model == 'v2_fgafc':
        model = GranIT_Ablation(use_lhpf=True, use_fg_afc=True, use_coord_inj=False, use_m_branch=True)
    elif args.ablation_model == 'v2_full':
        model = GranIT_Ablation(use_lhpf=True, use_fg_afc=True, use_coord_inj=True, use_m_branch=True)
    elif args.ablation_model == 'v2_no_m':
        model = GranIT_Ablation(use_lhpf=True, use_fg_afc=True, use_coord_inj=True, use_m_branch=False)
    elif args.ablation_model == 'CCSIM' or args.ablation_model ==  'SCF' or args.ablation_model == 'SSAF' or args.ablation_model =='PCAF' or args.ablation_model =='SCAF':
        model = GranIT_Fusion(module=args.ablation_model)
    else:
        raise ValueError(f'There is no ablation models named {args.ablation_model}!')
    
    print(f"Init [{model.__class__.__name__}] model")
    return model
def get_optimizer(model, args):
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'global_vit' in name or 'local_vit' in name or 'micro_branch' in name:
            backbone_params.append(param)
        else:  # AFC, interrogator, mlp, theta_proj
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": 5e-5},  # từ 2e-5
        {"params": head_params,     "lr": 2e-4},  # từ 1e-4
    ], weight_decay=config.WEIGHT_DECAY)
    
    return optimizer
def train_model(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)
    
    # Data preparation
   
    print(f"BETA = {config.BETA}")
    print(f"BATCH SIZE: {config.BATCH_SIZE}")
    data_dir = args.data_dir 
    train_loader, val_loader = get_dataloaders(
        mode='training', 
        batch_size=config.BATCH_SIZE, 
        dataset_model='faceforensic++',
        crop_margin=args.crop_margin,
        num_workers=config.NUM_WORKERS
    )

    # Model init
    model = get_model(args=args)
    
    model = model.to(device)

    # Optimizer & Loss
    # Trong ablation_training.py, sửa lại
    # Thay toàn bộ block optimizer cũ bằng:
    # Optimizer & Loss
    optimizer = get_optimizer(model, args)

    # CE weights
    # all_labels = [train_loader.dataset.get_label(p) 
    #             for p in train_loader.dataset.image_paths]
    # counts = Counter(all_labels)
    # ce_weights = torch.tensor([1.0/counts[0], 1.0/counts[1]], dtype=torch.float)
    # ce_weights = ce_weights / ce_weights.sum()
    # ce_weights = ce_weights.to(device)
    criterion_ce = nn.CrossEntropyLoss( label_smoothing=0.1)

    scaler = torch.cuda.amp.GradScaler()

    if args.ablation_model == 'margin':
        model_name = f"{args.save_name}_{config.BETA}"
    else:
        model_name = args.save_name

    early_stopping = DualEarlyStopping(
        patience=5, 
        delta=0.0001, 
        save_dir=config.SAVE_MODEL_DIR, 
        model_name=model_name
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    print(f"\nStart training on {model.__class__.__name__} branch...")

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
            # Thêm vào training loop sau forward pass, vài batch đầu
            if epoch >=2 and total <= 64:
                s_x = theta[:, 0, 0].mean().item()
                s_y = theta[:, 1, 1].mean().item()
                t_x = theta[:, 0, 2].mean().item()
                t_y = theta[:, 1, 2].mean().item()
                print(f"STN: sx={s_x:.3f} sy={s_y:.3f} tx={t_x:.3f} ty={t_y:.3f}")
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
    import gc

    base_args = get_args()

    MODELS_TO_TRAIN = {
        # 'CCSIM': 'GranIT_CCSIM', 
        # 'SCF': 'GranIT_SCF', 
        # 'SSAF': 'GranIT_SSAF', 
        # 'PCAF': 'GranIT_PCAF', 
        # 'SCAF': 'GranIT_SCAF'
        'v2_full': 'GranIT'
        # 'only_global': 'only_global'
    }

 

    for model_flag, save_name in MODELS_TO_TRAIN.items():
        print(f"\n\n{'*'*60}")
        print(f"TRAINING MODEL: [{model_flag}] | SAVE NAME: [{save_name}]")
        print(f"{'*'*60}")

        base_args.ablation_model = model_flag
        base_args.save_name = save_name

        train_model(base_args)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n{'='*60}")
    print("DONE TRAINING")
    print(f"{'='*60}")