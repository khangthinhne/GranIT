import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import os
import timm
from tqdm import tqdm
import csv

from data_preparation.dataset import get_dataloaders
from utils import AdvancedEarlyStopping 

class BaseBaselinePipeline:
    def __init__(self, dataset_name='faceforensic++', batch_size=16, epochs=30, lr=1e-4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        
        self.model_name = self.__class__.__name__.replace("Pipeline", "").lower()
        self.save_dir = './checkpoints'
        self.best_weight_path = os.path.join(self.save_dir, f"{self.model_name}_BEST_AUC.pth")
        
        self.model = self.build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self):
        raise NotImplementedError("Pls implement at child class")

    def train(self):
        print(f"\n[TRAIN] Training {self.model_name.upper()} on {self.dataset_name.upper()}")
        train_loader, val_loader = get_dataloaders(mode='training', batch_size=self.batch_size, dataset_model=self.dataset_name)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        early_stopping = AdvancedEarlyStopping(patience=5, save_dir=self.save_dir, model_name=self.model_name)
        scaler = torch.cuda.amp.GradScaler()

        # File log
        os.makedirs(self.save_dir, exist_ok=True)
        log_file = os.path.join(self.save_dir, f"{self.model_name}_training_log.csv")
        with open(log_file, mode='w', newline='') as f:
            f.write("Epoch,Train_Loss,Val_Loss,Val_AUC\n")

        for epoch in range(self.epochs):
            # --- TRAIN ---
            self.model.train()
            train_loss, train_total = 0.0, 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [TRAIN]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                    
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item() * images.size(0)
                train_total += labels.size(0)
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
                
            epoch_train_loss = train_loss / train_total

            # --- VALIDATION ---
            self.model.eval()
            val_loss, val_total = 0.0, 0
            all_labels, all_probs = [], []
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [VAL]", leave=False):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    with torch.cuda.amp.autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                        
                    val_loss += loss.item() * images.size(0)
                    val_total += labels.size(0)
                    
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
            epoch_val_loss = val_loss / val_total
            
            try:
                epoch_val_auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                epoch_val_auc = 0.0

            scheduler.step()

            print(f"Epoch {epoch+1}: Train Loss {epoch_train_loss:.4f} | Val Loss {epoch_val_loss:.4f} | Val AUC {epoch_val_auc:.4f}")

            with open(log_file, mode='a', newline='') as f:
                f.write(f"{epoch+1},{epoch_train_loss:.4f},{epoch_val_loss:.4f},{epoch_val_auc:.4f}\n")

            early_stopping(epoch_val_loss, epoch_val_auc, self.model)
            if early_stopping.early_stop:
                print(" EARLY STOPPING activated!")
                break

    def evaluate(self, target_dataset):
        print(f"\n🔍 [EVAL] Evaluating {self.model_name.upper()} ong {target_dataset.upper()}")
        
        if not os.path.exists(self.best_weight_path):
            print(f"No file weights {self.best_weight_path}")
            return

        self.model.load_state_dict(torch.load(self.best_weight_path, map_location=self.device))
        self.model.eval()

        test_loader = get_dataloaders(mode='testing', batch_size=self.batch_size, dataset_model=target_dataset)
        
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    
                probs = torch.softmax(logits, dim=1)[:, 1]
                _, preds = torch.max(logits, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0

        print(f"RESULT {self.model_name.upper()} on {target_dataset.upper()}: Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}")
        
        with open("evaluation_baselines.txt", "a") as f:
            f.write(f"{self.model_name.upper()} | Train: {self.dataset_name} | Test: {target_dataset} | Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}\n")


class XceptionPipeline(BaseBaselinePipeline):
    def build_model(self):
        return timm.create_model('xception', pretrained=True, num_classes=2)

class EffNetB4Pipeline(BaseBaselinePipeline):
    def build_model(self):
        return timm.create_model('tf_efficientnet_b4', pretrained=True, num_classes=2)

class ResNet50Pipeline(BaseBaselinePipeline):
    def build_model(self):
        return timm.create_model('resnet50', pretrained=True, num_classes=2)
    
