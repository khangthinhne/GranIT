import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import os
from tqdm import tqdm
import timm 
# from ..data_preparation.dataset import get_dataloaders
from data_preparation.dataset import get_dataloaders
class BaseEvaluator:
    def __init__(self, model_path, dataset_name='celebdf', batch_size=16):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.model_path = model_path
        
        self.model = self.build_model().to(self.device)
        self.load_weights()

    def build_model(self):
        raise NotImplementedError("not implemented")

    def load_weights(self):
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as e:
            print(f"Error load weights: {e}")
            
        self.model.eval()

    def forward_pass(self, images):
        return self.model(images)

    def evaluate(self):
        print(f"Evaluating [ {self.__class__.__name__} ] on [ {self.dataset_name.upper()} ]")
        test_loader = get_dataloaders(mode='testing', batch_size=self.batch_size, dataset_model=self.dataset_name)

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.cuda.amp.autocast():
                    logits = self.forward_pass(images)
                
                probs = torch.softmax(logits, dim=1)[:, 1] 
                _, preds = torch.max(logits, 1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0

        print(f"RESULT {self.__class__.__name__} | {self.dataset_name.upper()}:")
        print(f"   - Accuracy : {acc*100:.2f}%")
        print(f"   - AUC Score: {auc:.4f}")
        print(f"   - F1 Score : {f1:.4f}")
        print("-" * 50)
        
        with open("baseline/evaluation_baselines_results.txt", "a") as f:
            f.write(f"{self.__class__.__name__} | {self.dataset_name} | Acc: {acc:.4f} | AUC: {auc:.4f} | F1: {f1:.4f}\n")


class XceptionEvaluator(BaseEvaluator):
    def build_model(self):
        return timm.create_model('xception', pretrained=False, num_classes=2)

class EffNetB4Evaluator(BaseEvaluator):
    def build_model(self):
        return timm.create_model('efficientnet_b4', pretrained=False, num_classes=2)

class ResNetEvaluator(BaseEvaluator):
    def build_model(self):
        return timm.create_model('resnet50', pretrained=False, num_classes=2)

class GranITEvaluator(BaseEvaluator):
    def build_model(self):
        from model import GranIT
        return GranIT()
        
    def forward_pass(self, images):
        logits, _, _ = self.model(images)
        return logits
    
if __name__ == "__main__":
    datasets_to_test = ['dfdc']
    
    models_to_test = [
        # (XceptionEvaluator, "models/sota_xception_BEST.pth"),
        # (EffNetB4Evaluator, "models/sota_efficientnet_BEST.pth"),
        # (ResNetEvaluator,   "models/sota_resnet50_BEST.pth")
        (GranITEvaluator, "checkpoints/GranIT_BEST_AUC.pth")
    ]
    
    for EvaluatorClass, weight_path in models_to_test:
        if not os.path.exists(weight_path):
            print(f"Pass {EvaluatorClass.__name__} since there is no file: {weight_path}")
            continue
            
        for dataset in datasets_to_test:
            evaluator = EvaluatorClass(model_path=weight_path, dataset_name=dataset, batch_size=16)
            evaluator.evaluate()