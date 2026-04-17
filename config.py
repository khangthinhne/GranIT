'''
Config Variables using for initializing, training, evaluating the model.
'''
import torch

# System & Path
BACKBONE_NAME = "vit_base_patch16_clip_224.laion2b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data/faces_processed"
SAVE_MODEL_DIR = "./checkpoints"
LOG_DIR = "./logs/"

# Training Hyperparameters
MODEL_NAME = "GranIT"
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.05

# Settings (ViT & LoRA)
IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768
LORA_RANK = 8
LORA_ALPHA = 16

# AFC (Local Branch)
BETA = 1.5
SCALE_MIN = 0.4
SCALE_MAX = 0.9

# Loss 
LAMBDA_SCALE = 0.1
LAMBDA_TRANS = 0.1



