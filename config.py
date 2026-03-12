'''
Config Variables using for initializing, training, evaluating the model.
'''
import torch

# System & Path
BACKBONE_NAME = "vit_base_patch16_clip_224.laion2b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./datasets/FaceForensics++"
SAVE_MODEL_DIR = "./checkpoints/"
LOG_DIR = "./logs/"

# Training Hyperparameters
MODEL_NAME = "GranIT"
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05

# Settings (ViT & LoRA_
IMAGE_SIZE = 224
PATCH_SIZE = 16
EMBED_DIM = 768
LORA_RANK = 8
LORA_ALPHA = 16

# AFC (Local Branch)
MARGIN_BETA = 1.3
SCALE_MIN = 0.4
SCALE_MAX = 0.9

# Loss 
LAMBDA_SCALE = 0.1
LAMBDA_TRANS = 0.1