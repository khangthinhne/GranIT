import torch
import torch.nn as nn
import timm

class LoRA(nn.Module):
    '''
    LoRA: Low-Rank Adaption -> reduce computational cost
    '''
    def __init__(self, in_features, out_features, rank=4, alpha=16 ):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.scale = alpha / rank

    def forward(self, x):
        return (x @ self.A @ self.B) * self.scale
    
class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=16):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRA(in_features=self.original_linear.in_features, out_features=self.original_linear.out_features, rank=rank, alpha=alpha)

    def forward(self, x):
        output_original_linear = self.original_linear(x)
        lora = self.lora(x)
        return output_original_linear + lora
    
def merge_lora(backbone: nn.Module, rank=4, alpha=16):
    for block in backbone.blocks:
        block.attn.qkv = LoRALinear(original_linear=block.attn.qkv, rank=rank, alpha=alpha)
        block.attn.proj = LoRALinear(original_linear=block.attn.proj, rank=rank, alpha=alpha)
        block.mlp.fc1 = LoRALinear(original_linear=block.mlp.fc1, rank=rank, alpha=alpha)
        block.mlp.fc2 = LoRALinear(original_linear=block.mlp.fc2, rank=rank, alpha=alpha)

    for param in backbone.parameters():
        param.requires_grad = False

    for name, param in backbone.named_parameters():
        if 'lora' in name or 'norm' in name or 'head' in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"[LoRA] Trainable parameters (LoRA + MLP): {trainable_params:,}")
    print("Applied LoRA to module")