import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    LoRA wrapper for an nn.Linear layer.
    Approximates the weight update as \Delta W = B * A
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        # Freeze the original pre-trained backbone layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
            
        # Initialize low-rank adapter matrices
        # A reduces dimension from in_features -> rank
        # B expands dimension from rank -> out_features
        self.lora_A = nn.Parameter(torch.empty(self.linear.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.empty(self.rank, self.linear.out_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        # As per the LoRA paper: initialize A with random Gaussian/Kaiming, B with zeros.
        # This ensures the initial \Delta W is 0, keeping initial outputs identical to the base model.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):

        base_out = self.linear(x)
  
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


class LoRAConv2d(nn.Module):
    """
    LoRA wrapper for an nn.Conv2d layer.
    """
    def __init__(self, conv_layer: nn.Conv2d, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.conv = conv_layer
        self.rank = rank
        self.scaling = alpha / rank
        
        # Freeze the original layer
        self.conv.weight.requires_grad = False
        if self.conv.bias is not None:
            self.conv.bias.requires_grad = False
            
        # For convolutions, we represent A and B as sequential convolutions
        self.lora_A = nn.Conv2d(self.conv.in_channels, self.rank, 
                                kernel_size=self.conv.kernel_size, 
                                stride=self.conv.stride, 
                                padding=self.conv.padding, bias=False)
        self.lora_B = nn.Conv2d(self.rank, self.conv.out_channels, 
                                kernel_size=1, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        base_out = self.conv(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out