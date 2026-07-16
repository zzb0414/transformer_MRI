import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention_block import multi_head_self_attn

class encoder_block(nn.Module):
    """
    An encoder block class composed of a attention block, residual connections, normalization layers and a FFN.
    """

    def __init__(self, d_model, num_heads, dropout=0.0):
        """
        A sequential connection of a self attention block and a FFN.

        Args:
        d_model (int):          Model dimension.
        num_heads (int):        Number of attention heads.
        dropout (float):        Dropout rate.
        """
        super().__init__()

        self.self_attn = multi_head_self_attn(d_model=d_model, num_heads=num_heads)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),\
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )

        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        # Useful info
        self.d_model = d_model

        return
    
    def forward(self, input, omega):
        """
        Args:

        input (tensor):         [batch_size, seq_length, d_model]
        omega (tensor):         Sampling mask, 1: sampled, 0: not sampled. [batch_size, seq_length]
        """
        attention = self.self_attn(input, omega)
        input = self.LN1(input + self.dropout(attention))

        feed_forward = self.FFN(input)
        output = self.LN2(input + self.dropout(feed_forward))

        return output
    
    def flops(self):
        flops = 0
        flops += self.self_attn.flops()
        flops += 2 * 4 * self.d_model ** 2 # Two MLP's in the FFN.

        return flops