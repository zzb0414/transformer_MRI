"""
Attention blocks.
Author: Zhibo Zhu. Date: 07/16/2026.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class multi_head_self_attn(nn.Module):
    """
    A attention block class consisting of self attention using the multi heads attention.
    """
    def __init__(self, d_model, num_heads):
        """
        A multi head self attention.

        Args:
        d_model (int):          Model dimension.
        num_heads (int):        Number of attention heads.
        """
        super().__init__()

        # Queries, keys and values
        self.W_QKV = nn.Linear(in_features=d_model, out_features=3 * d_model)

        # Fusion layer
        self.W_O = nn.Linear(in_features=d_model, out_features=d_model)

        # Useful parameters
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        return
    
    def forward(self, input):
        """
        Args:

        input (tensor):         [batch_size, seq_length, d_model]
        omega (tensor):         Sampling mask, 1: sampled, 0: not sampled. [batch_size, seq_length]
        """
        batch_size, seq_length, _ = input.size()
        # omega = omega.unsqueeze(1).unsqueeze(2)

        # Calculate Q, K and V.
        QKV = self.W_QKV(input) # [batch_size, seq_length, 3 * d_model]
        QKV = QKV.view(batch_size, seq_length, 3, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4) # [3, batch_size, num_heads, seq_length, d_k]
        Q, K, V = QKV[0], QKV[1], QKV[2] # [batch_size, num_heads, seq_length, d_k]

        # Attention scores.
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # [batch_size, num_heads, seq_length, seq_length]

        # omega is not needed as input is undersampled.
        # if omega is not None:
        #     attention_scores = attention_scores.masked_fill(omega==0, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Multiply with V.
        out = torch.matmul(attention_scores, V) # [batch_size, num_heads, seq_length, d_k]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

        return self.W_O(out)
    
    def flops(self):
        flops = 0
        flops += 4 * self.d_model ** 2

        return flops
    

class multi_head_cross_attn(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        A multi head cross attention.

        Args:
        d_model (int):          Model dimension.
        num_heads (int):        Number of heads.
        """
        super().__init__()

        # Queries, keys and values.
        self.W_Q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_KV = nn.Linear(in_features=d_model, out_features=2 * d_model)

        # Fusion layer.
        self.W_O = nn.Linear(in_features=d_model, out_features=d_model)

        # Useful parameters.
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        return

    def forward(self, PE_p, O):
        """
        Args:

        PE_p (tensor):          Positional encodings to be queried, [batch_size, seq_length1, d_model]
        O (tensor):             Output of the encoder block, [batch_size, seq_length2, d_model]
        """
        batch_size, seq_length1, _ = PE_p.size()
        _, seq_length2, _ = O.size()

        # Calculate Q, K and V
        Q = self.W_Q(PE_p) # [batch_size, seq_length1, d_model]
        KV = self.W_KV(O) # [batch_size, seq_length2, 2 * d_model]
        Q = Q.view(batch_size, seq_length1, self.num_heads, self.d_k).permute(0, 2, 1, 3) # [batch_size, num_heads, seq_length1, d_k]
        KV = KV.view(batch_size, seq_length2, 2, self.num_heads, self.d_k).permute(2, 0, 3, 1, 4) # [2, batch_size, num_heads, seq_length2, d_k]
        K, V = KV[0], KV[1] # [batch_size, num_heads, seq_length2, d_k]

        # Calculate attention scores.
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # [batch_size, num_head, seq_length1, seq_length2]
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Multiply V.
        out = torch.matmul(attention_scores, V) # [batch_size, num_heads, seq_length1, d_k]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length1, self.d_model)

        return self.W_O(out)
    
    def flops(self):
        flops = 0
        flops += 4 * self.d_model ** 2 # W_Q, W_K, W_V and W_O

        return flops