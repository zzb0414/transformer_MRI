"""
Decoder block.
Author: Zhibo Zhu. Date: 07/20/2026.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention_block import multi_head_self_attn, multi_head_cross_attn
from refine_module import refine_module

class decoder_block(nn.Module):
    def __init__(self, d_model, num_heads, W, H, dropout=0.0, dtype=torch.float32):
        """
        A sequential connection of a cross attention block, a self attention block, a FFN and a refinement module.

        Args:
        d_model (int):          Model dimension.
        num_heads (int):        Number of attention heads.
        W (int):                Image domain width.
        H (int):                Image domain height.
        dropout (float):        Dropout rate.
        dtype (torch.dtype):    Parameter/compute dtype, e.g. torch.float32 (default), torch.bfloat16, torch.float16.
        """
        super().__init__()

        self.cross_attn = multi_head_cross_attn(d_model, num_heads, dtype=dtype)
        self.self_attn = multi_head_self_attn(d_model, num_heads, dtype=dtype)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, dtype=dtype),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model, dtype=dtype),
        )
        self.rm = refine_module(W, H, d_model, dtype=dtype)

        self.LN1 = nn.LayerNorm(d_model, dtype=dtype)
        self.LN2 = nn.LayerNorm(d_model, dtype=dtype)
        self.LN3 = nn.LayerNorm(d_model, dtype=dtype)

        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model

        return
    
    def forward(self, PE_p, O, LR=True):
        """
        Args:

        PE_p (tensor):          Positional encodings to be queried, [batch_size, seq_length1, d_model]
        O (tensor):             Output of the encoder block, [batch_size, seq_length2, d_model]
        omega (tensor):         Sampling mask, 1: sampled, 0: not sampled. [batch_size, seq_length]
        LR (bool):              Resolution flag: True: LR, False: HR
        """
        batch_size, seq_length, d_model = PE_p.size() # seq_length should be full coordinates of k-space.

        # MHCA and residual connection 1.
        cross_attention = self.cross_attn(PE_p, O)
        cross_attention = self.LN1(PE_p + self.dropout(cross_attention))

        # MHSA and residual connection 2 for LR decoder only. For HR decoder, do nothing (flow forward).
        if LR: # Only LR decoder does self attention.
            # Self attention block does not need sampling mask now.
            # omega = torch.ones(batch_size, seq_length, device=cross_attention.device)
            self_attention = self.self_attn(cross_attention)
            self_attention = self.LN2(cross_attention + self.dropout(self_attention))
        else:
            self_attention = cross_attention

        # FFN and residual connection 3.
        feed_forward = self.FFN(self_attention)
        feed_forward = self.LN3(self_attention + self.dropout(feed_forward))

        # Refinement module for HR decoder only. For LR decoder, do nothing.
        if not LR: # Only HR decoder does refinement.
            rm = self.rm(feed_forward)
        else:
            rm = feed_forward

        return rm

    def flops(self):
        flops = 0
        flops += self.cross_attn.flops() + self.self_attn.flops()
        flops += 2 * 4 * self.d_model ** 2 # Two MLP's in the FFN.

        return flops