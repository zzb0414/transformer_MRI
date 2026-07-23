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
    def __init__(self, d_model, num_heads, W, H, dropout=0.0):
        """
        A sequential connection of a cross attention block, a self attention block, a FFN and a refinement module.

        Args:
        d_model (int):          Model dimension.
        num_heads (int):        Number of attention heads.
        W (int):                Image domain width.
        H (int):                Image domain height.
        dropout (float):        Dropout rate.
        """
        super().__init__()

        self.cross_attn = multi_head_cross_attn(d_model, num_heads)
        self.self_attn = multi_head_self_attn(d_model, num_heads)
        self.FFN = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.rm = refine_module(W, H, d_model)

        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        self.LN3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

        return
    
    def forward(self, PE_p, O, omega, LR=True):
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
            # Create mask on the same device as the input
            omega = torch.ones(batch_size, seq_length, device=cross_attention.device)
            self_attention = self.self_attn(cross_attention, omega)  # TODO: Review against technical reference - currently using all coordinates as queries for fully sampled LR output
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