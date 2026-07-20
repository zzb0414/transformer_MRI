"""
Transformer model.
Author: Zhibo Zhu. Date: 07/20/2026.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from attention_block import multi_head_self_attn, multi_head_cross_attn
from refine_module import refine_module
from embedding import embedding
from positional_encoding import positional_encoding
from encoder_block import encoder_block
from decoder_block import decoder_block

class transformer(nn.Module):
    def __init__(self, input_channel, output_channel, bias, pos, d_model, N, num_heads, W, H, activition=nn.ReLU, approach='sine', dropout_emb=0.0, dropout_enc=0.0, dropout_dec=0.0):
        """
        Class initialization.

        Args:
        input_channel (int):    Dimension of data points.
        output_channel (int):   Dimension of the embedded space.
        bias (bool):            Bias term flag.
        pos (ndarray):          Position vector (1D) or matrix (2D). If 1D: [0, H - 1]. If 2D: [0, W - 1] x [0, H - 1].
        d_model (int):          Embedding space dimension.
        N (int):                Encoder/Decoder layers.
        num_head )int):         Number of attention heads.
        W (int):                Full k-space width.
        H (int):                Full k-space height.
        activition (nn.Module): Activition function.
        appraoch (string):      Encoding approach.
        dropout_emb (float):    Embedding layer dropout rate.
        dropout_enc (float):    Encoding layer dropout rate.
        dropout_dec (float):    Decoding layer dropout rate.
        """
        super().__init__()

        self.embedding = embedding(input_channel, output_channel, bias, activition, dropout_emb)
        self.PE = positional_encoding(pos, d_model, approach)
        self.PE_LR = positional_encoding(pos[H // 4:-H // 4][W // 4:-W // 4], d_model, approach)

        self.encoders = nn.ModuleList([encoder_block(d_model, num_heads, dropout_enc) for _ in range(N)])
        self.LR_decoders = nn.ModuleList([decoder_block(d_model, num_heads, W // 2, H // 2, dropout_dec) for _ in range(N)])
        self.HR_decoders = nn.ModuleList([decoder_block(d_model, num_heads, W, H, dropout_dec) for _ in range(N)])

        return

    def forward(self, ksp, omega):
        """
        Transformer foward operation.

        Args:
        ksp (tensor):           Input k-space tensor, [batch_size, seq_length, 2]
        omega (tensor):         Sampling mask, 1: sampled, 0: not sampled. [batch_size, seq_length]
        """
        # Linear embedding and PE ksp.
        input = self.embedding(ksp)
        input = self.PE(input)

        # Pass into encoders.
        output_enc = input
        for ii in range(len(self.encoders)):
            output_enc = self.encoders[ii](output_enc, omega)

        # Gather PE_p and output_enc for LR decoders.
        PE_p = self.PE_LR.PE
        output_LR_dec = output_enc
        for ii in range(len(self.LR_decoders)):
            output_LR_dec = self.LR_decoders(PE_p, output_LR_dec, None, LR=True)

        # Gather PE_p and output_LR_dec for HR decoders.
        PE_p = self.PE.PE # Need to re-investigate. This part is very complicated.
        output_HR_dec = output_LR_dec
        for ii in range(len(self.HR_decoders)):
            output_HR_dec = self.HR_decoders(PE_p, output_HR_dec, LR=False)

        return output_HR_dec