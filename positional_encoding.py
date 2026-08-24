"""
Positional encodings.
Author: Zhibo Zhu. Date: 07/16/2026.
"""
import math
import numpy as np
import torch
import torch.nn as nn

class positional_encoding(nn.Module):
    """
    A positional encoding class supporting the sinusoidal and the rotary positional encodings in 1D or 2D.
    """

    def __init__(self, pos, d_model, approach='sine', dtype=torch.float32):
        """
        Class initialization.

        Args:
        pos (ndarray):          Position vector (1D) or matrix (2D).
        d_model (int):          Embedding space dimension.
        appraoch (string):      Encoding approach.
        dtype (torch.dtype):    Data type of the PE buffer (default: torch.float32).
        """
        super().__init__()

        ndim = pos.ndim
        if ndim == 1:
            Ny = len(pos) # Ny lines.
            Nx = None
        elif ndim == 2:
            Ny, Nx = pos.shape # Ny lines, Nx points per line
        elif ndim == 3:
            # For 3D: pos shape is [2, H, W] from np.mgrid
            # We have H rows (y) and W columns (x)
            Ny, Nx = pos.shape[1], pos.shape[2]
        else:
            raise ValueError("1D, 2D, or 3D (from np.mgrid) positional encodings are supported.")

        if approach == 'sine':
            if ndim == 1: # Sinusoidal positional encode lines. All points within the same line share the encoding.
                PE = PE1D(Ny, d_model, dtype)
                self.approach = 'SINE1D'
            else: # Sinusoidal positional encode 2D space.
                PE = PE2D(Ny, Nx, d_model, dtype)
                self.approach = 'SINE2D'
        elif approach == 'RoPE': # TO-DO
            if ndim == 2: # Rotary positional encode lines.
                PE = RoPE1D(Ny, d_model)
                self.approach = 'RoPE1D'
            else: # Rotary encoding 2D space.
                PE = RoPE2D(Ny, Nx, d_model)
                self.approach = 'RoPE2D'

        # Register as buffer so it moves with .to(device)
        self.register_buffer('PE', PE)

        return

    def forward(self, input):
        """
        Args:
        input (Tensor):         Input tensor of shape [batch_size, seq_length, d_model].
        """
        output = input + self.PE

        return output
    

def PE1D(Ny, d_model, dtype=torch.float32):
    """
    Sinusoidal positional encoding in 1D.

    Args:
    Ny (int):               seq_length.
    d_model (int):          d_model.
    dtype (torch.dtype):    Data type of the output tensor (default: torch.float32).

    Output:
    PE (Tensor):            Output tensor of shape [1, seq_length, d_model].
    """
    PE = torch.zeros(Ny, d_model, dtype=dtype)
    position = torch.arange(-Ny // 2, Ny // 2, dtype=dtype).unsqueeze(1) # Shift by half.
    division_term = torch.exp(-torch.arange(0, d_model, 2, dtype=dtype) / d_model * math.log(10000))
    PE[:, 0::2] = torch.sin(position * division_term)
    PE[:, 1::2] = torch.cos(position * division_term)

    return PE.unsqueeze(0)


def PE2D(Ny, Nx, d_model, dtype=torch.float32):
    """
    Sinusoidal positional encoding in 2D. This is different from a PE for a 2D grid.

    Args:
    Ny (int):               Number of lines.
    Nx (int):               Number of points per line.
    d_model (int):          d_model.
    dtype (torch.dtype):    Data type of the output tensor (default: torch.float32).

    Output:
    PE (Tensor):            Output tensor of shape [1, seq_length, d_model].
    """
    PEy = PE1D(Ny, d_model // 2, dtype) # [1, Ny, d_model / 2]
    PEx = PE1D(Nx, d_model // 2, dtype) # [1, Nx, d_model / 2]

    PEy = PEy.squeeze(0).unsqueeze(1).repeat(1, Nx, 1) # [Ny, Nx, d_model / 2]
    PEx = PEx.repeat(Ny, 1, 1) # [Ny, Nx, d_model / 2]

    PE_grid = torch.cat([PEy, PEx], dim=-1) # [Ny, Nx, d_model]
    PE_flat = PE_grid.view(Ny * Nx, d_model) # [Ny * Nx, d_model]

    return PE_flat.unsqueeze(0)