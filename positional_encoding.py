import math
import numpy as np
import torch

class positional_encoding():
    """
    A positional encoding class supporting the sinusoidal and the rotary positional encodings in 1D or 2D.
    """

    def __init__(self, pos, d_model, approach='sine'):
        """
        Class initialization.

        Args:
        pos (ndarray):          Position vector (1D) or matrix (2D).          
        d_model (int):          Embedding space dimension.
        appraoch (string):      Encoding approach.
        """
        super().__init()

        ndim = pos.ndim
        if ndim == 1:
            Ny = len(pos) # Ny lines.
        elif ndim == 2:
            Ny, Nx = len(pos), len(pos[0]) # Ny lines and Nx points. Each invidual point has its own position in 2D space.
        else:
            raise ValueError("1D or 2D positional encodings are supported.")
        
        if approach == 'sine':
            if ndim == 1: # Sinusoidal positional encode lines. All points within the same line share the encoding.
                self.PE = PE1D(Ny, d_model)
                self.approach = 'SINE1D'
            else: # Sinusoidal positional encode 2D space.
                self.PE = PE2D(Ny, Nx, d_model)
                self.approach = 'SINE2D'
        elif approach == 'RoPE': # TO-DO
            if ndim == 2: # Rotary positional encode lines.
                self.PE = RoPE1D(Ny, d_model)
                self.approach = 'RoPE1D'
            else: # Rotary encoding 2D space.
                self.PE = RoPE2D(Ny, Nx, d_model)
                self.approach = 'RoPE2D'

        return
    
    def forward(self, input):
        """
        Args:
        input (Tensor):         Input tensor of shape [batch_size, seq_length, d_model].
        """
        output = output + self.PE

        return output
    

def PE1D(Ny, d_model):
    """
    Sinusoidal positional encoding in 1D.

    Args:
    Ny (int):               seq_length.
    d_model (int):          d_model.

    Output:
    PE (Tensor):            Output tensor of shape [1, seq_length, d_model].
    """
    PE = torch.zeros(Ny, d_model)
    position = torch.arange(0, Ny, dtype=torch.float).unsqueeze(1)
    division_term = torch.exp(-torch.arange(0, d_model, 2, dtype=torch.float) / d_model * math.log(10000))
    PE[:, 0::2] = torch.sin(position * division_term)
    PE[:, 1::2] = torch.cos(position * division_term)

    return PE.unsqueeze(0)


def PE2D(Ny, Nx, d_model):
    """
    Sinusoidal positional encoding in 2D. This is different from a PE for a 2D grid.

    Args:
    Ny (int):               Number of lines.
    Nx (int):               Number of points per line.
    d_model (int):          d_model.

    Output:
    PE (Tensor):            Output tensor of shape [1, seq_length, d_model].
    """
    PEy = PE1D(Ny, d_model // 2) # [1, Ny, d_model / 2]
    PEx = PE1D(Nx, d_model // 2) # [1, Nx, d_model / 2]

    PEy = PEy.squeeze(0).unsqueeze(1).repeat(1, Nx, 1) # [Ny, Nx, d_model / 2]
    PEx = PEx.repeat(Ny, 1, 1) # [Ny, Nx, d_model / 2]

    PE_grid = torch.cat([PEy, PEx], dim=-1) # [Ny, Nx, d_model]
    PE_flat = PE_grid.view(Ny * Nx, PE_grid) # [Ny * Nx, d_model]

    return PE_flat.unsqueeze(0)