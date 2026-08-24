"""
Image domain refinement module.
Author: Zhibo Zhu. Date: 07/16/2026.
"""
import torch
import torch.nn as nn


class refine_module(nn.Module):
    """
    Refinement module class that is composed of a predict layer, inverse FFT, a ResNet, forward FFT and an embedding layer.
    """
    def __init__(self, W, H, d_model, dtype=torch.float32):
        """
        Class initlization.

        Args:
        W (int):                Full k-space width.
        H (int):                Full k-space height.
        d_model (int):          Embedding space dimension.
        dtype (torch.dtype):    Parameter/compute dtype, e.g. torch.float32 (default), torch.bfloat16, torch.float16.
        """
        super().__init__()

        # Linear projections without activation/dropout for space mapping
        self.predict = nn.Linear(d_model, 2, bias=False, dtype=dtype)
        self.embedding = nn.Linear(2, d_model, bias=False, dtype=dtype)

        self.FFT = lambda x: torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(x, dim=(1, 2)), norm='ortho'), dim=(1, 2)) # Expect input tensor x to have shape [batch_size, W, H]
        self.iFFT = lambda x: torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(1, 2)), norm='ortho'), dim=(1, 2))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1, dtype=dtype),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, dtype=dtype),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, dtype=dtype),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, dtype=dtype),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, padding=1, dtype=dtype),
        )

        self.W = W
        self.H = H
        self.d_model = d_model
        return
    
    def forward(self, S):
        """
        Args:

        S (tensor):             Output tensor from the previous self attention block, [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = S.size()
        S_identity = S

        # Reshape S into 2D kspaces, [batch_size, W, H].
        S = self.predict(S) # [batch_size, seq_length, 2]
        S = S.contiguous().view(batch_size, self.W, self.H, 2)
        S = S[..., 0] + 1j * S[..., 1]

        # Apply iFFT.
        # img = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(S, dim=(1, 2)), norm='ortho'), dim=(1, 2))  # [batch_size, W, H], complex
        img = self.iFFT(S)

        # Convert complex to 2-channel real tensor [batch_size, 2, W, H]
        img = torch.stack((torch.real(img), torch.imag(img)), dim=1)

        # Pass through convolution layers with residual.
        img = img + self.conv(img)

        # Convert 2-channel real back to complex [batch_size, W, H]
        img_complex = torch.complex(img[:, 0, :, :], img[:, 1, :, :])

        # Apply FFT back to 2D kspace.
        # ksp = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(img_complex, dim=(1, 2)), norm='ortho'), dim=(1, 2))  # [batch_size, W, H], complex
        ksp = self.FFT(img_complex)
        ksp = torch.stack((torch.real(ksp), torch.imag(ksp)), dim=-1)  # [batch_size, W, H, 2]
        ksp = ksp.contiguous().view(batch_size, seq_length, 2)  # [batch_size, seq_length, 2]

        # Embed back to be passed into next layer.
        S_hat = S_identity + self.embedding(ksp) # Need to think whether residual connection is proper.

        return S_hat