import torch
import torch.nn as nn

from embedding import embedding

class refine_module(nn.Module):
    """
    Refinement module class that is composed of a predict layer, inverse FFT, a ResNet, forward FFT and an embedding layer.
    """
    def __init__(self, W, H, d_model):
        super().__init__()

        self.predict = embedding(input_channel=d_model, output_channel=2, bias=False, activation=nn.ReLU, drop=0.1)
        self.embedding = embedding(input_channel=2, output_channel=d_model, bias=False, activation=nn.ReLU, drop=0.1)

        self.FFT = lambda x: torch.fft.fftshift(torch.fft.fft2(x, norm='ortho')) # Expect input tensor x to have shape [batch_size, W, H]
        self.iFFT = lambda x: torch.fft.ifft2(torch.fft.ifftshift(x), norm='ortho')
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, padding=1),
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
        img = torch.fft.ifft2(torch.fft.ifftshift(S), norm='ortho')

        # Pass through convoluntion layers.
        img += self.conv(img)

        # Apply FFT back to 2D kspace.
        ksp = torch.fft.fftshift(torch.fft.fft2(img, norm='ortho')).contiguous().view(batch_size, seq_length).unsqueeze(2) # [batch_size, seq_length, 1]
        ksp = torch.stack((torch.real(ksp), torch.imag(ksp)), dim=-1) # [batch_size, seq_length, 2]

        # Embed back to be passed into next layer.
        S_hat = S_identity + self.embedding(ksp)

        return S_hat