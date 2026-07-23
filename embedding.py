"""
Embedding layer.
Author: Zhibo Zhu. Date: 07/15/2026.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class embedding(nn.Module):
    """
    A straight forward single linear embedding layer.
    """

    def __init__(self, input_channel=256, output_channel=768, bias=False, activation=nn.ReLU, drop=0.0):
        """
        A simple MLP.

        Args:
        input_channel (int):    Dimension of data points.
        output_channel (int):   Dimension of the embedded space.
        bias (bool):            Bias term flag.
        drop (float):           Dropout rate.
        """
        super().__init__()

        self.in_features = input_channel
        self.out_features = output_channel
        self.bias = bias
        self.drop = drop
        self.W = nn.Linear(in_features=input_channel, out_features=output_channel, bias=bias)
        self.drop = nn.Dropout(p=drop)

        if activation is not None:
            self.activation = activation()  # Instantiate the activation
        else:
            self.activation = None
        
        return

    def forward(self, input):
        """
        Args:
        input (Tensor):         Input tensor of shape [batch_size, seq_length, data_dimension].
        """
        out = self.W(input)
        if self.activation is not None:
            out = self.activation(out)
        out = self.drop(out)

        return out
    

    def flops(self):
        flops = 0
        flops += self.in_features * self.out_features

        return flops