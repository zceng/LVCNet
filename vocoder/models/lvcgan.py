

import logging
import math

import numpy as np
import torch

from vocoder.layers import Conv1d
from vocoder.layers import Conv1d1x1
from vocoder.layers import ResidualBlock
from vocoder.layers import upsample
from vocoder import models

from .parallel_wavegan import ParallelWaveGANDiscriminator
from .lvcnet import LVCBlock 


class LVCNetWaveGAN(torch.nn.Module):
    """Parallel WaveGAN module"""

    def __init__(self, generator_params={}, discriminator_params={}):
        super().__init__() 

        self.generator = LVCNetGenerator(**generator_params) 
        self.discriminator = ParallelWaveGANDiscriminator(**discriminator_params) 

    def generator_forward(self, x, c):
        return self.generator(x, c) 
    
    def discriminator_forward(self, x):
        return self.discriminator(x) 


class LVCNetGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 inner_channels=8,
                 cond_channels=80,
                 cond_hop_length=256,
                 lvc_block_nums=3,
                 lvc_layers_each_block=10,
                 lvc_kernel_size=3,
                 kpnet_hidden_channels=64,
                 kpnet_conv_size=1,
                 dropout=0.0,
                 use_weight_norm=True,
                 ):
        """Initialize Parallel WaveGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            layers (int): Number of residual block layers.
            stacks (int): Number of stacks i.e., dilation cycles.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.

        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.lvc_block_nums = lvc_block_nums

        # define first convolution
        self.first_conv = Conv1d1x1(in_channels, inner_channels, bias=True)

        # define residual blocks
        self.lvc_blocks = torch.nn.ModuleList()
        for n in range(lvc_block_nums):
            lvcb = LVCBlock(
                in_channels=inner_channels, 
                cond_channels=cond_channels,
                conv_layers=lvc_layers_each_block,
                conv_kernel_size=lvc_kernel_size,
                cond_hop_length=cond_hop_length,
                kpnet_hidden_channels=kpnet_hidden_channels,
                kpnet_conv_size=kpnet_conv_size,
                kpnet_dropout=dropout,
            )
            self.lvc_blocks += [lvcb]

        # define output layers
        self.last_conv_layers = torch.nn.ModuleList([
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(inner_channels, inner_channels, bias=True),
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(inner_channels, out_channels, bias=True),
        ])

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, c):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            c (Tensor): Local conditioning auxiliary features (B, C ,T').

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """

        x = self.first_conv(x)
        x = self.lvc_blocks[0]( x, c )
        for n in range(1, self.lvc_block_nums):
            x = x + self.lvc_blocks[n]( x, c )

        # apply final layers
        for f in self.last_conv_layers:
            x = f(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size,
                                  dilation=lambda x: 2 ** x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(self.layers, self.stacks, self.kernel_size)

    def inference(self, c=None, x=None):
        """Perform inference.

        Args:
            c (Union[Tensor, ndarray]): Local conditioning auxiliary features (T' ,C).
            x (Union[Tensor, ndarray]): Input noise signal (T, 1).

        Returns:
            Tensor: Output tensor (T, out_channels)

        """
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(next(self.parameters()).device)
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor).to(next(self.parameters()).device)
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(x, c).squeeze(0).transpose(1, 0)
