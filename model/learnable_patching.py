import torch.nn as nn
from einops import rearrange


class Learnable1dPatching(nn.Module):
    def  __init__(
        self, 
        in_channels: int,
        out_channels: int,
        temporal_patch_size: int = 4,
        activation: nn.Module = nn.GELU()
    ):
        """Learnable patching for 1d data (only temporal).

        :param in_channels: How many channels for the 1d convolution.
        :type in_channels: int
        :param batch_size: Batch size (Needed vor shape inference).
        :type batch_size: int
        :param out_channels: Model dimension size (later for the transformer input).
        :type out_channels: int
        :param temporal_patch_size: Size of temporal patch, defaults to 4
        :type temporal_patch_size: int, optional
        :param activation: Activation for learnable patching, defaults to GELU
        :type activation: nn.Module
        """
        super().__init__()

        self.P = [0,2,1]
        self.out_channels = out_channels
        self.patch = nn.Conv1d(in_channels,
                               out_channels,
                               temporal_patch_size,
                               temporal_patch_size,
                               padding='valid')
        self.proj = nn.Linear(out_channels, out_channels)
        self.act = activation
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch(x.permute(self.P))
        #x = x.view(batch_size, -1, self.out_channels)
        x = rearrange(x, 'b t c -> b c t')
        return self.act(self.proj(x))


class Learnable3dPatching(nn.Module):

    def __init__(
        self,
        out_channels: int,
        spatial_patch_size: int = 16,
        temporal_patch_size: int = 4,
        activation: nn.Module = nn.GELU()
    ):
        """Implements a patching processes using Convolutions and Linear Projections
        with learnable weights for time series data. Prepares the input for a transformer.

        :param batch_size: Batch size (Needed vor shape inference).
        :type batch_size: int
        :param out_channels: Model dimension size (later for the transformer input).
        :type out_channels: int
        :param spatial_patch_size: Size of spatial patch, defaults to 16
        :type spatial_patch_size: int, optional
        :param temporal_patch_size: Size of temporal patch, defaults to 4
        :type temporal_patch_size: int, optional
        :param activation: Activation for learnable patching, defaults to GELU
        :type activation: nn.Module
        """
        super().__init__()

        self.P = [0,2,1,3,4]
        self.out_channels = out_channels
        self.patch = nn.Conv3d(3, 
                               out_channels,
                               (temporal_patch_size, spatial_patch_size, spatial_patch_size),
                               (temporal_patch_size, spatial_patch_size, spatial_patch_size),
                               padding='valid')
        self.proj = nn.Linear(out_channels, out_channels)
        self.act = activation 


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch(x.permute(self.P))
        #breakpoint()
        #x = x.view(batch_size, -1, self.out_channels)
        x = rearrange(x, 'b c t w h -> b t w h c')
        return self.act(self.proj(x))
