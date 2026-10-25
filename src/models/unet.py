"""
2D U-Net Architecture Implementation for Multimodal Medical Image Segmentation.

Module Overview:
----------------
1. What it does:
   - Implements a standard 2D Encoder-Decoder U-Net with skip connections in PyTorch.
   - Encoder: Extracts multi-scale feature representations using contracting double-convolution blocks.
   - Skip Connections: Preserves high-resolution fine spatial details by concatenating encoder features with decoder features.
   - Decoder: Expands spatial resolution using transpose convolutions and constructs final segmentation predictions.

2. Why it is required:
   - U-Net is the gold-standard baseline architecture for medical image segmentation tasks.
   - Fully convolutional design allows end-to-end pixel-level tumor classification.

3. Inputs:
   - PyTorch Tensor of shape (Batch_Size, in_channels, Height, Width). Default in_channels = 4.
4. Outputs:
   - PyTorch Logits Tensor of shape (Batch_Size, out_channels, Height, Width). Default out_channels = 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm2D -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class UNet2D(nn.Module):
    """
    Standard 2D U-Net Architecture.
    """
    def __init__(self, in_channels=4, out_channels=1, base_filters=32, depth=4, dropout=0.1):
        """
        Parameters:
            in_channels (int): Number of input MRI modalities (default: 4).
            out_channels (int): Number of output segmentation classes (default: 1 for binary tumor mask).
            base_filters (int): Number of feature channels in first encoder block (default: 32).
            depth (int): Depth levels of encoder/decoder (default: 4).
            dropout (float): Dropout probability in bottleneck layer (default: 0.1).
        """
        super(UNet2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Encoder Contracting Path
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        curr_in = in_channels
        curr_out = base_filters
        
        for d in range(depth):
            self.encoder_blocks.append(DoubleConv(curr_in, curr_out))
            self.pool_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            curr_in = curr_out
            curr_out *= 2
            
        # Bottleneck Layer
        self.bottleneck = DoubleConv(curr_in, curr_out, dropout=dropout)
        
        # Decoder Expanding Path
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for d in range(depth):
            self.up_convs.append(nn.ConvTranspose2d(curr_out, curr_in, kernel_size=2, stride=2))
            self.decoder_blocks.append(DoubleConv(curr_out, curr_in))
            curr_out = curr_in
            curr_in = curr_in // 2

        # Final 1x1 Classification Layer
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of 2D U-Net.

        Input:
            x (Tensor): Shape (B, 4, H, W)
        Output:
            Tensor: Logits of shape (B, 1, H, W)
        """
        skip_connections = []

        # Encoder Path
        for block, pool in zip(self.encoder_blocks, self.pool_layers):
            x = block(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder Path with Skip Connections
        skip_connections = skip_connections[::-1] # Reverse skips
        for idx in range(self.depth):
            x = self.up_convs[idx](x)
            skip = skip_connections[idx]
            
            # Handle padding if spatial dimensions are odd
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
                
            x = torch.cat([skip, x], dim=1) # Concatenate along channel dimension
            x = self.decoder_blocks[idx](x)

        # Final Logits
        logits = self.final_conv(x)
        return logits
