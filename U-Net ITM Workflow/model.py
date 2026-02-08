import glob
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SDRHDRLuminanceDataset(Dataset):
    def __init__(self, sdr_luminance_dir, hdr_luminance_dir, max_nits, transform=None):
        self.sdr_luminance_paths = sorted(glob.glob(os.path.join(sdr_luminance_dir, '*.npy')))
        self.hdr_luminance_paths = sorted(glob.glob(os.path.join(hdr_luminance_dir, '*.npy')))
        self.max_nits = max_nits
        self.transform = transform

    def __len__(self):
        return len(self.sdr_luminance_paths)

    def __getitem__(self, idx):
        sdr_luminance = np.load(self.sdr_luminance_paths[idx]).astype(np.float32)
        hdr_luminance = np.load(self.hdr_luminance_paths[idx]).astype(np.float32)

        # Konvertieren in Tensoren
        sdr_luminance_tensor = torch.from_numpy(sdr_luminance).unsqueeze(0)
        hdr_luminance_tensor = torch.from_numpy(hdr_luminance).unsqueeze(0)

        if self.transform:
            sdr_luminance_tensor = self.transform(sdr_luminance_tensor)
            hdr_luminance_tensor = self.transform(hdr_luminance_tensor)

        return sdr_luminance_tensor, hdr_luminance_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True, padding_mode="replicate"), # kernelsize 3, stride 1, padding 1 (spiegelt Randwerte)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, padding_mode="replicate"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        final_output = self.final_conv(x)
        return final_output

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)



# #Pixel Shuffel
# import glob
# import numpy as np
# import os

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, Dataset


# class SDRHDRLuminanceDataset(Dataset):
#     def __init__(self, sdr_luminance_dir, hdr_luminance_dir, max_nits, transform=None):
#         self.sdr_luminance_paths = sorted(glob.glob(os.path.join(sdr_luminance_dir, '*.npy')))
#         self.hdr_luminance_paths = sorted(glob.glob(os.path.join(hdr_luminance_dir, '*.npy')))
#         self.max_nits = max_nits
#         self.transform = transform

#     def __len__(self):
#         return len(self.sdr_luminance_paths)

#     def __getitem__(self, idx):
#         sdr_luminance = np.load(self.sdr_luminance_paths[idx]).astype(np.float32)
#         hdr_luminance = np.load(self.hdr_luminance_paths[idx]).astype(np.float32)

#         # Konvertieren in Tensoren
#         sdr_luminance_tensor = torch.from_numpy(sdr_luminance).unsqueeze(0)
#         hdr_luminance_tensor = torch.from_numpy(hdr_luminance).unsqueeze(0)

#         if self.transform:
#             sdr_luminance_tensor = self.transform(sdr_luminance_tensor)
#             hdr_luminance_tensor = self.transform(hdr_luminance_tensor)

#         return sdr_luminance_tensor, hdr_luminance_tensor


# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True, padding_mode="replicate"),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True, padding_mode="replicate"),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.conv(x)


# class UNET(nn.Module):
#     def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024], upscale_factor=2):
#         super(UNET, self).__init__()
#         self.downs = nn.ModuleList()
#         self.ups = nn.ModuleList()
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.upscale_factor = upscale_factor

#         # Downsampling
#         for feature in features:
#             self.downs.append(DoubleConv(in_channels, feature))
#             in_channels = feature

#         self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

#         # Upsampling mit Pixel Shuffle
#         for feature in reversed(features):
#             self.ups.append(
#                 nn.Conv2d(
#                     in_channels=feature * 2,
#                     out_channels=feature * (upscale_factor ** 2),
#                     kernel_size=3,
#                     padding=1
#                 )
#             )
#             self.ups.append(DoubleConv(feature * 2, feature))

#         self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

#     def forward(self, x):
#         skip_connections = []

#         # Downsampling
#         for down in self.downs:
#             x = down(x)
#             skip_connections.append(x)
#             x = self.pool(x)

#         # Bottleneck
#         x = self.bottleneck(x)
#         skip_connections = skip_connections[::-1]

#         # Upsampling mit Pixel Shuffle
#         for idx in range(0, len(self.ups), 2):
#             x = self.ups[idx](x)  # 1. Conv2d
#             x = nn.PixelShuffle(self.upscale_factor)(x)  # Pixel Shuffle

#             skip_connection = skip_connections[idx // 2]

#             # Ensure matching dimensions
#             if x.shape != skip_connection.shape:
#                 x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

#             concat_skip = torch.cat((skip_connection, x), dim=1)
#             x = self.ups[idx + 1](concat_skip)  # 2. DoubleConv

#         final_output = self.final_conv(x)
#         return final_output


# def to_tensor(x):
#     if isinstance(x, torch.Tensor):
#         return x.clone().detach().to(dtype=torch.float32)
#     return torch.as_tensor(x, dtype=torch.float32)