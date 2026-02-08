import glob
import numpy as np
import os
import tifffile as tiff

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset



class SDRHDRColourDataset(Dataset):
    def __init__(self, sdr_colour_dir, hdr_colour_dir, max_nits, transform=None):
        self.sdr_colour_paths = sorted(glob.glob(os.path.join(sdr_colour_dir, '*.tiff')))
        self.hdr_colour_paths = sorted(glob.glob(os.path.join(hdr_colour_dir, '*.tiff')))
        self.max_nits = max_nits
        self.transform = transform

    def __len__(self):
        return len(self.sdr_colour_paths)

    def __getitem__(self, idx):
        sdr_colour = tiff.imread(self.sdr_colour_paths[idx])
        sdr_colour = np.array(sdr_colour, dtype=np.float32)/255
        hdr_colour = tiff.imread(self.hdr_colour_paths[idx])
        hdr_colour = np.array(hdr_colour, dtype=np.float32)/65535

        # # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
        # print("Original SDR colour - Min:", sdr_colour.min(), "Max:", sdr_colour.max())
        # print("Original HDR colour - Min:", hdr_colour.min(), "Max:", hdr_colour.max())

        # Entfernen überflüssiger Dimensionen, falls vorhanden
        sdr_colour = np.squeeze(sdr_colour)
        hdr_colour = np.squeeze(hdr_colour)

        # Anpassen der Achsenreihenfolge von (x, y, z) auf (z, x, y)
        sdr_colour = np.transpose(sdr_colour, (2, 0, 1))  # (z, x, y)
        hdr_colour = np.transpose(hdr_colour, (2, 0, 1))  # (z, x, y)

        # print(sdr_colour.shape)  # Sollte [816, 1920, 3] sein
        # print(hdr_colour.shape)  # Sollte ebenfalls [816, 1920, 3] sein


        # Normalisieren
        # sdr_colour = sdr_colour / self.max_nits
        # hdr_colour = hdr_colour / self.max_nits    # Adjust to HDR movie max nits

        # # Debugging: Ausgabe der Min- und Max-Werte nach der Normalisierung
        # print("Normalized SDR colour - Min:", sdr_colour.min(), "Max:", sdr_colour.max())
        # print("Normalized HDR colour - Min:", hdr_colour.min(), "Max:", hdr_colour.max())


        # Konvertieren in Tensoren
        sdr_colour_tensor = torch.from_numpy(sdr_colour)
        hdr_colour_tensor = torch.from_numpy(hdr_colour)    #.unsqueeze(0)

        if self.transform:
            sdr_colour_tensor = self.transform(sdr_colour_tensor)
            hdr_colour_tensor = self.transform(hdr_colour_tensor)

        return sdr_colour_tensor, hdr_colour_tensor

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True , padding_mode="replicate"), # kernelsize 3, stride 1, padding 1 (spiegelt Randwerte)
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True , padding_mode="replicate"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET_colour(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512, 1024]):
        super(UNET_colour, self).__init__()
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
            # # Debugging: Ausgabe der Min- und Max-Werte nach jedem Down-Sampling-Schritt
            # print("After down-sampling layer - Min:", x.min().item(), "Max:", x.max().item())
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # # Debugging: Ausgabe der Min- und Max-Werte nach dem Bottleneck
        # print("After bottleneck - Min:", x.min().item(), "Max:", x.max().item())
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # # Debugging: Ausgabe der Min- und Max-Werte nach der ersten Upsample-Schicht
            # print(f"After upsample {idx//2} - Min:", x.min().item(), "Max:", x.max().item())
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
            # # Debugging: Ausgabe der Min- und Max-Werte nach der Concatenation
            # print(f"After concat {idx//2} - Min:", x.min().item(), "Max:", x.max().item())
            # print(f"After concat {idx//2} - Shape:", x.shape)

        final_output = self.final_conv(x)

        # final_output = torch.clamp(self.final_conv(x), min=0, max=4000)

        # # Debugging: Ausgabe der Min- und Max-Werte nach der finalen Ausgabe
        # print("After final conv - Min:", final_output.min().item(), "Max:", final_output.max().item())
        # print("After final conv - Shape:", final_output.shape)


        return final_output

def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)
