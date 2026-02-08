# test.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys

from model import *
from model_colour import *
from utils import *


# Definiere das Gerät (CPU oder GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model(sdr_dir, model_path, output_dir, hdr_test_colour_tiff_dir, model_type, batch_size, max_nits):
    if model_type == 1:
        print(f"The model is set to luminance")
    elif model_type == 2:
        print(f"The model is set to colour")

    print(f"Testing model {model_path}")

    transform = transforms.Compose([
        transforms.Lambda(to_tensor)
    ])

    if model_type == 1:
        dataset = SDRHDRLuminanceDataset(sdr_dir, sdr_dir, max_nits, transform)
        model = UNET()
    elif model_type == 2:
        dataset = SDRHDRColourDataset(sdr_dir, sdr_dir, max_nits, transform)
        model = UNET_colour()
    else:
        print("Bitte das zu verwendende Model angeben")
        sys.exit()  # Das Programm wird hier beendet

    data_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8,  
        pin_memory=True  
    )

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    tiff_dir = os.path.join(output_dir, 'Tiff')
    os.makedirs(tiff_dir, exist_ok=True)
    image_count = 1
    with torch.no_grad():
        for idx, (sdr, hdr) in enumerate(data_loader):
            sdr = sdr.to(device)
            output = model(sdr)

            
            output = output.squeeze().cpu().numpy()
            output = np.clip(output, 0, 1) * 65535


            if model_type == 1:
                if output.ndim == 2:
                    image_number = 1
                else: image_number = output.shape[0]

            elif model_type == 2:
                if output.ndim == 3:
                    image_number = 1
                else: image_number = output.shape[0]


            for i in range(image_number):
                  # Greife auf jedes Bild im Batch zu
                if model_type == 1:
                    # Ordner festlegen

                    if output.ndim == 3:  # Check if it's 4-dimensional
                        single_output = output[i]
                        single_output = single_output.squeeze()
                    else: single_output = output

                    GT_tiff_dir = os.path.join(tiff_dir, 'GT_Farbe')
                    os.makedirs(GT_tiff_dir, exist_ok=True)
                    numpy_dir = os.path.join(output_dir, '.npy')
                    os.makedirs(numpy_dir, exist_ok=True)
                    tiff_dir = os.path.join(output_dir, 'Tiff')
                    os.makedirs(tiff_dir, exist_ok=True)
                    output_path = os.path.join(numpy_dir, f"pred_lum_{image_count:04d}.npy")

                    # Numpy speichern
                    np.save(output_path, single_output)
                    print(f"Saved Image .npy of image {image_count} as pred_lum_{image_count:04d}.npy")
                    image_count += 1

                elif model_type == 2:

                    if output.ndim == 4:  # Check if it's 4-dimensional
                        single_output = output[i]
                        single_output = single_output.squeeze()
                    else: single_output = output

                    
                    # Ordner festlegen
                    GT_tiff_dir = os.path.join(tiff_dir, 'GT')
                    os.makedirs(GT_tiff_dir, exist_ok=True)
                    pred_tiff_dir = os.path.join(tiff_dir, 'Pred')
                    os.makedirs(pred_tiff_dir, exist_ok=True)
                    output_path = os.path.join(pred_tiff_dir, f"pred_{image_count:04d}.tiff")

                    # GT Bilder kopieren
                    copy_folder_contents(hdr_test_colour_tiff_dir, GT_tiff_dir)
                    single_output_16bit = np.array(single_output, dtype = np.uint16)
                    
                    # Tiff speichern
                    tiff.imwrite(output_path, single_output_16bit)
                    print(f"Saved Image colour-tiff of {image_count} as pred_{image_count:04d}.tiff")
                    image_count += 1
                else:
                    print("Die vorhergesagten Dateien können nicht gespeichert werden.")

    # Umbenennen der GT Dateien
    count = 0
    for count, filename in enumerate(os.listdir(GT_tiff_dir), 1):
        if filename.endswith('.tiff'):
            new_name = f"gt_{count:04d}.tiff"
            os.rename(os.path.join(GT_tiff_dir, filename), os.path.join(GT_tiff_dir, new_name))
            count = count + 1

    print("Finished Testing")
  
