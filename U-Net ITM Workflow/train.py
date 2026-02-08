# train.py
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

from model import *
from model_colour import *
from loss_function import choose_loss_function

# Definiere das Gerät (CPU oder GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if torch.cuda.is_available():
#     print("Cuda tut :)")
# else:
#     print("Cuda tut nicht :(")

def train_model(sdr_dir, hdr_dir, model_save_path, model, batch_size, num_epochs, patience, lr, loss_function, max_nits): 
    # print(f"sdr_dir ist: {sdr_dir}")
    # print(f"hdr_dir ist: {hdr_dir}")
    transform = transforms.Compose([
        transforms.Lambda(to_tensor)
    ])

    if model == 1:
        dataset = SDRHDRLuminanceDataset(sdr_dir, hdr_dir, max_nits)
        model = UNET()
    elif model == 2:
        dataset = SDRHDRColourDataset(sdr_dir, hdr_dir, max_nits)
        model = UNET_colour()
    else:
        print("Bitte das zu verwendende Model angeben")
        sys.exit()  # Das Programm wird hier beendet

    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=8,  
        pin_memory=True  
    )

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    loss_array = np.zeros(num_epochs)
    lr_array = np.zeros(num_epochs)  # Array zum Speichern der Lernrate


    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        batch_num = 0
        for sdr, hdr in train_loader:
            sdr, hdr = sdr.to(device), hdr.to(device)

            batch_num = batch_num + 1

            optimizer.zero_grad()
            outputs = model(sdr)

            # Debugging: Vor der Interpolation
            # print("Outputs before interpolation - Min:", outputs.min().item(), "Max:", outputs.max().item())
            # print("Outputs before interpolation- Shape:", outputs.shape)

            outputs = torch.nn.functional.interpolate(outputs, size=hdr.shape[2:], mode='bilinear', align_corners=False)

            # Debugging: Nach der Interpolation
            print(f"Outputs of batch {batch_num}: - Min:", outputs.min().item(), "Max:", outputs.max().item())
            # print("Outputs after interpolation- Shape:", outputs.shape)

            loss = choose_loss_function(outputs, hdr, loss_function)
            print(f'Loss value for batch {batch_num} in epoch {epoch+1}: {loss}')  # Add this line
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * sdr.size(0)

            torch.cuda.empty_cache()

        print("")
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}')
        loss_array[epoch] = epoch_loss
        current_lr = optimizer.param_groups[0]['lr']  # Lernrate aus dem Optimizer holen
        lr_array[epoch] = current_lr  # Lernrate im Array speichern

                
        if epoch > 0:
            print(f"Changes in Loss from epoch {epoch-1} to epoch {epoch}: ", loss_array[epoch-1] - loss_array[epoch])
        print(f"Current loss progression: ", loss_array)
        print("")
        print("")

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_save_path)  # Save the best model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        model_save_dir = os.path.dirname(model_save_path)
        np.save(f'{model_save_dir}/loss_array.npy', loss_array)  # Loss-Werte in Datei speichern
        np.save(f'{model_save_dir}/lr_array.npy', lr_array)      # Lernraten in Datei speichern

        # torch.save(model.state_dict(), model_save_path)  # Save the best model
        torch.cuda.empty_cache()

    # Informationen für das predicted Array
    # print("predicted array info:")
    # print(f"Dimensions: {outputs.shape}")
    # print(f"Min value: {outputs.min()}")
    # print(f"Max value: {outputs.max()}")
    # print(f"Mean value: {outputs.mean()}")
    # print(f"Standard Deviation: {outputs.std()}")
    # print(f"Data Type: {outputs.dtype}")
    
    #plt.plot(loss_array)
    # plt.savefig('epoch_loss_plot.png', dpi=300, bbox_inches='tight')
    # plt.show()

