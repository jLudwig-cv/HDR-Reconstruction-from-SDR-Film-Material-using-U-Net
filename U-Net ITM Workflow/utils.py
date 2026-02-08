import os
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import shutil
import tifffile as tiff
import numpy as np
from PIL import Image
from save_images import *


def create_folder_test(base_dir, model_path, model):

    model_dir = os.path.dirname(model_path)  # Hol den Ordnerpfad
    model_folder_name = os.path.basename(model_dir)  # Hol den Ordnernamen
    timestamp = model_folder_name.split('model_')[1]  # Nimm den Zeitstempel aus dem Ordnernamen

    # Erstelle den neuen Ordner mit dem gleichen Zeitstempel
    if model == 1:
        output_dir = os.path.join(base_dir, f"predicted_HDR_luminance_{timestamp}")
    elif model == 2:
        output_dir = os.path.join(base_dir, f"predicted_HDR_colour_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def create_folder_model(base_dir, model):
    os.makedirs(base_dir, exist_ok=True)
    current_date = datetime.now().strftime('%Y%m%d_%H%M')
    model_save_dir = os.path.join(base_dir, f"model_{current_date}")
    os.makedirs(model_save_dir, exist_ok=True)
    if model == 1:
        model_save_path = os.path.join(model_save_dir, 'sdr_to_hdr_in_luminance.pth')
    elif model == 2:
        model_save_path = os.path.join(model_save_dir, 'sdr_to_hdr_in_colour.pth')
    
    return model_save_path

def get_latest_file(dir, model):
   
    # Suche nach allen Dateien im Verzeichnis, die dem Muster entsprechen
    if model == 1:
        files = glob.glob(os.path.join(dir, "model_*", "sdr_to_hdr_in_luminance.pth"))
    elif model == 2:
        files = glob.glob(os.path.join(dir, "model_*", "sdr_to_hdr_in_colour.pth"))

    # Überprüfen, ob Dateien gefunden wurden
    if not files:
        raise FileNotFoundError("No model files found.")

    # Sortiere die Dateien nach dem Datum und der Uhrzeit, die im Dateipfad enthalten sind
    files.sort(key=os.path.getmtime, reverse=True)

    # Die aktuellste Datei ist nun die erste in der Liste
    latest_path = files[0]

    # Beispiel für die Ausgabe des Pfads der aktuellsten Datei
    print(f"The latest model path is: {latest_path}")

    return latest_path

def get_latest_folder(dir, model):
   
    # Suche nach allen Dateien im Verzeichnis, die dem Muster entsprechen
    if model == 1:
        files = glob.glob(os.path.join(dir, "predicted_HDR_luminance_*"))
    elif model ==2:
        files = glob.glob(os.path.join(dir, "predicted_HDR_colour_*"))

    # Überprüfen, ob Dateien gefunden wurden
    if not files:
        raise FileNotFoundError("No model folder found.")

    # Sortiere die Dateien nach dem Datum und der Uhrzeit, die im Dateipfad enthalten sind
    files.sort(key=os.path.getmtime, reverse=True)

    # Die aktuellste Datei ist nun die erste in der Liste
    latest_path = files[0]

    # Beispiel für die Ausgabe des Pfads der aktuellsten Datei
    print(f"The latest test path is: {latest_path}")

    return latest_path



def save_args_to_txt(args, output_dir):
    file_path = os.path.join(output_dir, "arguments.txt")  # Ergänze den Dateinamen zum Pfad
    with open(file_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
    print(f"Arguments saved to {file_path}")


# Funktion, um den Ordner zu leeren
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        # Löscht alle Dateien und Unterordner im Ordner
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Datei oder Symbolischer Link löschen
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Ordner und Inhalt löschen
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(folder_path)  # Ordner erstellen, falls er nicht existiert


def copy_folder_contents(source_folder, destination_folder):
    # Sicherstellen, dass das Zielverzeichnis existiert
    os.makedirs(destination_folder, exist_ok=True)

    # Iteriere durch alle Dateien und Unterordner im Quellordner
    for item in os.listdir(source_folder):
        source_path = os.path.join(source_folder, item)
        destination_path = os.path.join(destination_folder, item)

        # Wenn es sich um eine Datei handelt, kopiere sie
        if os.path.isfile(source_path):
            shutil.copy2(source_path, destination_path)
        # Wenn es sich um ein Verzeichnis handelt, kopiere den gesamten Ordner
        elif os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)


def sdrImg_to_dispLightNpy(sdr_file):
    sdr_tiff = Image.open(sdr_file)
    sdr_arr = np.array(sdr_tiff, dtype=np.float32) / 255
    # print(f"After normalization (SDR) - Min: {sdr_arr.min()}, Max: {sdr_arr.max()}")

    SDR_2020_arr = rec709_to_rec2020(sdr_arr)
    ICtCp_SDR_img = rgb2020_to_ICtCp(SDR_2020_arr)

    I_SDR = ICtCp_SDR_img[..., 0]
    # print("After sdr image to displaylight - Min:", I_SDR.min(), "Max:", I_SDR.max())

    return I_SDR


def hdrTiff_to_dispLightNpy(hdr_file):
    hdr_tiff = tiff.imread(hdr_file)
    hdr_arr = np.array(hdr_tiff, dtype=np.float32) / 65535
    # print(f"Before normalization (HDR) - Min: {hdr_arr.min()}, Max: {hdr_arr.max()}")

    ICtCp_hdr_img = rgb2020_to_ICtCp(hdr_arr)

    I_hdr = ICtCp_hdr_img[..., 0]
    # print("After hdr image to displaylight - Min:", I_hdr.min(), "Max:", I_hdr.max())
    
    return I_hdr


def convert_images_to_luminance_train(data_dir):
    train_dir = os.path.join(data_dir, "Train_dataset")

    sdr_train_colour_dir = os.path.join(train_dir, "SDR_train_colour_tiff")
    hdr_train_colour_dir = os.path.join(train_dir, "HDR_train_colour_tiff")

    sdr_train_luminance_dir = os.path.join(train_dir, "SDR_train_luminance")
    hdr_train_luminance_dir = os.path.join(train_dir, "HDR_train_luminance")

    # Ordner machen
    os.makedirs(sdr_train_luminance_dir, exist_ok=True)
    os.makedirs(hdr_train_luminance_dir, exist_ok=True)

    # Ordner leeren
    clear_folder(sdr_train_luminance_dir)
    clear_folder(hdr_train_luminance_dir)

    # SDR-Train-TIFFs in Luminanz umwandeln und speichern
    sdr_train_paths = glob.glob(os.path.join(sdr_train_colour_dir, '*.tiff'))
    for sdr_train_path in sdr_train_paths:
        sdr_train_disp_light = sdrImg_to_dispLightNpy(sdr_train_path)
        sdr_train_luminance_path = os.path.join(sdr_train_luminance_dir, os.path.basename(sdr_train_path).replace('.tiff', '_luminance.npy'))
        np.save(sdr_train_luminance_path, sdr_train_disp_light)
        print(f"SDR Train Luminance for {os.path.basename(sdr_train_luminance_path)} saved")

    # HDR-Train-TIFFs in Luminanz umwandeln und speichern
    hdr_train_paths = glob.glob(os.path.join(hdr_train_colour_dir, '*.tiff'))
    for hdr_train_path in hdr_train_paths:
        hdr_train_disp_light = hdrTiff_to_dispLightNpy(hdr_train_path)
        hdr_train_luminance_path = os.path.join(hdr_train_luminance_dir, os.path.basename(hdr_train_path).replace('.tiff', '_luminance.npy'))
        np.save(hdr_train_luminance_path, hdr_train_disp_light)
        print(f"HDR Train Luminance for {os.path.basename(hdr_train_luminance_path)} saved")

def convert_images_to_luminance_test(data_dir):
    test_dir = os.path.join(data_dir, "Test_dataset")

    sdr_test_colour_dir = os.path.join(test_dir, "SDR_test_colour_tiff")
    hdr_test_colour_dir = os.path.join(test_dir, "HDR_test_colour_tiff")

    sdr_test_luminance_dir = os.path.join(test_dir, "SDR_test_luminance")
    hdr_test_luminance_dir = os.path.join(test_dir, "HDR_test_luminance")

    # Ordner machen
    os.makedirs(sdr_test_luminance_dir, exist_ok=True)
    os.makedirs(hdr_test_luminance_dir, exist_ok=True)

    # Ordner leeren
    clear_folder(sdr_test_luminance_dir)
    clear_folder(hdr_test_luminance_dir)

    # SDR-Test-TIFFs in Luminanz umwandeln und speichern
    sdr_test_paths = glob.glob(os.path.join(sdr_test_colour_dir, '*.tiff'))
    for sdr_test_path in sdr_test_paths:
        sdr_test_disp_light = sdrImg_to_dispLightNpy(sdr_test_path)
        sdr_test_luminance_path = os.path.join(sdr_test_luminance_dir, os.path.basename(sdr_test_path).replace('.tiff', '_luminance.npy'))
        np.save(sdr_test_luminance_path, sdr_test_disp_light)
        print(f"SDR Test Luminance for {os.path.basename(sdr_test_luminance_path)} saved")

    # HDR-Test-TIFFs in Luminanz umwandeln und speichern
    hdr_test_paths = glob.glob(os.path.join(hdr_test_colour_dir, '*.tiff'))
    for hdr_test_path in hdr_test_paths:
        hdr_test_disp_light = hdrTiff_to_dispLightNpy(hdr_test_path)
        hdr_test_luminance_path = os.path.join(hdr_test_luminance_dir, os.path.basename(hdr_test_path).replace('.tiff', '_luminance.npy'))
        np.save(hdr_test_luminance_path, hdr_test_disp_light)
        print(f"HDR Test Luminance for {os.path.basename(hdr_test_luminance_path)} saved")



def get_dataset_paths(data_dir, dataset_option):
    """Returns the appropriate paths for the selected dataset."""
    dataset_names = {
        1: 'digital',
        2: 'analog',
        3: 'komplett',
        4: 'hable_digital',
        5: 'reinhard_digital',
        6: 'moebius_digital'
    }

    if dataset_option not in dataset_names:
        raise ValueError(f"Dataset option {dataset_option} not recognized. Choose between 1 and {len(dataset_names)}.")
    
    dataset_name = dataset_names[dataset_option]
    dataset_dir = os.path.join(data_dir, dataset_name)

    # Train dataset directories
    train_dir = os.path.join(dataset_dir, 'Train_dataset')
    sdr_colour_dir = os.path.join(train_dir, 'SDR_train_colour_tiff')
    sdr_luminance_dir = os.path.join(train_dir, 'SDR_train_luminance')
    hdr_colour_dir = os.path.join(train_dir, 'HDR_train_colour_tiff')
    hdr_luminance_dir = os.path.join(train_dir, 'HDR_train_luminance')
    
    # Test dataset directories
    test_dir = os.path.join(dataset_dir, 'Test_dataset')
    sdr_test_luminance_dir = os.path.join(test_dir, 'SDR_test_luminance')
    hdr_test_luminance_dir = os.path.join(test_dir, 'HDR_test_luminance')
    sdr_test_colour_tiff_dir = os.path.join(test_dir, 'SDR_test_colour_tiff')
    hdr_test_colour_tiff_dir = os.path.join(test_dir, 'HDR_test_colour_tiff')

    return {
        "dataset_dir": dataset_dir,
        "sdr_colour_dir": sdr_colour_dir,
        "sdr_luminance_dir": sdr_luminance_dir,
        "hdr_colour_dir": hdr_colour_dir,
        "hdr_luminance_dir": hdr_luminance_dir,
        "sdr_test_luminance_dir": sdr_test_luminance_dir,
        "hdr_test_luminance_dir": hdr_test_luminance_dir,
        "sdr_test_colour_tiff_dir": sdr_test_colour_tiff_dir,
        "hdr_test_colour_tiff_dir": hdr_test_colour_tiff_dir
    }
