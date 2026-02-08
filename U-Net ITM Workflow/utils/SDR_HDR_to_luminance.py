import os
import glob
import numpy as np
from PIL import Image
import tifffile
from bt709_to_bt2020 import calculate_luminance, rgb709_to_xyz_to_yuv
import shutil

# Hilfsfunktionen
def PQ_EOTF(img):
    m1 = 0.1593017578125 
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    E_ = np.power(img, 1/m2)
    # Debugging: Ausgabe der Min- und Max-Werte nach EOTF-Transformation
    print(f"After EOTF step 1 - Min: {E_.min()}, Max: {E_.max()}")
    max = np.where(E_ - c1 < 0, 0, E_- c1)
    Y = np.power((max)/(c2 - (c3 * E_)), 1/m1)
    Fd = 10000 * Y                       # Adjust to moviematerial max nits
    # Debugging: Ausgabe der Min- und Max-Werte nach der PQ-Umwandlung
    print(f"After PQ EOTF - Min: {Fd.min()}, Max: {Fd.max()}")

    return Fd

def rgb2020_to_xyz_to_yuv(rgb_image):
    RGB_to_XYZ_matrix = np.array([
        [0.6369580483012914, 0.14461690358620832, 0.1688809751641721],
        [0.2627002120112671, 0.6779980715188708, 0.05930171646986196],
        [0.0000000000000000, 0.028072693049087428, 1.060985057710791]
    ])
    shape = rgb_image.shape
    xyz_image = np.dot(rgb_image.reshape(-1, 3), RGB_to_XYZ_matrix.T).reshape(shape)
    X, Y, Z = xyz_image[..., 0], xyz_image[..., 1], xyz_image[..., 2]
    # Debugging: Ausgabe der Min- und Max-Werte von Y (Luminanz)
    print(f"Y (Luminance) - Min: {Y.min()}, Max: {Y.max()}")
    denom = X + 15 * Y + 3 * Z + 1e-10
    u_prime = 4 * X / denom
    v_prime = 9 * Y / denom
    Yuv_image = np.stack((Y, u_prime, v_prime), axis=-1)

    return Y, u_prime, v_prime, Yuv_image

# Relevante Funktionen zur Umwandlung
def sdrImg_to_dispLight(input_file):
    img = Image.open(input_file)
    arr = np.array(img, dtype=np.float32)
    # Debugging: Ausgabe der Min- und Max-Werte vor der Normalisierung
    print(f"Before normalization (SDR) - Min: {arr.min()}, Max: {arr.max()}")    
    arr_norm = arr / 255    
    arr_clamped = np.clip(arr_norm, 0.0, 1.0)
    # Debugging: Ausgabe der Min- und Max-Werte nach der Normalisierung
    print(f"After normalization (SDR) - Min: {arr_clamped.min()}, Max: {arr_clamped.max()}")
    L, u, v, Yuv_arr = rgb709_to_xyz_to_yuv(arr_clamped)
    # disp_light = calculate_luminance(L)
    # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
    print("After sdr image to displaylight - Min:", L.min(), "Max:", L.max())
    return L

def hdrTiff_to_dispLight(input_file):
    ima = tifffile.imread(input_file)
    arr = np.array(ima, dtype=np.float32)
    # Debugging: Ausgabe der Min- und Max-Werte vor der Normalisierung
    print(f"Before normalization (HDR) - Min: {arr.min()}, Max: {arr.max()}")    
    norm_arr = arr / 65535  
    arr_clamped = np.clip(norm_arr, 0.0, 1.0)  
    # Debugging: Ausgabe der Min- und Max-Werte nach der Normalisierung
    print(f"After normalization (HDR) - Min: {arr_clamped.min()}, Max: {arr_clamped.max()}")
    L, u, v, Yuv_arr = rgb2020_to_xyz_to_yuv(arr_clamped)
    # disp_light = PQ_EOTF(L)
    # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
    print("After hdr image to displaylight - Min:", L.min(), "Max:", L.max())
    return L

# Hauptskript zur Umwandlung
def convert_images(sdr_dir, hdr_dir, sdr_output_dir, hdr_output_dir):
    os.makedirs(sdr_output_dir, exist_ok=True)
    os.makedirs(hdr_output_dir, exist_ok=True)

    # Ordner leeren
    clear_folder(sdr_output_dir)
    # Ordner leeren
    clear_folder(hdr_output_dir)

    # SDR-PNGs in Luminanz umwandeln und speichern
    sdr_paths = glob.glob(os.path.join(sdr_dir, '*.tiff'))
    for sdr_path in sdr_paths:
        disp_light = sdrImg_to_dispLight(sdr_path)
        output_path = os.path.join(sdr_output_dir, os.path.basename(sdr_path).replace('.tiff', '_luminance.npy'))
        np.save(output_path, disp_light)
        print(f'Saved SDR luminance image: {output_path}')
        # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
        print(f"SDR Luminance for {os.path.basename(sdr_path)} - Min:", disp_light.min(), "Max:", disp_light.max())

    # HDR-TIFFs in Luminanz umwandeln und speichern
    hdr_paths = glob.glob(os.path.join(hdr_dir, '*.tiff'))
    for hdr_path in hdr_paths:
        disp_light = hdrTiff_to_dispLight(hdr_path)
        output_path = os.path.join(hdr_output_dir, os.path.basename(hdr_path).replace('.tiff', '_luminance.npy'))
        np.save(output_path, disp_light)
        print(f'Saved HDR luminance image: {output_path}')
        # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
        print(f"HDR Luminance for {os.path.basename(sdr_path)} - Min:", disp_light.min(), "Max:", disp_light.max())

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

# Verzeichnisse
sdr_dir = r'C:\Users\rippthehorror\Documents\Lernmaterial\Star_Trek_Nemesis_SDR_Klein'
hdr_dir = r'C:\Users\rippthehorror\Documents\Lernmaterial\Star_Trek_Nemesis_HDR_FullHD_Klein'
sdr_output_dir = r'U-Net Upmapping Workflow\data\SDR_luminance'
hdr_output_dir = r'U-Net Upmapping Workflow\data\HDR_luminance'

# Umwandlung ausführen
convert_images(sdr_dir, hdr_dir, sdr_output_dir, hdr_output_dir)









def sdrImg_to_dispLight(input_file):
    img = Image.open(input_file)
    arr = np.array(img, dtype=np.float32)
    # Debugging: Ausgabe der Min- und Max-Werte vor der Normalisierung
    print(f"Before normalization (SDR) - Min: {arr.min()}, Max: {arr.max()}")    
    arr_norm = arr / 255    
    arr_clamped = np.clip(arr_norm, 0.0, 1.0)
    # Debugging: Ausgabe der Min- und Max-Werte nach der Normalisierung
    print(f"After normalization (SDR) - Min: {arr_clamped.min()}, Max: {arr_clamped.max()}")
    L, u, v, Yuv_arr = rgb709_to_xyz_to_yuv(arr_clamped)
    # disp_light = calculate_luminance(L)
    # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
    print("After sdr image to displaylight - Min:", L.min(), "Max:", L.max())
    return L


def hdrTiff_to_dispLight(input_file):
    ima = tifffile.imread(input_file)
    arr = np.array(ima, dtype=np.float32)
    # Debugging: Ausgabe der Min- und Max-Werte vor der Normalisierung
    print(f"Before normalization (HDR) - Min: {arr.min()}, Max: {arr.max()}")    
    norm_arr = arr / 65535  
    arr_clamped = np.clip(norm_arr, 0.0, 1.0)  
    # Debugging: Ausgabe der Min- und Max-Werte nach der Normalisierung
    print(f"After normalization (HDR) - Min: {arr_clamped.min()}, Max: {arr_clamped.max()}")
    L, u, v, Yuv_arr = rgb2020_to_xyz_to_yuv(arr_clamped)
    # disp_light = PQ_EOTF(L)
    # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
    print("After hdr image to displaylight - Min:", L.min(), "Max:", L.max())
    return L


def convert_images_neu(data_dir):

    train_dir = os.path.join(data_dir, "Train_dataset")
    test_dir = os.path.join(data_dir, "Test_dataset")

    sdr_test_colour_dir = os.path.join(test_dir, "SDR_test_colour_tiff")
    hdr_test_colour_dir = os.path.join(test_dir, "HDR_test_colour_tiff")

    sdr_train_colour_dir = os.path.join(train_dir, "SDR_train_colour_tiff")
    hdr_train_colour_dir = os.path.join(train_dir, "HDR_train_colour_tiff")

    sdr_test_luminance_dir = os.path.join(test_dir, "SDR_test_luminance")
    hdr_test_luminance_dir = os.path.join(test_dir, "SDR_test_luminance")

    sdr_train_luminance_dir = os.path.join(train_dir, "SDR_train_luminance")
    hdr_train_luminance_dir = os.path.join(train_dir, "SDR_train_luminance")

    # Ordner machen
    os.makedirs(sdr_test_luminance_dir, exist_ok=True)
    os.makedirs(hdr_test_luminance_dir, exist_ok=True)
    os.makedirs(sdr_train_luminance_dir, exist_ok=True)
    os.makedirs(hdr_train_luminance_dir, exist_ok=True)

    # Ordner leeren
    clear_folder(sdr_test_luminance_dir)
    clear_folder(hdr_test_luminance_dir)
    clear_folder(sdr_train_luminance_dir)
    clear_folder(hdr_train_luminance_dir)

    # SDR-Test-tiffs in Luminanz umwandeln und speichern
    sdr_test_paths = glob.glob(os.path.join(sdr_test_colour_dir, '*.tiff'))
    for sdr_test_path in sdr_test_paths:
        
        sdr_test_disp_light = sdrImg_to_dispLight(sdr_test_path)
        
        sdr_test_luminance_path = os.path.join(sdr_test_luminance_dir, os.path.basename(sdr_test_path).replace('.tiff', '_luminance.npy'))
        np.save(sdr_test_luminance_path, sdr_test_disp_light)
        print(f'Saved SDR luminance image: {sdr_test_luminance_path}')
        # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
        print(f"SDR Luminance for {os.path.basename(sdr_test_luminance_path)} - Min:", sdr_test_disp_light.min(), "Max:", sdr_test_disp_light.max())

    # SDR-Train-tiffs in Luminanz umwandeln und speichern
    sdr_train_paths = glob.glob(os.path.join(sdr_train_colour_dir, '*.tiff'))
    for sdr_train_path in sdr_train_paths:
        
        sdr_train_disp_light = sdrImg_to_dispLight(sdr_train_path)
        
        sdr_train_luminance_path = os.path.join(sdr_train_luminance_dir, os.path.basename(sdr_train_path).replace('.tiff', '_luminance.npy'))
        np.save(sdr_train_luminance_path, sdr_train_disp_light)
        print(f'Saved SDR luminance image: {sdr_train_luminance_path}')
        # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
        print(f"SDR Luminance for {os.path.basename(sdr_train_luminance_path)} - Min:", sdr_train_disp_light.min(), "Max:", sdr_train_disp_light.max())





    # HDR-TIFFs in Luminanz umwandeln und speichern
    hdr_test_paths = glob.glob(os.path.join(hdr_test_colour_dir, '*.tiff'))
    for hdr_test_path in hdr_test_paths:
        
        hdr_test_disp_light = hdrTiff_to_dispLight(hdr_test_path)
        
        
        hdr_test_luminance_path = os.path.join(hdr_output_dir, os.path.basename(hdr_test_path).replace('.tiff', '_luminance.npy'))
        np.save(hdr_test_luminance_path, hdr_test_disp_light)
        print(f'Saved HDR luminance image: {hdr_test_luminance_path}')
        # Debugging: Ausgabe der Min- und Max-Werte der Eingabedaten
        print(f"HDR Luminance for {os.path.basename(hdr_test_luminance_path)} - Min:", hdr_test_disp_light.min(), "Max:", hdr_test_disp_light.max())