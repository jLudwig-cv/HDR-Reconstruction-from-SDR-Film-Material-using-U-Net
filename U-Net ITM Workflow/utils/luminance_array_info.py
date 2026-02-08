import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
from bt709_to_bt2020 import yuv_to_xyz_to_rgb709, rgb709_to_xyz_to_yuv, calculate_luminance
import os

def inverse_PQ_EOTF(L):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    Y = L / 1000
    Y = np.clip(L, 0, None)
    Ym1 = np.power(Y, m1)
    E_ = ((c1 + c2 * Ym1) / (1 + c3 * Ym1))
    F = np.power(E_, m2)
    
    return F

def lum_to_tiff_pred(lum_array):
    # Normierung
    lum_pred = lum_array
    # Schwarz/Weiß tiff
    lum_norm = np.clip(lum_pred, 0, 1)
    lum_PQ = inverse_PQ_EOTF(lum_norm)
    lum_16bit = lum_PQ * 65536
    lum_final = lum_16bit.astype(np.uint16)

    return lum_final

def lum_to_tiff_GT(lum_array):
    # Normierung
    lum_pred = lum_array/1000
    # Schwarz/Weiß tiff
    lum_norm = np.clip(lum_pred, 0, 1)
    lum_PQ = inverse_PQ_EOTF(lum_norm)
    lum_16bit = lum_PQ * 65536
    lum_final = lum_16bit.astype(np.uint16)

    return lum_final


# Lade die Arrays
SDR_luminance_array = np.load(r'U-Net Upmapping Workflow\data\SDR_luminance\frame_0014_luminance.npy')
pred_luminance_array = np.load(r'U-Net Upmapping Workflow\image_output\predicted_HDR_luminance_20240903_1029\predicted_hdr_0.npy')
GT_luminance_array = np.load(r'U-Net Upmapping Workflow\data\HDR_luminance\frame_0014_luminance.npy')
# Verzeichnis des pred_luminance_array ermitteln
pred_dir = os.path.dirname(r'U-Net Upmapping Workflow\image_output\predicted_HDR_luminance_20240903_1029\predicted_hdr_0.npy')
# Dateinamen für die zu speichernden Dateien erstellen
filename_pred = os.path.join(pred_dir, "predicted_hdr_0.tiff")
filename_GT = os.path.join(pred_dir, "GT_hdr_0.tiff")


# Informationen für das SDR Array
print("SDR array info:")
print(f"Dimensions: {SDR_luminance_array.shape}")
print(f"Min value: {SDR_luminance_array.min()}")
print(f"Max value: {SDR_luminance_array.max()}")
print(f"Mean value: {SDR_luminance_array.mean()}")
print(f"Standard Deviation: {SDR_luminance_array.std()}")
print(f"Data Type: {SDR_luminance_array.dtype}")

# Informationen für das Predicted Array
print("\nPredicted array info:")
print(f"Dimensions: {pred_luminance_array.shape}")
print(f"Min value: {pred_luminance_array.min()}")
print(f"Max value: {pred_luminance_array.max()}")
print(f"Mean value: {pred_luminance_array.mean()}")
print(f"Standard Deviation: {pred_luminance_array.std()}")
print(f"Data Type: {pred_luminance_array.dtype}")

# Informationen für das Ground Truth Array
print("\nGround Truth array info:")
print(f"Dimensions: {GT_luminance_array.shape}")
print(f"Min value: {GT_luminance_array.min()}")
print(f"Max value: {GT_luminance_array.max()}")
print(f"Mean value: {GT_luminance_array.mean()}")
print(f"Standard Deviation: {GT_luminance_array.std()}")
print(f"Data Type: {GT_luminance_array.dtype}")


# # Erstelle beide Bilder und speichere es als PNG
# tiff.imwrite(filename_pred, lum_to_tiff_pred(pred_luminance_array))
# tiff.imwrite(filename_GT, lum_to_tiff_GT(GT_luminance_array))
# print(f"Die TIFF-Dateien wurde erfolgreich als '{filename_pred}' und '{filename_GT}' gespeichert.")





# Farbe hinzufügen
# sdr_file = r"C:\Users\rippthehorror\Documents\Lernmaterial\Casino_klein_SDR\frame_0017.tiff"


# # Alte SDR Farbkanäle wiederholen
# img = Image.open(sdr_file)
# arr = np.array(img, dtype=np.float32)
# arr_norm = arr / 255
# L_SDR, u_SDR, v_SDR, Yuv_arr = rgb709_to_xyz_to_yuv(arr_norm)

# # Neue Luminanzkanal hinzufügen
# Yuv_image = np.stack((L_HDR_Pred, u_SDR, v_SDR), axis=-1)

# rgb_neu = yuv_to_xyz_to_rgb709(Yuv_image)