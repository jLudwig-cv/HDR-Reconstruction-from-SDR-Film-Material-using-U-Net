import numpy as np
import os
import tifffile as tiff 
import matplotlib.pyplot as plt
import subprocess
from PIL import Image

def inverse_PQ_EOTF(L):
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    
    Y_clip = np.where(L < 0, 0, L)
    num = c1 + c2 * np.power(Y_clip, m1)
    denum = 1 + c3 * np.power(Y_clip, m1)

    # Inverse PQ-Funktion anwenden
    E = np.power(num/denum, m2)    

    return E

def PQ_EOTF(img):
    m1 = 0.1593017578125 
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    img_clip = np.clip(img, 1e-10, 0.99999)
    E_ = np.power(img_clip, 1/m2)

    max = np.where(E_ - c1 < 0, 0, E_- c1)
    Y = np.power((max)/(c2 - (c3 * E_)), 1/m1)

    return Y

def calculate_luminance(V, gamma=2.404):
    Lw = 100
    Lb = 0
    a = (Lw ** (1/gamma) - Lb ** (1/gamma)) ** gamma
    b = Lb ** (1/gamma) / ((Lw ** (1/gamma) - Lb ** (1/gamma)) ** gamma)
    L = a * np.maximum((V + b), 0) ** gamma

    return L

def rec709_to_rec2020(rgb_709):
    # RGB as input
    rec709_to_rec2020_matrix = np.array([
        [0.6274, 0.3293, 0.0433],
        [0.0691, 0.9195, 0.0114],
        [0.0164, 0.0880, 0.8956]
    ])

    # Normalize Rec709 image
    rec_709_norm = ((rgb_709))

    # Lineralize Rec709 Image
    rec_709_lin = calculate_luminance(rec_709_norm)

    rec_709_flat = rec_709_lin.reshape(-1, 3)
    
    rec_2020_flat = np.dot(rec_709_flat, rec709_to_rec2020_matrix.T)

    rec2020_lin = rec_2020_flat.reshape(rgb_709.shape)
    rec2020_norm = rec2020_lin / 4000
    arr_PQ = np.zeros_like(rec2020_lin)
   
    # Anwendung der EOTF
    for i in range(3):  # Durchlaufen der RGB-Kanäle
        arr_PQ[:, :, i] = inverse_PQ_EOTF(rec2020_norm[:, :, i])

    return arr_PQ


rgb_to_lms_matrix = np.array([
    [1688, 2146, 262],
    [683, 2951, 462],
    [99, 309, 3688]
]) / 4096

lms_to_rgb_matrix = np.linalg.inv(rgb_to_lms_matrix)

lms_to_ictcp_matrix = np.array([
    [2048, 2048, 0.0],
    [6610, -13613, 7003],
    [17933, -17390, -543]
]) / 4096

ictcp_to_lms_matrix = np.linalg.inv(lms_to_ictcp_matrix)

 
def rgb2020_to_ICtCp(rgb2020_img):
    # Linearisierung von RGB2020
    rgb2020_lin = PQ_EOTF(rgb2020_img)

    rgb2020_flat = rgb2020_lin.reshape(-1, 3)

    # RGB to LMS
    lms = np.dot(rgb2020_flat, rgb_to_lms_matrix)

    lms_img = lms.reshape(rgb2020_lin.shape)
    L = lms_img[..., 0]
    M = lms_img[..., 1]
    S = lms_img[..., 2]

    # Apply PQ encoding
    L_pq = inverse_PQ_EOTF(L)
    M_pq = inverse_PQ_EOTF(M)
    S_pq = inverse_PQ_EOTF(S)

    lms_pq = np.stack((L_pq, M_pq, S_pq), axis=-1)
    lms_pq_flat = lms_pq.reshape(-1, 3)

    # LMS to ICtCp
    ictcp = np.dot(lms_pq_flat ,lms_to_ictcp_matrix.T)

    ictcp = ictcp.reshape(rgb2020_img.shape)

    return ictcp

def ICtCp_to_rbg2020(ICtCp_img):
    ICtCp_img_flat = ICtCp_img.reshape(-1, 3)

    # ICtCp to LMS_PQ
    lms_pq = np.dot(ICtCp_img_flat, ictcp_to_lms_matrix.T)

    # Apply PQ decoding
    lms_img = lms_pq.reshape(ICtCp_img.shape)
    L = lms_img[..., 0]
    M = lms_img[..., 1]
    S = lms_img[..., 2]

    # Apply PQ encoding
    L_pq = PQ_EOTF(L)
    M_pq = PQ_EOTF(M)
    S_pq = PQ_EOTF(S)

    lms_pq = np.stack((L_pq, M_pq, S_pq), axis=-1)
    lms_pq_flat = lms_pq.reshape(-1, 3)

    # LMS to RGB
    rgb2020 = np.dot(lms_pq_flat, lms_to_rgb_matrix)

    rgb2020 = rgb2020.reshape(ICtCp_img.shape)

    rgb2020_PQ = inverse_PQ_EOTF(rgb2020)
    return rgb2020_PQ



# Function to convert luminance array to 16-bit TIFF format
def luminance_to_hdr(lum_array, max_nits):
    # Normalize the luminance values
    lum_norm = np.clip(lum_array, 0.005, 4000)

    # Convert using the inverse PQ EOTF function
    lum_PQ = inverse_PQ_EOTF(lum_norm, max_nits)
    lum_16bit = lum_PQ * 65536
    lum_final = lum_16bit.astype(np.uint16)

    return lum_final

# Main function to handle conversion from .npy to .tiff and organizing in directories
def npy_to_bwtiff(output_dir, gt_dir, max_nits):
    npy_dir = os.path.join(output_dir, '.npy')

    # Create separate folders for predicted and ground truth TIFFs
    tiff_dir = os.path.join(output_dir, 'Tiff')


    tiff_gt_dir = os.path.join(tiff_dir, 'GT_Luminance')
    tiff_pred_dir = os.path.join(tiff_dir, 'Pred_Luminance')

    os.makedirs(tiff_gt_dir, exist_ok=True)
    os.makedirs(tiff_pred_dir, exist_ok=True)

    # Load all .npy files
    pred_files = sorted([f for f in os.listdir(npy_dir) if f.endswith(".npy")])
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".npy")])
    
    # Ensure both directories have the same number of files
    if len(pred_files) != len(gt_files):
        print("The number of files in the directories does not match!")
        return

    # Process each file individually
    for i, (pred_file, gt_file) in enumerate(zip(pred_files, gt_files)):

        # Create paths for the current files
        pred_path = os.path.join(npy_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        # Load the .npy arrays (single image per file)
        pred_luminance_array = np.load(pred_path)
        pred_luminance_array = np.clip(pred_luminance_array, 0, 65535)
        GT_luminance_array = np.load(gt_path) * 65535

        # Convert predicted luminance to TIFF
        # pred_tiff = luminance_to_hdr(pred_luminance_array, max_nits)
        pred_filename = os.path.join(tiff_pred_dir, f"predicted_hdr_{i+1:04d}.tiff")
        tiff.imwrite(pred_filename, pred_luminance_array.astype(np.uint16))
        print(f"Saved Image pred_lum Tiff of image {i+1} as {pred_filename}")

        # Convert ground truth luminance to TIFF
        # gt_tiff = luminance_to_hdr(GT_luminance_array, max_nits)
        gt_filename = os.path.join(tiff_gt_dir, f"gt_hdr_{i+1:04d}.tiff")
        tiff.imwrite(gt_filename, GT_luminance_array.astype(np.uint16))
        print(f"Saved Image gt_Lum Tiff of image {i+1} as {gt_filename}")




def one_bwtiff_to_avif(gt_colour_tiff_dir, sdr_colour_tiff_dir, pred_lum_dir, output_dir, image_count): 

    avif_dir = os.path.join(output_dir, 'Avif')
    GT_avif_dir = os.path.join(avif_dir, 'GT')

    pred_GTcolour_avif_dir = os.path.join(avif_dir, 'Pred_GTFarbe')
    pred_SDRcolour_avif_dir = os.path.join(avif_dir, 'Pred_SDRFarbe')

    tiff_dir = avif_dir = os.path.join(output_dir, 'Tiff')
    pred_GTcolour_tiff_dir = os.path.join(tiff_dir, 'Pred_GTFarbe')
    pred_SDRcolour_tiff_dir = os.path.join(tiff_dir, 'Pred_SDRFarbe')
    GT_tiff_dir = os.path.join(tiff_dir, 'GT_Farbe')
    
    os.makedirs(pred_GTcolour_avif_dir, exist_ok=True)
    os.makedirs(pred_SDRcolour_avif_dir, exist_ok=True)
    os.makedirs(GT_avif_dir, exist_ok=True)
    os.makedirs(pred_GTcolour_tiff_dir, exist_ok=True)
    os.makedirs(pred_SDRcolour_tiff_dir, exist_ok=True)
    os.makedirs(GT_tiff_dir, exist_ok=True)

    # Einlesen GT Bild
    GT_tiff = tiff.imread(gt_colour_tiff_dir)
    Pred_tiff = tiff.imread(pred_lum_dir)
    # SDR_tiff = tiff.imread(sdr_colour_tiff_dir)
    SDR_tiff = Image.open(sdr_colour_tiff_dir)

    # Normalisierung
    GT_arr = np.array(GT_tiff, dtype=np.float32) / 65535
    Pred_arr = np.array(Pred_tiff, dtype=np.float32) / 65535
    SDR_arr = np.array(SDR_tiff, dtype=np.float32) / 255

    SDR_2020_arr = rec709_to_rec2020(SDR_arr)

    # Konvertiere RGB zu YUV
    ICtCp_GT_img = rgb2020_to_ICtCp(GT_arr)
    ICtCp_SDR_img = rgb2020_to_ICtCp(SDR_2020_arr)

    I_GT = ICtCp_GT_img[..., 0]
    Ct_GT = ICtCp_GT_img[..., 1]
    Cp_GT = ICtCp_GT_img[..., 2]

    I_SDR = ICtCp_SDR_img[..., 0]
    Ct_SDR = ICtCp_SDR_img[..., 1]
    Cp_SDR = ICtCp_SDR_img[..., 2]

    L_pred_clip = np.clip(Pred_arr, 0, None)

    # Ersetzen des Luminanzkanals
    ICtCp_pred_image_GTcolour = np.stack((L_pred_clip, Ct_GT, Cp_GT), axis=-1)

    ICtCp_pred_image_SDRcolour = np.stack((L_pred_clip, Ct_SDR, Cp_SDR), axis=-1)

    ICtCp_GT_image = np.stack((I_GT, Ct_GT, Cp_GT), axis=-1)

    # Umwandlung in RGB
    RGB_pred_image_GTcolour = ICtCp_to_rbg2020(ICtCp_pred_image_GTcolour) * 65535

    RGB_pred_image_SDRcolour = ICtCp_to_rbg2020(ICtCp_pred_image_SDRcolour) * 65535

    # print("GT Image")
    RGB_GT_image = ICtCp_to_rbg2020(ICtCp_GT_image) * 65535

    # RGB auf 16 bit
    RGB_pred_image_GTcolour_16bit = np.array(RGB_pred_image_GTcolour, dtype=np.uint16)
    RGB_pred_image_SDRcolour_16bit = np.array(RGB_pred_image_SDRcolour, dtype=np.uint16)
    RGB_GT_image_16bit = np.array(RGB_GT_image, dtype=np.uint16)

    new_pred_GTcolour_tiff_path = os.path.join(pred_GTcolour_tiff_dir, f"pred_gtcolour_{image_count:04d}.tiff")
    new_pred_GTcolour_avif_path = os.path.join(pred_GTcolour_avif_dir, f"pred_gtcolour_{image_count:04d}.avif")

    new_pred_SDRcolour_tiff_path = os.path.join(pred_SDRcolour_tiff_dir, f"pred_sdrcolour_{image_count:04d}.tiff")
    new_pred_SDRcolour_avif_path = os.path.join(pred_SDRcolour_avif_dir, f"pred_sdrcolour_{image_count:04d}.avif")

    GT_tiff_path = os.path.join(GT_tiff_dir, f"gt_{image_count:04d}.tiff")
    GT_avif_path = os.path.join(GT_avif_dir, f"gt_{image_count:04d}.avif")

    # Speichere TIFF-Bilder
    tiff.imwrite(new_pred_GTcolour_tiff_path, RGB_pred_image_GTcolour_16bit)
    tiff.imwrite(new_pred_SDRcolour_tiff_path, RGB_pred_image_SDRcolour_16bit)
    tiff.imwrite(GT_tiff_path, RGB_GT_image_16bit)

    # ffmpeg Befehle zum Speichern der AVIF-Dateien
    save_pred_GTcolour = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', new_pred_GTcolour_tiff_path,
        '-vf', 'zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:rin=full:t=smpte2084:p=bt2020:m=bt2020nc:r=full:c=topleft',
        '-c:v', 'libaom-av1',
        '-crf', '0',
        '-preset', 'slow',
        '-strict', 'experimental',
        '-pix_fmt', 'yuv420p10le',
        '-metadata:s:v:0', 'master-display=G(26500,69000)B(15000,6000)R(68000,32000)WP(31270,32900)L(40000000,50)',
        '-metadata:s:v:0', 'max-cll=1000,716',
        new_pred_GTcolour_avif_path
    ]

    save_pred_SDRcolour = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', new_pred_SDRcolour_tiff_path,
        '-vf', 'zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:rin=full:t=smpte2084:p=bt2020:m=bt2020nc:r=full:c=topleft',
        '-c:v', 'libaom-av1',
        '-crf', '0',
        '-preset', 'slow',
        '-strict', 'experimental',
        '-pix_fmt', 'yuv420p10le',
        '-metadata:s:v:0', 'master-display=G(26500,69000)B(15000,6000)R(68000,32000)WP(31270,32900)L(40000000,50)',
        '-metadata:s:v:0', 'max-cll=1000,716',
        new_pred_SDRcolour_avif_path
    ]

    save_GT = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', GT_tiff_path,
        '-vf', 'zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:rin=full:t=smpte2084:p=bt2020:m=bt2020nc:r=full:c=topleft',
        '-c:v', 'libaom-av1',
        '-crf', '0',
        '-preset', 'slow',
        '-strict', 'experimental',
        '-pix_fmt', 'yuv420p10le',
        '-metadata:s:v:0', 'master-display=G(26500,69000)B(15000,6000)R(68000,32000)WP(31270,32900)L(40000000,50)',
        '-metadata:s:v:0', 'max-cll=1000,716',
        GT_avif_path
    ]
  
    # subprocess.run(save_pred_GTcolour)
    # print(f"Saved Pred_GTColour AVIF of image {image_count} as {new_pred_GTcolour_avif_path}")

    # subprocess.run(save_pred_SDRcolour)
    # print(f"Saved Pred_SDRColour AVIF of image {image_count} as {new_pred_SDRcolour_avif_path}")

    # subprocess.run(save_GT)
    # print(f"Saved GT AVIF of image {image_count} as {GT_avif_path}")




# Funktion zur Verarbeitung aller Bilder in einem Verzeichnis
def process_bw_tiff_folder_to_colour_avif(gt_colour_tiff_path, sdr_colour_tiff_path, output_dir, max_nits):

    tiff_dir = os.path.join(output_dir, 'Tiff')
    pred_tiff_bw_dir = os.path.join(tiff_dir, 'Pred_Luminance')
    pred_bw_images = sorted([os.path.join(pred_tiff_bw_dir, f) for f in os.listdir(pred_tiff_bw_dir) if f.endswith('.tiff')])

    
    gt_images = sorted([os.path.join(gt_colour_tiff_path, f) for f in os.listdir(gt_colour_tiff_path) if f.endswith('.tiff')])
    sdr_images = sorted([os.path.join(sdr_colour_tiff_path, f) for f in os.listdir(sdr_colour_tiff_path) if f.endswith('.tiff')])
    i = 0
    for gt_dir, sdr_dir, pred_lum_dir in zip(gt_images, sdr_images, pred_bw_images):
        i = i+1
        one_bwtiff_to_avif(gt_dir, sdr_dir, pred_lum_dir, output_dir, i)



def process_colour_tiff_folder_to_colour_avif(output_folder):
    # Sicherstellen, dass der Ausgabepfad existiert
    tiff_folder = os.path.join(output_folder, 'Tiff')
    avif_folder = os.path.join(output_folder, 'Avif')

    tiff_pred_folder = os.path.join(tiff_folder, 'Pred')
    avif_pred_folder = os.path.join(avif_folder, 'Pred')
    os.makedirs(tiff_pred_folder, exist_ok=True)
    os.makedirs(avif_pred_folder, exist_ok=True)

    tiff_GT_folder = os.path.join(tiff_folder, 'GT')
    avif_GT_folder = os.path.join(avif_folder, 'GT')
    os.makedirs(tiff_GT_folder, exist_ok=True)
    os.makedirs(avif_GT_folder, exist_ok=True)

    image_count = 0
    # Iteriere durch alle TIFF-Dateien im Eingabeordner
    for filename in os.listdir(tiff_pred_folder):
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            image_count = image_count + 1
            # Erstelle den vollständigen Pfad zur TIFF-Datei
            tiff_pred_path = os.path.join(tiff_pred_folder, filename)
            
            # Erstelle den Pfad zur Ausgabe-AVIF-Datei
            output_filename = f"Pred_{image_count:04d}.avif"
            avif_pred_path = os.path.join(avif_pred_folder, output_filename)
            
            # Definiere den FFMPEG-Befehl
            save_pred = [
                'ffmpeg',
                '-loglevel', 'error',
                '-i', tiff_pred_path,
                '-vf', 'zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:rin=full:t=smpte2084:p=bt2020:m=bt2020nc:r=full:c=topleft',
                '-c:v', 'libaom-av1',
                '-crf', '0',
                '-preset', 'slow',
                '-strict', 'experimental',
                '-pix_fmt', 'yuv420p10le',
                '-metadata:s:v:0', 'master-display=G(26500,69000)B(15000,6000)R(68000,32000)WP(31270,32900)L(40000000,50)',
                '-metadata:s:v:0', 'max-cll=1000,716',
                avif_pred_path
            ]

            # Führe den FFMPEG-Befehl aus
            subprocess.run(save_pred)
            print(f"Saved Pred AVIF of image {image_count} as {avif_pred_path}")

    image_count = 0
    for filename in os.listdir(tiff_GT_folder):
        image_count = image_count + 1
        if filename.endswith(".tiff") or filename.endswith(".tif"):
            # Erstelle den vollständigen Pfad zur TIFF-Datei
            tiff_GT_path = os.path.join(tiff_GT_folder, filename)
            
            # Erstelle den Pfad zur Ausgabe-AVIF-Datei
            output_filename = f"GT_{image_count:04d}.avif"
            avif_GT_path = os.path.join(avif_GT_folder, output_filename)
            
            # Definiere den FFMPEG-Befehl
            save_GT = [
                'ffmpeg',
                '-loglevel', 'error',
                '-i', tiff_GT_path,
                '-vf', 'zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:rin=full:t=smpte2084:p=bt2020:m=bt2020nc:r=full:c=topleft',
                '-c:v', 'libaom-av1',
                '-crf', '0',
                '-preset', 'slow',
                '-strict', 'experimental',
                '-pix_fmt', 'yuv420p10le',
                '-metadata:s:v:0', 'master-display=G(26500,69000)B(15000,6000)R(68000,32000)WP(31270,32900)L(40000000,50)',
                '-metadata:s:v:0', 'max-cll=1000,716',
                avif_GT_path
            ]

            # Führe den FFMPEG-Befehl aus
            subprocess.run(save_GT)
            print(f"Saved GT AVIF of image {image_count} as {avif_GT_path}")