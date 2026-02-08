import numpy as np
import subprocess
from PIL import Image
import imageio
import os
from img_analysis import plot_histogram_nits, plot_cie1976_diagram_with_gamuts
import config 

# Hauptfunktion zur Umwandlung von SDR auf HDR nach BT.2446-1
def image_processing(input_dir, temp_dir, output_dir, input_image, avif_conversion ):

    if input_image:
        # If a specific test file is provided, process only that file
        files = [os.path.basename(input_image)]  # Ensure it's just the file name
        input_dir = os.path.dirname(input_image)  # Override the input_dir to the directory of the test file

    else:
        # Otherwise, process all .png files in the directory
        files = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    total_files = len(files)
    processed_files = 0

    for file_name in files:

        file_extension = os.path.splitext(file_name)[1].lower()  # Extract the file extension
        base_name = os.path.splitext(file_name)[0]  # Extract the base file name

        # Define the new file paths with appropriate replacements for extensions
        input_file = os.path.join(input_dir, file_name)
        temp_file = os.path.join(temp_dir, f"{base_name}.tiff")
        output_file = os.path.join(output_dir, f"{base_name}.avif")
        
        try:    
            # Einlesen der Bilder
            img = Image.open(input_file)

            # Bild auf eine gerade Größe bringen
            img_crop = crop_image(img)

            # Umwandeln des Bildes in Array
            arr = np.array(img_crop, dtype=np.float32)

            # Plotten der Farben des Eingangsbildes
            if config.show_colordiagram == True and avif_conversion == True:
                plot_cie1976_diagram_with_gamuts(arr, "Farben vor Umwandlung")

            # Normalisieren des Arrays
            arr_norm = arr / 255

            # Berechne die Luminanz für jedes Pixel und jeden Kanal
            Dis_Light = calculate_luminance(arr_norm)

            # Umwandlung zu yuv
            L, u, v, Yuv_arr = rgb709_to_xyz_to_yuv(Dis_Light)

            # Plotten der Helligkeiten des Eingangsbildes
            if config.show_histogram == True and avif_conversion == True:
                plot_histogram_nits(L, "Helligkeiten vor Umwandlung")

            # Skalierung der Luminanz
            L_ska = L * 2

            # Gamma Adjustment in der Luminanz
            L_adj = np.power(L_ska, np.power(1.1111, np.log2(200/100)))

            # Tonemapping (Anhebung der Spitzenlichter)
            L_TM = inverse_tone_mapping(L_adj)

            # Plotten der Helligkeiten nach des finalen Bildes
            if config.show_histogram == True and avif_conversion == True:
                plot_histogram_nits(L_TM, "Helligkeiten nach Umwandlung")

            # Zusammenführung der Y- U- und V-Kanäle
            Yuv_arr_HDR = np.stack((L_TM, u, v), axis=-1)

            # Umwandlung zu RGB2020
            arr_rgb2020 = yuv_to_xyz_to_rgb2020(Yuv_arr_HDR)
            arr_PQ = np.zeros_like(arr_rgb2020)

            # Anwendung der EOTF
            for i in range(3):  # Durchlaufen der RGB-Kanäle
                arr_PQ[:, :, i] = inverse_pq(arr_rgb2020[:, :, i])

            # Skalierung auf 16 Bit und Zwischenspeicher als TIFF
            arr_PQ_16 = (arr_PQ * 65535).astype(np.uint16)

            # Plotten der Farben des Ausgangsbildes
            if config.show_colordiagram == True and avif_conversion == True:
                plot_cie1976_diagram_with_gamuts(arr, "Farben nach Umwandlung")

            # Temporäres Speichern der Tiff
            imageio.imwrite(temp_file, arr_PQ_16, format='tiff')

            # Bei Umwandlung eines Einzelbildes
            if avif_conversion is True:
                # Definition des FFMpeg Kommandos zur Umwandlung eines Tiffs in ein Avif
                ffmpeg_command = [
                    'ffmpeg',
                    '-loglevel', 'error',
                    '-i', temp_file,
                    '-vf', 'zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:rin=full:t=smpte2084:p=bt2020:m=bt2020nc:r=full:c=topleft',
                    '-c:v', 'libaom-av1',
                    '-crf', '0',
                    '-preset', 'slow',
                    '-strict', 'experimental',
                    '-pix_fmt', 'yuv420p10le',
                    '-metadata:s:v:0', 'master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(1000,50)',
                    '-metadata:s:v:0', 'max-cll=1000,400',
                    output_file
                ]

                #Ausführen des Kommandos
                process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if process.returncode == 0:
                    print("AVIF Umwandlung war erfolgreich.")
                else:
                    print(f"Fehler in der AVIF Umwandlung: {process.stderr}")

            processed_files += 1
            print(f'Verarbeitet: {processed_files}/{total_files} Frames ({(processed_files/total_files * 100):.2f}%)')

        except IOError as e:
            print(f"Kann Bild {file_name} nicht öffnen. Fehler: {e}")
            


# Umrechnung des SDR Bildes von Bitwerten auf Luminanz
def calculate_luminance(V, gamma=2.404):
    Lw = 100
    Lb = 0
    a = (Lw ** (1/gamma) - Lb ** (1/gamma)) ** gamma
    b = Lb ** (1/gamma) / ((Lw ** (1/gamma) - Lb ** (1/gamma)) ** gamma)
    L = a * np.maximum((V + b), 0) ** gamma

    return L


# Umwandeln des Farbraums des SDR Bildes von Rec.709 zu Yuv
def rgb709_to_xyz_to_yuv(rgb_image):
    RGB_to_XYZ_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    shape = rgb_image.shape
    xyz_image = np.dot(rgb_image.reshape(-1, 3), RGB_to_XYZ_matrix.T).reshape(shape)
    X, Y, Z = xyz_image[..., 0], xyz_image[..., 1], xyz_image[..., 2]
    denom = X + 15 * Y + 3 * Z + 1e-10
    u_prime = 4 * X / denom
    v_prime = 9 * Y / denom
    Yuv_image = np.stack((Y, u_prime, v_prime), axis=-1)

    return Y, u_prime, v_prime, Yuv_image


# Definition einer Sigmoid-Kurve
def sigmoid(x, center, width):
    return 1 / (1 + np.exp(-(x - center) / width))


# Funktion zum Inversen Tonemapping (Erhöhen der Spitzlichter)
def inverse_tone_mapping(x, breakpoint=204, multiplier=2.3, width=50):
    # Sigmoidfunktion zur Weichheit des Übergangs
    smooth_transition = sigmoid(x, breakpoint, width)
    
    # Interpoliere zwischen x und x * multiplier basierend auf der Sigmoidfunktion
    return x * (1 - smooth_transition) + x * multiplier * smooth_transition


# Umwandeln des Farbraumes des HDR Bildes von Yuv zu Rec.2020
def yuv_to_xyz_to_rgb2020(Yuv_image):
    Y, u, v = Yuv_image[..., 0], Yuv_image[..., 1], Yuv_image[..., 2]

    denom = 6 * u - 16 * v + 12
    x = 9 * u / denom
    y = 4 * v / denom

    X = (Y / (y + 1e-10)) * x
    Z = (Y / (y + 1e-10)) * (1 - x - y)

    XYZ_image = np.stack((X, Y, Z), axis=-1)

    shape = XYZ_image.shape
    matrix = np.array([
        [1.7166512, -0.3556708, -0.2533663],
        [-0.6666844, 1.6164812, 0.0157685],
        [0.0176399, -0.0427706, 0.9421031]
    ])

    rgb2020_img = np.dot(XYZ_image.reshape(-1, 3), matrix.T).reshape(shape)

    return rgb2020_img

def yuv_to_xyz_to_rgb709(Yuv_image):
    Y, u, v = Yuv_image[..., 0], Yuv_image[..., 1], Yuv_image[..., 2]

    denom = 6 * u - 16 * v + 12
    x = 9 * u / denom
    y = 4 * v / denom

    X = (Y / (y + 1e-10)) * x
    Z = (Y / (y + 1e-10)) * (1 - x - y)

    XYZ_image = np.stack((X, Y, Z), axis=-1)

    shape = XYZ_image.shape
    matrix = np.array([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ])

    rgb709_img = np.dot(XYZ_image.reshape(-1, 3), matrix.T).reshape(shape)

    return rgb709_img


# Definition der inversen PQ Funktion
def inverse_pq(F):
    # Konstanten definieren
    m1 = 2610 / 16384
    m2 = 2523 / 32
    c1 = 3424 / 4096
    c2 = 2413 / 128
    c3 = 2392 / 128

    Y = F/10000
    Y = np.where(Y < 0, 0, Y)
    
    num = c1 + c2 * np.power(Y, m1)
    denum = 1 + c3 * np.power(Y, m1)

    # Inverse PQ-Funktion anwenden
    E = np.power(num/denum, m2)
    return E


# Bringen des Bildes auf eine gerade Größe
def crop_image(input_img):
    width, height = input_img.size
    new_width = width - width % 2
    new_height = height - height % 2
    img_cropped = input_img.crop((0, 0, new_width, new_height))
    return img_cropped
