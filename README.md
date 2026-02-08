### HDR Reconstruction from SDR Film Material using U-Net

This project investigates the impact of native dataset on the performance of a U-Net-based architecture for Inverse Tone Mapping (SDR → HDR) in film material.

The main goal is to integrate human color grading decisions into the training process by using paired SDR and HDR versions of real movies instead of synthetically generated data.

Unlike common approaches, the SDR data in this project is not generated algorithmically from HDR data via Tone Mapping.  
Instead, it is extracted from commercially available SDR and HDR film versions, resulting in a native dataset that better reflects real production pipelines.

## Model Architecture

The reconstruction model is based on a U-Net architecture designed for image-to-image translation tasks.
It follows an encoder–decoder structure that progressively compresses and reconstructs spatial information while preserving details through skip connections. These skip connections enable the direct transfer of high-resolution features from the encoding to the decoding stages, which helps maintain structural consistency in the reconstructed images.

Training is performed using paired SDR and HDR frames, enabling supervised learning of the inverse tone mapping process and facilitating accurate reconstruction of both luminance and color information.

<p align="center">
  <img src="assets/U-Net-Structure.png" width="700"/>
</p>


### Loss Function

Several loss functions were evaluated during preliminary experiments:

- Mean Squared Error (MSE)
- Structural Similarity Index (SSIM)

SSIM was selected as the final loss function due to its superior performance.

### Color Representation

Two approaches were evaluated:

- Luminance-based training
- RGB-based training

Both showed similar quantitative performance.  
The RGB model was selected due to its ability to additionally learn color space expansion from Rec.709 (SDR) to Rec.2020 (HDR).

---

## Visual Comparison

Below is a qualitative comparison between the input SDR image, the reconstructed HDR output using the RGB-based model trained on the native dataset and the ground truth HDR image. 


<p align="center"> <!-- Example 1 --> <img src="assets/0004_SDR.jpg" width="250" alt="SDR Input"/> <img src="assets/0004_Model.jpg" width="250" alt="HDR Prediction"/> <img src="assets/0004_GT.jpg" width="250" alt="Ground Truth HDR"/> </p> <p align="center"> <!--

In most cases the model was able to achieve a visual alignment to the ground truth image.

---

## Quantitative Results

Evaluation was performed using the image quality metric HDR-VDP-3 which was specifically designed to assess HDR image quality.

| Dataset Type | Mean HDR-VDP |
|--------------|--------------|
| Native       | 7.98         |
| Reinhard TMO | 7.74         |
| Habel TMO    | 7.01         |
| Möbius TMO   | 7.89         |

The native dataset achieved the highest average score.

Due to the limited size of the test set, statistical significance could not always be established. Larger datasets may improve reliability.

---

## Future Work

Potential extensions include:

- Evaluation of additional tone mapping operators
- Integration of local operators
- Multi-frame / video reconstruction
- Combination with super-resolution
- Compression-aware reconstruction

---

## Structure

The processing pipeline used for training and testing the models can be found under "U-Net ITM Workflow". Additional image processing functions can be found under "U-Net ITM Workflow/utils". Due to copyrighting issues the used dataset cannot be shared in this repository.

