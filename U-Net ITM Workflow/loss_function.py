import torch
import torch.nn as nn
import pytorch_ssim 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def choose_loss_function(pred, target, loss_type):
    if loss_type == 1:
        return mse_loss(pred, target)
    elif loss_type == 2:
        return ssim_loss(pred, target)
    elif loss_type == 3:
        return log_weighted_mse(pred, target)         # log to be created
    if loss_type == 4:
        return mae_loss(pred, target)

def ssim_loss(pred, target):
    # Erstelle SSIM-Modul
    ssim_module = pytorch_ssim.SSIM(window_size=11, size_average=True, device=device).to(device)
    
    # Konvertiere pred und target auf das richtige Gerät
    pred = pred.to(device)
    target = target.to(device)

    # Überprüfen, ob es sich um ein Bild mit nur einem Kanal (Luminanz) handelt
    if pred.shape[1] == 1:  # Wenn es nur einen Kanal gibt (Luminanzbild)
        # SSIM für den Luminanzkanal berechnen
        loss = 1 - ssim_module(pred, target)
    elif pred.shape[1] == 3:  # Wenn es drei Kanäle gibt (RGB-Bild)
        # SSIM für jeden RGB-Kanal berechnen und dann den Mittelwert bilden
        ssim_r = ssim_module(pred[:, [0], :, :], target[:, [0], :, :])  # R-Kanal
        ssim_g = ssim_module(pred[:, [1], :, :], target[:, [1], :, :])  # G-Kanal
        ssim_b = ssim_module(pred[:, [2], :, :], target[:, [2], :, :])  # B-Kanal
        
        # Den Durchschnitt der SSIM-Werte berechnen
        ssim_mean = (ssim_r + ssim_g + ssim_b) / 3.0
        
        # SSIM Loss (1 - Durchschnitt der SSIM-Werte)
        loss = 1 - ssim_mean
    else:
        raise ValueError("Bild muss entweder 1 oder 3 Kanäle haben (Luminanz oder RGB).")

    return loss


def mse_loss(pred, target):
    pred = pred.to(device)
    target = target.to(device)
    
    mse = nn.functional.mse_loss(pred, target)

    loss = mse
    return loss

def mae_loss(pred, target):
    pred = pred.to(device)
    target = target.to(device)
    
    mae = nn.functional.l1_loss(pred, target)

    loss = mae

    # feature_extractor = LuminanceFeatureExtractor().to(device)
    # perceptual_loss_fn = PerceptualLoss(feature_extractor).to(device)
    # perceptual_loss = perceptual_loss_fn(pred, target)

    return loss


def log_weighted_mse(output, target, epsilon=1e-6):
    # Berechne den MSE-Fehler
    mse = (output - target) ** 2

    # Berechne den logaritmischen Gewichts-Faktor, wobei epsilon eine kleine Zahl ist, um Division durch Null zu verhindern
    log_weight = torch.log(target + epsilon)

    # Addiere 1 zu log_weight, um zu verhindern, dass der Gewichtungsfaktor negativ wird
    log_weight = 1 / (log_weight + 1)

    # Wende das Gewicht auf den MSE-Fehler an
    weighted_mse = log_weight * mse

    # Berechne den Mittelwert über alle Pixel
    return torch.mean(weighted_mse)


# class LuminanceFeatureExtractor(nn.Module):
#     def __init__(self, layers=None):
#         super(LuminanceFeatureExtractor, self).__init__()
#         # Verwende ein vortrainiertes VGG-Modell
#         vgg = models.vgg16(pretrained=True).features
#         self.layers = layers if layers is not None else [0, 5, 10, 17]  # VGG Layer Indices
#         self.vgg_layers = nn.ModuleList([vgg[i] for i in self.layers])

#         # Anpassung des ersten Convs für Einkanalbilder (Luminanz)
#         self.vgg_layers[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)  # Einkanal-Input

#         # Gefriert die VGG-Gewichte (Optional)
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         features = []
#         for i, layer in enumerate(self.vgg_layers):
#             x = layer(x)
#             features.append(x)
#         return features
    
# class PerceptualLoss(nn.Module):
#     def __init__(self, feature_extractor):
#         super(PerceptualLoss, self).__init__()
#         self.feature_extractor = feature_extractor
#         self.mse_loss = nn.MSELoss()

#     def forward(self, pred, target):
#         # Extrahiere Features für Vorhersage und Ziel
#         pred_features = self.feature_extractor(pred)
#         target_features = self.feature_extractor(target)

#         # Berechne den Perceptual Loss als MSE über die Feature-Antworten
#         perceptual_loss = 0
#         for pred_f, target_f in zip(pred_features, target_features):
#             perceptual_loss += self.mse_loss(pred_f, target_f)
        
#         return perceptual_loss


# Log Loss

def weight_luminance(luminance, epsilon=1e-6):

    # Avoid division by zero
    return 1.0 / (luminance + epsilon)


def weighted_loss(pred, target):
    pred = pred.to(device)
    target = target.to(device)

    # Calculate luminance for both predicted and target images
    pred_luminance = pred[:, 0, :, :]
    target_luminance = target[:, 0, :, :]

    # Calculate weighting based on the luminance
    weight = weight_luminance(target_luminance)

    # Compute the MSE and L1 loss
    mse = nn.functional.mse_loss(pred, target, reduction='none')  # Compute element-wise loss
    l1_loss = nn.functional.l1_loss(pred, target, reduction='none')

    # Apply luminance-based weighting
    weighted_mse = weight * mse
    weighted_l1_loss = weight * l1_loss

    # Return the final weighted loss, averaged over all pixels
    loss = 0.5 * weighted_mse.mean() + 0.5 * weighted_l1_loss.mean()

    return loss