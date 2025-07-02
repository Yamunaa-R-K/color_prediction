import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
from torchvision.models.segmentation import deeplabv3_resnet50

# ------------------ MODEL CLASS ------------------ #

class DeepLabV3Plus(nn.Module):
    def __init__(self):
        super(DeepLabV3Plus, self).__init__()
        self.deeplab = deeplabv3_resnet50(weights="DEFAULT")
        self.deeplab.classifier[4] = nn.Conv2d(256, 3, 1)
        self._init_weights(self.deeplab.classifier[4])

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        output = self.deeplab(x)['out']
        return torch.sigmoid(output)

# ------------------ COLOR PROCESSING FUNCTIONS ------------------ #

def normalize_image(img):
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    mean_rgb = img.mean(axis=(0, 1))
    if mean_rgb.any():
        scale = mean_rgb.mean() / mean_rgb
        img = np.clip(img * scale, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_mean = lab[:, :, 0].mean()
    clip_limit = 3.0 if l_mean > 50 else 5.0
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    normalized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    mean_intensity = gray.mean() / 255.0
    gamma = 1.0 / (1.2 if mean_intensity > 0.5 else 0.8)
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    normalized = cv2.LUT(normalized, table)

    return normalized

def get_dominant_colors(image, cob_masks, min_area=50):
    from scipy.ndimage import center_of_mass, label

    image = normalize_image(image)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    colors = []

    for mask in cob_masks:
        labeled, num_features = label(mask)
        if num_features == 0 or mask.sum() < min_area:
            continue

        kernel = np.ones((3, 3), np.uint8)
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        if mask_eroded.sum() < min_area:
            continue

        l_channel = lab_img[:, :, 0]
        valid_pixels = (mask_eroded > 0) & (l_channel > 30)
        if not valid_pixels.any():
            continue

        center_y, center_x = center_of_mass(mask_eroded)
        y, x = np.ogrid[:256, :256]
        dist = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)
        max_dist = dist[valid_pixels].max() if valid_pixels.any() else 1
        weights = np.clip(1 - dist / (max_dist + 1e-6), 0, 1) * valid_pixels

        roi = lab_img[valid_pixels]
        roi_weights = weights[valid_pixels]
        if roi.size == 0 or len(roi) < min_area:
            continue

        pixels = roi.reshape(-1, 3).astype(np.float32)
        weights_flat = roi_weights.reshape(-1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 2, weights_flat, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        cluster_weights = [weights_flat[labels.ravel() == i].sum() for i in range(2)]
        dominant_idx = np.argmax(cluster_weights)
        dominant_color = centers[dominant_idx].astype(np.int32)

        colors.append((dominant_color, mask))

    return colors

def find_closest_color(target_lab, chart_df):
    lab_vals = chart_df[['L', 'a', 'b']].values.astype(np.float32)
    weights = np.array([2.0, 1.0, 1.0])
    dists = np.sqrt(((lab_vals - target_lab) ** 2 * weights).sum(axis=1))
    idx = dists.argmin()
    return {
        'RHS': chart_df.iloc[idx]['RHS'],
        'Color name': chart_df.iloc[idx]['Color name'],
        'L': float(chart_df.iloc[idx]['L']),
        'a': float(chart_df.iloc[idx]['a']),
        'b': float(chart_df.iloc[idx]['b'])
    }

# ------------------ MODEL & COLOR CHART LOADING ------------------ #

def load_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepLabV3Plus().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def load_color_chart(chart_path):
    return pd.read_csv(chart_path)

# ------------------ PREDICTION FUNCTION ------------------ #

def predict_colors(image_path, model, color_chart, device):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    input_img = cv2.resize(image_rgb, (256, 256)) / 255.0
    input_img = (input_img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    input_tensor = torch.tensor(input_img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)[0].cpu().numpy().transpose(1, 2, 0)
        pred = (pred * 255).astype(np.uint8)

    binary_mask = (pred.mean(axis=2) > 0).astype(np.uint8)

    # Label connected components
    from scipy.ndimage import label
    labeled, num_features = label(binary_mask)

    cob_masks = []
    for i in range(1, num_features + 1):
        region_mask = (labeled == i).astype(np.uint8)
        cob_masks.append(region_mask)

    colors = get_dominant_colors(image_rgb, cob_masks)
    results = []

    for lab, mask in colors:
        matched_color = find_closest_color(lab, color_chart)
        if matched_color:
            results.append(matched_color)

    return results

