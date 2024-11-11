# generate_masks.py

import os
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

# Import your segmentation model components
from models.models_seg.models import build_encoder, build_decoder, SegmentationModule
from utils.constants import DEVICE

def load_segmentation_model(encoder_path, decoder_path):
    net_encoder = build_encoder()
    net_decoder = build_decoder()
    crit = torch.nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    segmentation_module.to(DEVICE)
    segmentation_module.eval()

    # Load weights
    net_encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
    net_decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE))

    return segmentation_module

def generate_masks(image_root, mask_root, segmentation_module, transform):
    os.makedirs(mask_root, exist_ok=True)

    image_paths = sorted([os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')])

    for image_path in tqdm(image_paths):
        image = Image.open(image_path).convert('RGB')
        image_transformed = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = segmentation_module({'img_data': image_transformed})
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Convert pred to a binary mask (0 for background, 1 for wall)
        mask = (pred == 1).astype('uint8') * 255  # Assuming class 1 is wall

        # Save mask
        mask_image = Image.fromarray(mask)
        mask_filename = os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(mask_root, mask_filename)
        mask_image.save(mask_path)

if __name__ == '__main__':
    # Define paths
    IMAGE_ROOT = 'path_to_your_images'  # Replace with your image directory
    MASK_ROOT = 'path_to_save_masks'    # Replace with directory to save generated masks
    ENCODER_WEIGHTS = 'path_to_best_encoder_weights.pth'
    DECODER_WEIGHTS = 'path_to_best_decoder_weights.pth'

    # Load segmentation model
    segmentation_module = load_segmentation_model(ENCODER_WEIGHTS, DECODER_WEIGHTS)

    # Define image transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # Add normalization if needed
    ])

    # Generate and save masks
    generate_masks(IMAGE_ROOT, MASK_ROOT, segmentation_module, transform)
