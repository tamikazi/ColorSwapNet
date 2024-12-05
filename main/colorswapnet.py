import os
import sys
# import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models_seg.models import SegmentationModule, build_encoder, build_decoder
from src.src_seg.eval import segment_image
from utils.constants import DEVICE

import torch
from torchvision import transforms
from PIL import Image
import argparse
from torchvision.utils import save_image

# Import the generator class
from models.models_cyclegan.generator import WallColorGenerator

def load_image(image_path, transform):
    """
    Load and preprocess the input image.

    Args:
        image_path (str): Path to the input image.
        transform (torchvision.transforms.Compose): Transformations to apply.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 3, 256, 256).
    """
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)  # Shape: (1, 3, H, W)
    return image

def process_mask(segmentation_mask, device):
    """
    Process the segmentation mask to match GAN input requirements.

    Args:
        segmentation_mask (np.ndarray): Numpy array of the segmentation mask.
        device (torch.device): Device to place the tensor.

    Returns:
        torch.Tensor: Processed mask tensor of shape (1, 1, 256, 256).
    """
    # Convert numpy array to PIL Image
    # Assuming mask is binary (0 and 1)
    mask_pil = Image.fromarray((segmentation_mask * 255).astype(np.uint8))
    
    # Define mask transformations
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),  # Converts to [0.0,1.0]
    ])
    
    mask = mask_transform(mask_pil).unsqueeze(0).to(device)  # Shape: (1, 1, 256, 256)
    mask = (mask == 0.0).float()  # Adjust threshold as needed
    return mask

def set_custom_color(target_color, device):
    """
    Create a tensor representing the target color.

    Args:
        target_color (list or tuple): RGB values in [0, 1].
        device (torch.device): Device to place the tensor.

    Returns:
        torch.Tensor: Target color tensor of shape (1, 3, 256, 256).
    """
    # Ensure target_color has three components
    if len(target_color) != 3:
        raise ValueError("Custom color must have three components (R, G, B).")
    
    # Create a tensor of shape (1, 3, 256, 256) with the target color
    target_color_tensor = torch.tensor(target_color, dtype=torch.float32).view(1, 3).to(device)
    
    # Normalize to [-1, 1] as per GAN's expectation
    target_color_tensor = (target_color_tensor * 2) - 1
    return target_color_tensor

def save_generated_image(fake_image, output_path):
    """
    Save the generated image to disk.

    Args:
        fake_image (torch.Tensor): Generated image tensor of shape (1, 3, 256, 256).
        output_path (str): Path to save the generated image.
    """
    # Rescale from [-1, 1] to [0, 1]
    fake_image = (fake_image + 1) / 2
    fake_image = fake_image.clamp(0, 1)
    
    # Save image
    save_image(fake_image, output_path)
    print(f"Generated image saved to {output_path}")

def load_segmentation_model(encoder_path, decoder_path):
    """
    Load the pre-trained segmentation model.

    Args:
        encoder_path (str): Path to the encoder weights.
        decoder_path (str): Path to the decoder weights.

    Returns:
        SegmentationModule: Loaded segmentation model.
    """
    # Build encoder and decoder
    net_encoder = build_encoder(encoder_path)
    net_decoder = build_decoder(decoder_path)
    
    # Create segmentation module
    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    segmentation_module = segmentation_module.to(DEVICE).eval()
    
    return segmentation_module

def load_generator(generator_checkpoint):
    """
    Load the pre-trained CycleGAN generator.

    Args:
        generator_checkpoint (str): Path to the generator checkpoint.

    Returns:
        WallColorGenerator: Loaded generator model.
    """
    generator = WallColorGenerator().to(DEVICE)
    generator.load_state_dict(torch.load(generator_checkpoint, map_location=DEVICE))
    generator.eval()
    return generator

def create_target_color_image(target_color, size=(256, 256)):
    """
    Create an image filled with the target color.

    Args:
        target_color (list or tuple): RGB values in [0, 1].
        size (tuple): Size of the image (width, height).

    Returns:
        PIL.Image: Image filled with the target color.
    """
    # Create a numpy array with the target color
    color_array = np.ones((size[1], size[0], 3), dtype=np.float32)
    color_array[:, :] = target_color  # Set all pixels to the target color
    
    # Convert to PIL Image
    color_image = Image.fromarray((color_array * 255).astype(np.uint8))
    return color_image

def display_results(original_image_path, segmentation_mask, target_color_image, generated_image_path):
    """
    Display the original image, segmentation mask, target color, and generated image in a 2x2 grid.

    Args:
        original_image_path (str): Path to the original input image.
        segmentation_mask (np.ndarray): Numpy array of the segmentation mask.
        target_color_image (PIL.Image): Image filled with the target color.
        generated_image_path (str): Path to the generated image.
    """
    # Load original image
    original_image = Image.open(original_image_path).convert('RGB')
    
    # Convert segmentation mask to displayable format
    # If mask is binary, display as grayscale
    if len(segmentation_mask.shape) == 2:
        mask_display = Image.fromarray((segmentation_mask * 255).astype(np.uint8))
    else:
        # If mask has multiple channels, take the first channel
        mask_display = Image.fromarray((segmentation_mask[:, :, 0] * 255).astype(np.uint8))
    
    # Load generated image
    generated_image = Image.open(generated_image_path).convert('RGB')
    
    # Create a 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Original Image
    axs[0, 0].imshow(original_image)
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    # Segmentation Mask
    axs[0, 1].imshow(mask_display, cmap='gray')
    axs[0, 1].set_title('Segmentation Mask')
    axs[0, 1].axis('off')
    
    # Target Color
    axs[1, 0].imshow(target_color_image)
    axs[1, 0].set_title('Target Color')
    axs[1, 0].axis('off')
    
    # Generated Image
    axs[1, 1].imshow(generated_image)
    axs[1, 1].set_title('Generated Image with Changed Wall Color')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():

    # Path to the input image
    #path_image = '/home/shaakira.gadiwan/project/data/ADEChallengeData2016/images/test/ADE_test_00000256.jpg'
    # path_image = '/home/shaakira.gadiwan/project/data/st126_2.jpg'
    path_image = 'C:/Users/tahmi/Downloads/st126_1.PNG'


    # Model weights (encoder and decoder)
    # weights_encoder = '/home/shaakira.gadiwan/project/best_models/best_encoder_epoch_18.pth'
    weights_encoder = 'C:/Users/tahmi/Downloads/best_encoder_epoch_18.pth'
    # weights_decoder = '/home/shaakira.gadiwan/project/best_models/best_decoder_epoch_18.pth'
    weights_decoder = 'C:/Users/tahmi/Downloads/best_decoder_epoch_18.pth'

    generator_path = 'C:/Users/tahmi/Documents/MENG2023/ENEL645/ColorSwapNet/main/best_generator.pt'

    # Custom target color (RGB values in [0, 1])
    custom_color = [0.682, 0.145, 1.0]  # Example: Green

    # Output path for the generated image
    output_image_path = 'C:/Users/tahmi/Downloads/generated_output.png'

    image_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
    ])

    input_image = load_image(path_image, image_transform)

    segmentation_module = load_segmentation_model(weights_encoder, weights_decoder)

    segmentation_mask = segment_image(segmentation_module, path_image, disp_image=False)


    mask = process_mask(segmentation_mask, DEVICE)

    # ============================
    # Set Custom Target Color
    # ============================
    target_color_tensor = set_custom_color(custom_color, DEVICE)

    # ============================
    # Load CycleGAN Generator
    # ============================
    generator = load_generator(generator_path)

    # ============================
    # Generate Colorized Image
    # ============================
    with torch.no_grad():
        # Generate the fake image
        fake_image = generator(input_image, mask, target_color_tensor)

    # ============================
    # Save the Generated Image
    # ============================
    save_generated_image(fake_image, output_image_path)

    # ============================
    # Create Target Color Image for Display
    # ============================
    target_color_image = create_target_color_image(custom_color, size=(256, 256))

    # ============================
    # Display Results in 2x2 Grid
    # ============================
    display_results(path_image, segmentation_mask, target_color_image, output_image_path)

if __name__ == '__main__':
    main()