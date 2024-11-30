import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models_seg.models import SegmentationModule, build_encoder, build_decoder
from src.src_seg.eval import segment_image
from utils.constants import DEVICE

def main():

    # Path to the input image
    path_image = '/home/shaakira.gadiwan/project/data/ADEChallengeData2016/images/test/ADE_test_00000256.jpg'

    # Model weights (encoder and decoder)
    weights_encoder = '/home/shaakira.gadiwan/project/best_models/best_encoder_epoch_18.pth'
    weights_decoder = '/home/shaakira.gadiwan/project/best_models/best_decoder_epoch_18.pth'

    # Build and load the segmentation model
    net_encoder = build_encoder(weights_encoder)
    net_decoder = build_decoder(weights_decoder)
    segmentation_module = SegmentationModule(net_encoder, net_decoder)
    segmentation_module = segmentation_module.to(DEVICE).eval()

    # Predict segmentation mask
    segmentation_mask = segment_image(segmentation_module, path_image)

    cv2.imwrite('segmentation_mask.png', (segmentation_mask * 255).astype('uint8'))  # Scale to 0-255 and save as grayscale

    # Normalize and convert the mask for edge detection
    segmentation_mask = (segmentation_mask / segmentation_mask.max() * 255).astype(np.uint8)

    # Load the original image
    original_image = cv2.imread(path_image)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Apply Canny edge detection to find boundaries in the segmentation mask
    edges = cv2.Canny(segmentation_mask, 10, 150)  # Adjust thresholds for better results

    # Make the edges thicker using dilation
    kernel = np.ones((3, 3), np.uint8)  # Define a kernel for dilation
    thick_edges = cv2.dilate(edges, kernel, iterations=1)  # Increase iterations for thicker edges


    # Overlay boundaries on the original image
    overlay_image = original_image.copy()
    overlay_image[thick_edges > 0] = [255, 0, 0]  # Draw red boundaries

    # Save the overlayed image for verification
    cv2.imwrite('overlayed_boundaries.png', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))


if __name__ == '__main__':
    main()