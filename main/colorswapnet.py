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
    #path_image = '/home/shaakira.gadiwan/project/data/ADEChallengeData2016/images/test/ADE_test_00000256.jpg'
    path_image = '/home/shaakira.gadiwan/project/data/st126_2.jpg'

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

if __name__ == '__main__':
    main()