# export_model.py

import torch
import os
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Suppress INFO messages
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.dataset_cyclegan import CycleGANDataset
from models.models_cyclegan.generator import WallColorGenerator
from models.models_cyclegan.discriminator import WallColorDiscriminator
from src.src_cyclegan.train import train_one_epoch, PerceptualLoss
from src.src_cyclegan.train import save_sample_images  # Make sure to import if in separate file

def main():
    # Device configuration
    device = torch.device('cpu')  # Change to 'cuda' if using GPU

    # Initialize the generator
    generator = WallColorGenerator().to(device)
    generator.eval()  # Set to evaluation mode

    # Load model weights (if available)
    checkpoint_path = 'path_to_your_checkpoint.pth'  # Update this path
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
    else:
        print("No checkpoint found. Using randomly initialized weights.")

    # Create dummy inputs
    batch_size = 1
    input_image = torch.randn(batch_size, 3, 256, 256).to(device)       # [1, 3, 256, 256]
    wall_mask = torch.randn(batch_size, 1, 256, 256).to(device)         # [1, 1, 256, 256]
    target_color = torch.randn(batch_size, 3).to(device)                # [1, 3]

    # Export the model to ONNX
    onnx_filename = "generator.onnx"
    torch.onnx.export(
        generator,                                      # Model to export
        (input_image, wall_mask, target_color),         # Model input (tuple)
        onnx_filename,                                  # Output file
        export_params=True,                             # Store the trained parameter weights inside the model file
        opset_version=11,                               # The ONNX version to export the model to
        do_constant_folding=True,                       # Whether to execute constant folding for optimization
        input_names=['input_image', 'wall_mask', 'target_color'],   # Model input names
        output_names=['output'],                        # Model output name
        dynamic_axes={
            'input_image': {0: 'batch_size'},
            'wall_mask': {0: 'batch_size'},
            'target_color': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },                                              # Variable batch size
        verbose=True                                     # Print the model's computational graph
    )

    print(f"Model has been converted to {onnx_filename}")

if __name__ == "__main__":
    main()
