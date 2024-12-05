# ColorSwapNet

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)

## Project Overview

The **CycleGAN Wall Colorization Project** leverages the power of CycleGAN and advanced image segmentation techniques to automatically change the color of walls in images. By accurately identifying wall regions through segmentation masks, the system enables realistic and seamless color transformations, making it an invaluable tool for interior design, virtual staging, and creative applications.

![Before and After](https://github.com/tamikazi/ColorSwapNet/blob/main/results/GAN_images/input_epoch_101_batch_200.png)

## Features

- **Accurate Wall Segmentation**: Utilizes a robust segmentation module based on ResNet architectures to precisely identify wall regions in images.
- **Seamless Color Transformation**: Implements a CycleGAN-based generator to alter wall colors while preserving the natural look and feel of the environment.
- **Flexible Target Colors**: Allows users to specify custom target colors for walls, facilitating a wide range of design possibilities.
- **Early Stopping and Checkpointing**: Employs early stopping mechanisms and checkpointing to optimize training efficiency and model performance.
- **Visualization Tools**: Provides utilities to visualize segmentation masks, generated images, and overlayed results for easy verification and analysis.

## Installation

### Clone the Repository

```bash
git clone https://github.com/tamikazi/ColorSwapNet.git
cd ColorSwapNet
```

## Usage

### Data Preparation

1. **Dataset Structure**:

   The dataset should be organized as follows:

   ```
   ADEChallengeData2016/
   ├── images/
   │   ├── training/
   │   ├── validation/
   │   └── test/
   ├── annotations/
   │   ├── training/
   │   ├── validation/
   │   └── test/
   └── sceneCategories.txt
   ```

## Project Structure

```
cyclegan-wall-colorization/
├── datasets/
│   └── dataset_cyclegan.py         # Dataset classes for CycleGAN
    └── dataset_seg.py              # Dataset classes for Segmentation
├── models/
│   ├── models_cyclegan/
│   │   ├── generator.py            # CycleGAN Generator
│   │   └── discriminator.py        # CycleGAN Discriminator
│   └── models_seg/
│       └── models.py               # Segmentation models (ResNet-based)
├── src/
│   ├── src_cyclegan/
│   │   ├── train.py                # Training script for CycleGAN
│   │   └── utils.py                # Utility functions for CycleGAN
│   └── src_seg/
│       ├── eval.py                  # Evaluation script for Segmentation
│       └── train.py                 # Training script for Segmentation
├── main/
│   └── process_data.py              # Data preprocessing script
│   └── cyclegan_train.py            # Main training for CycleGAN
│   └── wall_seg.py                  # Main training for Segmentation
├── utils/
│   ├── constants.py                 # Constant definitions
│   └── utils.py                     # General utility functions
├── main/
│   ├── cyclegan_train.py            # Main training script for CycleGAN
│   └── seg_train.py                 # Main training script for Segmentation
├── requirements.txt                 # Python dependencies
├── README.md                        # Project README
```

## Dependencies

The project relies on the following key libraries:

- **Python 3.8+**
- **PyTorch**: For building and training neural networks.
- **Torchvision**: Image processing utilities.
- **PIL (Pillow)**: Image handling.
- **NumPy**: Numerical operations.
- **Matplotlib**: Plotting and visualization.
- **Scikit-learn**: Evaluation metrics.
- **TQDM**: Progress bars.
- **TensorBoard**: Monitoring training metrics.
- **TorchMetrics**: Evaluation metrics like SSIM and FID.

*Ensure CUDA is properly configured for GPU acceleration.*

Notes:

There's two separate datasets because the input to the gan might be slightly different in terms of image size, which would change how the mask was downsampled. Same for the resnet model, not exactly sure what would be different so separated them for now.

Make sure to ignore index -1 at some point, for example in the loss function:
```
crit = nn.NLLLoss(ignore_index=-1)
```
Index -1 are don't cares, 0 is wall, 1 is everything else.

---

## Image Segmentation

Adapted from the papers and code by Mihailo Bjekic et al.:

- [Getting Started with Wall Segmentation](https://www.researchgate.net/publication/363059238_Getting_Started_with_Wall_Segmentation)
- [Wall segmentation in 2D images using convolutional neural networks](https://www.researchgate.net/publication/373861585_Wall_segmentation_in_2D_images_using_convolutional_neural_networks)
- [Wall segmentation code](https://github.com/bjekic/WallSegmentation/tree/main)
