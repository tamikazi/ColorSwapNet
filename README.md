# ColorSwapNet
---

## Directory and File Structure

```
- data/
- datasets/                # Contains the dataset classes
- main/                    # Main files to run
    - data_process.py      # Get the indoor images from the original dataset
    - wall_seg.py          # Main file to run to train/eval the seg
    - wall_gan.py          # Main file to run to train/eval the gan
- models/                  # Contains the segmentation and GAN algorithms
- checkpoints/           # Contains the best model paths/checkpoints, useful when training takes over 24 hours
  - best_seg/            # Best segmentation model
    - best_encoder_and_decoder_paths/
  - best_gan/            # Best GAN model
- model_weights/           # Contains the model weights (loaded from https://drive.google.com/drive/folders/1xh-MBuALwvNNFnLe-eofZU_wn8y3ZxJg)
    - resnet50_or_101/       # Probably will use ResNet50 or ResNet101
- src/                     # Contains train/val + test algorithms
    - src_seg/               # Segmentation source files
      - train.py
      - eval.py
      - test.py
    - src_gan/               # GAN source files
- utils/                   # Contains miscellaneous algorithms
    - constants.py           # Configuration file, holds all the global variables
    - utils.py               # Contains utility functions like custom image resizing, etc.
```

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


---

## Generative Adversarial Network