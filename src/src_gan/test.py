# src/src_gan/test.py

import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm

def test(generator, dataloader, device, output_dir):
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            real_images = batch['image'].to(device)
            segmented_images = batch['segmented_image'].to(device)
            target_colors = batch['target_color'].to(device)
            image_paths = batch['image_path']

            # Generate fake images
            fake_images = generator(real_images, segmented_images, target_colors)

            # Since the generator outputs the modified image, we can use fake_images directly
            combined_images = fake_images

            # Save images
            for j in range(real_images.size(0)):
                output_image = combined_images[j]
                image_name = os.path.basename(image_paths[j])
                save_path = os.path.join(output_dir, f"{i * dataloader.batch_size + j}_{image_name}")
                # Denormalize the output image if necessary
                output_image = (output_image + 1) / 2  # Assuming output is in [-1, 1]
                save_image(output_image, save_path)
