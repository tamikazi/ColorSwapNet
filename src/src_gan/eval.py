import torch
import os
from torchvision.utils import save_image
from tqdm import tqdm

def evaluate(generator, dataloader, epoch, device, output_dir):
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            real_images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            target_colors = batch['target_color'].to(device)
            image_paths = batch['image_path']

            # Generate fake images
            fake_images = generator(real_images, masks, target_colors)

            # Since the generator already combines the output, we can use fake_images directly
            combined_images = fake_images

            # Save images
            for j in range(real_images.size(0)):
                output_image = combined_images[j]
                image_name = os.path.basename(image_paths[j])
                save_path = os.path.join(output_dir, f"epoch_{epoch}_{i * dataloader.batch_size + j}_{image_name}")
                save_image(output_image, save_path)
