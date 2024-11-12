import os
import sys
import torch
import shutil
import random
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def identity_collate(batch):
    return batch

def count_files(directory):
    """Count the number of files in a directory."""
    return len([file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))])


def move_and_rename_files(src_dir_images, src_dir_annotations, dest_dir_images, dest_dir_annotations, num_files, old_str, new_str):
    """Move a randomized selection of files from src_dir to dest_dir and rename the specified part of filename."""
    files = [f for f in os.listdir(src_dir_images) if os.path.isfile(os.path.join(src_dir_images, f))]
    random.shuffle(files)  # Randomize the order of files
    files_to_move = files[-num_files:]  # Select the required number of files
    
    for file_name in files_to_move:
        new_file_name = file_name.replace(old_str, new_str)

        src_path = os.path.join(src_dir_images, file_name)
        dest_path = os.path.join(dest_dir_images, new_file_name)
        shutil.move(src_path, dest_path)

        src_path = os.path.join(src_dir_annotations, file_name.replace("jpg", "png"))
        dest_path = os.path.join(dest_dir_annotations, new_file_name.replace("jpg", "png"))
        shutil.move(src_path, dest_path)


def delete_files(image_filename, images_train_dir, annotations_train_dir, images_val_dir, annotations_val_dir):
    """Delete corresponding image and annotation files based on the image filename."""
    annotation_filename_png = image_filename + ".png"
    image_filename_jpg = image_filename + ".jpg"
    paths = [
        (images_train_dir, annotations_train_dir),
        (images_val_dir, annotations_val_dir)
    ]

    for image_dir, annotation_dir in paths:
        image_path = os.path.join(image_dir, image_filename_jpg)
        annotation_path = os.path.join(annotation_dir, annotation_filename_png)

        if os.path.exists(image_path):
            os.remove(image_path)
        if os.path.exists(annotation_path):
            os.remove(annotation_path)


def create_data_loaders(
    root_dataset,
    image_training_root,
    annotation_training_root,
    image_validation_root,
    annotation_validation_root,
    image_test_root,
    annotation_test_root,
    batch_per_gpu,
    num_workers,
    collate_fn_train,
    collate_fn_val=lambda x: x,
    collate_fn_test=lambda x: x
):
    from datasets.dataset_seg import TrainDataset, ValDataset, TestDataset

    # Training dataset and loader
    dataset_train = TrainDataset(
        root_dataset=root_dataset,
        image_root=image_training_root,
        annotation_root=annotation_training_root,
        batch_per_gpu=batch_per_gpu
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=1,  # batch creation handled within TrainDataset
        shuffle=False,
        collate_fn=collate_fn_train,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    iterator_train = iter(loader_train)

    # Validation dataset and loader
    dataset_val = ValDataset(
        image_root=image_validation_root,
        annotation_root=annotation_validation_root
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,  # batch size 1 since images may vary in size
        shuffle=False,
        collate_fn=collate_fn_val,
        num_workers=num_workers,
        drop_last=True
    )
    iterator_val = iter(loader_val)

    # Test dataset and loader
    dataset_test = TestDataset(
        image_root=image_test_root,
        annotation_root=annotation_test_root
    )
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,  # batch size 1 since images may vary in size
        shuffle=False,
        collate_fn=collate_fn_test,
        num_workers=num_workers,
        drop_last=True
    )
    iterator_test = iter(loader_test)

    return {
        "train_loader": loader_train,
        "train_iterator": iterator_train,
        "val_loader": loader_val,
        "val_iterator": iterator_val,
        "test_loader": loader_test,
        "test_iterator": iterator_test
    }


def visualize_sample(image, mask, title, filename, height=None, width=None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Handle batch dimension in image (e.g., if shape is [B, C, H, W])
    if image.dim() == 4:
        image = image[0]  # Remove batch dimension
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert tensor to image format
    axes[0].set_title(f"{title} - Image")
    axes[0].axis('off')

    # Prepare mask for display
    mask = mask.cpu().numpy()
    if mask.ndim == 1 and height is not None and width is not None:
        mask = mask.reshape((height, width))  # Reshape if mask is flat
    elif mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]  # Remove extra dimension if mask is (1, H, W)

    axes[1].imshow(mask, cmap="gray")  # Display mask in grayscale
    axes[1].set_title(f"{title} - Mask")
    axes[1].axis('off')

    # Save figure
    plt.savefig(filename, bbox_inches='tight')


def imresize(im, size, interp='bilinear'):
    """
        Function for image resizing with given interpolation method
    """
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


def accuracy(preds, label): #TODO check difference with pixel_acc
    """
        Function for calculating pixel accuracy of an image
    """
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum

def pixel_acc(pred, label):
    """
        Function for calculating the pixel accuracy between the predicted image and labeled image
    """
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0)  # some labels are -1 and are ignored
    acc_sum = (valid * (preds == label)).sum()
    pixel_sum = valid.sum()
    return acc_sum / (pixel_sum + 1e-10)


def IOU(pred, labels):
    """
        Function for calculating IOU of an image
    """
    _, preds = torch.max(pred, dim=1)
    intersection = ((preds == 0) * (labels == 0)).sum()
    union = ((preds == 0) + (labels == 0)).sum() + 1e-15  # protection from division with 0
    return intersection / union


def visualize_wall(img, pred, class_to_display=0):
    """
        Function for visualizing wall prediction 
        (original image, segmentation mask and original image with the segmented wall)
    """
    img_green = img.copy()
    black_green = img.copy()
    img_green[pred == class_to_display] = [0, 255, 0]
    black_green[pred == class_to_display] = [0, 255, 0]
    black_green[pred != class_to_display] = [0, 0, 0]
    im_vis = np.concatenate((img, black_green, img_green), axis=1)
    PIL.Image.fromarray(im_vis).show()


def not_None_collate(x):
    return x