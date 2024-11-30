import torch
import sys
import os
import torchvision.transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from sklearn.metrics import jaccard_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.constants import IMAGENET_MEAN, IMAGENET_STD
from utils.utils import IOU, visualize_wall, accuracy, calculate_boundary_iou, calculate_hausdorff_distance
from utils.constants import DEVICE

def validation_step(segmentation_module, loader, writer, epoch):
    """
    Function for evaluating the segmentation module on the validation dataset
    """
    segmentation_module.eval()
    segmentation_module.to(DEVICE)
    
    total_acc = 0
    total_IOU = 0
    total_jaccard = 0
    total_boundary_IOU = 0
    total_hausdorff = 0
    counter = 0
    
    for batch_data in tqdm(loader):
        batch_data = batch_data[0]

        seg_label = np.array(batch_data['seg_label'])
        seg_size = (seg_label.shape[0], seg_label.shape[1])

        with torch.no_grad():
            scores = segmentation_module(batch_data, seg_size=seg_size)

        _, pred = torch.max(scores, dim=1)
        pred = pred.cpu()[0].numpy()

        # Calculate Accuracy and IOU
        acc, _ = accuracy(pred, seg_label)
        IOU_curr = IOU(scores.cpu(), seg_label)
        
        # Calculate Jaccard Index
        jaccard = jaccard_score(seg_label.flatten(), pred.flatten(), average='macro')
        
        # Calculate Boundary IoU
        boundary_iou = calculate_boundary_iou(pred, seg_label)
        
        # Calculate Hausdorff Distance
        hausdorff = calculate_hausdorff_distance(pred, seg_label)

        # Accumulate metrics
        total_acc += acc
        total_IOU += IOU_curr
        total_jaccard += jaccard
        total_boundary_IOU += boundary_iou
        total_hausdorff += hausdorff
        counter += 1

    # Average metrics
    average_acc = total_acc / counter
    average_IOU = total_IOU / counter
    average_jaccard = total_jaccard / counter
    average_boundary_IOU = total_boundary_IOU / counter
    average_hausdorff = total_hausdorff / counter

    # Log to TensorBoard
    writer.add_scalar('Validation set: accuracy', average_acc, epoch)
    writer.add_scalar('Validation set: IOU', average_IOU, epoch)
    writer.add_scalar('Validation set: Jaccard Index', average_jaccard, epoch)
    writer.add_scalar('Validation set: Boundary IoU', average_boundary_IOU, epoch)
    writer.add_scalar('Validation set: Hausdorff Distance', average_hausdorff, epoch)

    # Print averaged metrics
    print(f"Validation Results at Epoch {epoch}:")
    print(f"  Average Accuracy: {average_acc:.4f}")
    print(f"  Average IoU: {average_IOU:.4f}")
    print(f"  Average Jaccard Index: {average_jaccard:.4f}")
    print(f"  Average Boundary IoU: {average_boundary_IOU:.4f}")
    print(f"  Average Hausdorff Distance: {average_hausdorff:.4f}")
    
    return average_acc, average_IOU, average_jaccard, average_boundary_IOU, average_hausdorff

def segment_image(segmentation_module, img, disp_image=True):
    """
        Function for segmenting wall in the input image. The input can be path to image, or a loaded image
    """
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    if isinstance(img, str):
        img = Image.open(img)
    
    img_original = np.array(img)
    img_data = pil_to_tensor(img)
    singleton_batch = {'img_data': img_data[None].to(DEVICE)}
    seg_size = img_original.shape[:2]

    with torch.no_grad():
        scores = segmentation_module(singleton_batch, seg_size=seg_size)

    _, pred = torch.max(scores, dim=1)
    pred = pred.cpu()[0].numpy()

    if disp_image:
        visualize_wall(img_original, pred)

    return pred