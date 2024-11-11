import os
import sys
import numpy as np
from PIL import Image
from glob import glob

import torch
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.utils import imresize
from utils.constants import IMAGENET_MEAN, \
                            IMAGENET_STD, \
                            IMG_SIZES_GAN, \
                            IMG_MAX_SIZE_GAN, \
                            PADDING_GAN, \
                            LIST_SCENES, \
                            SEGM_DOWNSAMPLING_RATE_GAN

def create_scene_dict(path, list_scenes):
    # Initialize an empty dictionary to store scene mappings.
    dict_scene = {}
    
    # Open the file at the specified path in read mode.
    file = open(path, 'r')
    
    # Initialize counters for validation and training scenes.
    counter_val = 0
    counter_train = 0
    
    # Iterate over each line in the file.
    for line in file:
        # Split the line by spaces to separate elements.
        temp = line.split(' ')
        
        # Extract the scene name by removing the newline character from the second element.
        scene = temp[1][:-1]
        
        # Map the first element as the key and the scene name as the value in the dictionary.
        dict_scene[temp[0]] = scene
        
        # Increment validation counter if the scene is in the list and starts with 'ADE_val'.
        if scene in list_scenes and temp[0].startswith('ADE_val'):
            counter_val += 1
        
        # Increment training counter if the scene is in the list and starts with 'ADE_train'.
        if scene in list_scenes and temp[0].startswith('ADE_train'):
            counter_train += 1
    
    # Return the dictionary and counters for validation and training scenes.
    return dict_scene, counter_val, counter_train

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, annotation_root, split='training', **kwargs):
        # Initialize root paths for images and annotations.
        self.image_root = image_root
        self.annotation_root = annotation_root
        
        # Define dataset parameters.
        self.imgSizes = IMG_SIZES_GAN
        self.imgMaxSize = IMG_MAX_SIZE_GAN
        self.padding_constant = PADDING_GAN
        self.list_scenes = LIST_SCENES

        # Parse and store the list of sample files.
        self.parse_input_list()

        # Define the normalization transformation using ImageNet mean and std.
        self.normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def parse_input_list(self):
        """
        Parse input list by listing files in the given image and annotation folders.
        """
        # Sort image and annotation paths to maintain consistent ordering.
        image_paths = sorted(glob(os.path.join(self.image_root, '*.jpg')))  # Adjust extension if needed.
        annotation_paths = sorted(glob(os.path.join(self.annotation_root, '*.png')))  # Adjust extension if needed.

        # Create a list to store samples with matching images and annotations.
        self.list_sample = []
        for img_path, ann_path in zip(image_paths, annotation_paths):
            # Load image dimensions and store paths and dimensions in sample dictionary.
            sample = {
                'fpath_img': img_path,
                'fpath_segm': ann_path,
                'height': Image.open(img_path).height,
                'width': Image.open(img_path).width
            }
            # Append sample dictionary to the list of samples.
            self.list_sample.append(sample)

    def img_transform(self, img):
        """
        Transform image by reordering channels and normalizing.
        """
        # Convert image from 0-255 range to 0-1.
        img = np.float32(np.array(img)) / 255.
        
        # Reorder dimensions to (channels, height, width).
        img = img.transpose((2, 0, 1))
        
        # Apply normalization and return transformed image as a tensor.
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        """
        Transform segmentation mask to tensor with values in the range [-1, 149].
        """
        # Convert segmentation mask from numpy array to long tensor.
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    def round2nearest_multiple(self, x, p):
        """ 
        Round x to the nearest multiple of p where the result is >= x.
        """
        # Calculate the nearest multiple of p greater than or equal to x.
        # Used for properly downsampling the mask.
        return ((x - 1) // p + 1) * p

class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, image_root, annotation_root, batch_per_gpu=1, **kwargs):
        # Initialize the base dataset class with image and annotation paths and additional settings.
        super(TrainDataset, self).__init__(image_root=image_root, annotation_root=annotation_root, **kwargs)

        # Path to the file that contains scene categories.
        scene_categories = root_dataset + '/sceneCategories.txt'

        # Initialize class attributes.
        self.root_dataset = root_dataset
        self.batch_per_gpu = batch_per_gpu  # Number of samples per GPU.
        self.num_sample = len(self.list_sample)  # Total number of samples.

        # Additional settings for segmentation and tracking batch state.
        self.segm_downsampling_rate = SEGM_DOWNSAMPLING_RATE_GAN
        self.cur_idx = 0
        self.if_shuffled = False  # To track if the samples have been shuffled.

        # Prepare lists to classify images based on aspect ratios for efficient batch grouping.
        self.batch_record_list = [[], []]

        # Load scene information and count the number of training examples for scenes of interest.
        self.scene_dict, _, num_ex_train = create_scene_dict(scene_categories, self.list_scenes)

        self.target_colors = [
            [1.0, 0.0, 0.0],  # Wall (red)
            [0.0, 1.0, 0.0],  # Everything else (green)
            [0.0, 0.0, 1.0]   # Background (blue)
        ]

    def _get_sub_batch(self):
        """
        Group images with similar aspect ratios into a sub-batch.
        :return: A batch of images with either larger width or larger height.
        """
        while True:
            # Retrieve the current sample and increment the index.
            this_sample = self.list_sample[self.cur_idx]
            self.cur_idx += 1

            # Shuffle samples if the end of the dataset is reached.
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            # Retrieve paths for the current sample's image and segmentation map.
            image_path = os.path.join(self.image_root, this_sample['fpath_img'])
            segm_path = os.path.join(self.annotation_root, this_sample['fpath_segm'])

            # Group images by aspect ratio.
            if os.path.exists(image_path) and os.path.exists(segm_path):
                if this_sample['height'] > this_sample['width']:
                    self.batch_record_list[0].append(this_sample)  # Portrait orientation.
                else:
                    self.batch_record_list[1].append(this_sample)  # Landscape orientation.

            # If a batch list reaches the required batch size, retrieve it and reset the list.
            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break

        return batch_records

    def __getitem__(self, index):
        """
        Obtain a batch used for training.
        :param index: Index for shuffling.
        :return: Dictionary with 'img_data' containing a batch of images, and 'seg_label' with segmentation labels.
        """
        # Shuffle the sample list if this is the first access.
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # Get a sub-batch of images with similar aspect ratios.
        batch_records = self._get_sub_batch()

        # Determine a random resizing size for the shorter edge of images in the batch.
        if isinstance(self.imgSizes, (list, tuple)):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # Calculate each sample's scale factor to maintain aspect ratio and avoid exceeding max dimensions.
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(this_short_size / min(img_height, img_width),
                             self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Determine the final padded batch dimensions, ensuring divisibility by the padding constant.
        batch_width = int(self.round2nearest_multiple(np.max(batch_widths), self.padding_constant))
        batch_height = int(self.round2nearest_multiple(np.max(batch_heights), self.padding_constant))

        # Allocate tensors for images and segmentation labels in the batch.
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(self.batch_per_gpu,
                                  batch_height // self.segm_downsampling_rate,
                                  batch_width // self.segm_downsampling_rate).long()

        # Load, process, and add each sample in the batch to the allocated tensors.
        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # Load image and segmentation paths.
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.open(segm_path)
            assert segm.mode == "L"  # Confirm segmentation map is in grayscale.
            assert img.size == segm.size  # Ensure image and segmentation sizes match.

            # Randomly flip images and segmentation maps for data augmentation.
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # Resize images and segmentation maps to match target batch dimensions.
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # Adjust segmentation map size to maintain alignment after downsampling.
            segm_rounded = Image.new('L', (self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate),
                                           self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(segm_rounded,
                            (segm_rounded.size[0] // self.segm_downsampling_rate, 
                             segm_rounded.size[1] // self.segm_downsampling_rate),
                            interp='nearest')

            # Transform image and segmentation data to tensors.
            img = self.img_transform(img)
            segm = self.segm_transform(segm)
            segm[segm > 0] = 1  # Wall is 0, everything else is 1.

            # Store transformed samples in the batch tensor.
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

            target_color = random.choice(self.target_colors)
            target_color = torch.tensor(target_color, dtype=torch.float32)

        return {'img_data': batch_images, 'seg_label': batch_segms, 'target_color': target_color}

    def __len__(self):
        return int(1e8)

class ValDataset(BaseDataset):
    """
    Validation dataset class derived from the BaseDataset class.
    """
    def __init__(self, image_root, annotation_root, **kwargs):
        # Initialize the base dataset with image and annotation root paths.
        super(ValDataset, self).__init__(image_root=image_root, annotation_root=annotation_root, **kwargs)
        
        # Set the total number of samples for validation and an index pointer for tracking sample retrieval.
        self.num_sample = len(self.list_sample)
        self.index = 0

    def __getitem__(self, index):
        """
        Retrieve the next sample in the validation dataset.
        :param index: Sample index, though not used since retrieval is continuous.
        :return: Dictionary containing 'img_data' (image tensor), 'seg_label' (segmentation label), 
                 and 'name' (file name of the image).
        """
        while True:
            # Retrieve the current sample's file paths for image and segmentation.
            this_record = self.list_sample[self.index]
            self.index += 1

            # Reset index if the end of the dataset is reached to enable continuous looping.
            if self.index >= self.num_sample:
                self.index = 0

            # Construct paths for the image and segmentation files.
            image_path = os.path.join(self.image_root, this_record['fpath_img'])
            segm_path = os.path.join(self.annotation_root, this_record['fpath_segm'])

            # Exit loop if both files exist; otherwise, move to the next sample.
            if os.path.exists(image_path) and os.path.exists(segm_path):
                break

        # Load and validate image and segmentation map, converting image to RGB and segmentation to grayscale.
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert segm.mode == "L"  # Ensure segmentation is in grayscale.
        assert img.size == segm.size  # Ensure dimensions of image and segmentation match.

        # Apply transformations to the image and segmentation data.
        img = self.img_transform(img)
        segm = self.segm_transform(segm)
        segm[segm > 0] = 1  # Wall is 0, everything else is 1.

        # Return a dictionary with transformed image data, segmentation label, and image name.
        return {'img_data': img[None], 'seg_label': segm, 'name': this_record['fpath_img'].split('/')[-1]}

    def __len__(self):
        # Return the number of samples in the validation dataset.
        return self.num_sample


class TestDataset(BaseDataset):
    """
    Test dataset class derived from the BaseDataset class.
    """
    def __init__(self, image_root, annotation_root, **kwargs):
        # Initialize the base dataset with image and annotation root paths.
        super(TestDataset, self).__init__(image_root=image_root, annotation_root=annotation_root, **kwargs)
        
        # Set the total number of samples for testing and an index pointer for tracking sample retrieval.
        self.num_sample = len(self.list_sample)
        self.index = 0

    def __getitem__(self, index):
        """
        Retrieve the next sample in the test dataset.
        :param index: Sample index, though not used since retrieval is continuous.
        :return: Dictionary containing 'img_data' (image tensor), 'seg_label' (segmentation label), 
                 and 'name' (file name of the image).
        """
        while True:
            # Retrieve the current sample's file paths for image and segmentation.
            this_record = self.list_sample[self.index]
            self.index += 1

            # Reset index if the end of the dataset is reached to enable continuous looping.
            if self.index >= self.num_sample:
                self.index = 0

            # Construct paths for the image and segmentation files.
            image_path = os.path.join(self.image_root, this_record['fpath_img'])
            segm_path = os.path.join(self.annotation_root, this_record['fpath_segm'])

            # Exit loop if both files exist; otherwise, move to the next sample.
            if os.path.exists(image_path) and os.path.exists(segm_path):
                break

        # Load and validate image and segmentation map, converting image to RGB and segmentation to grayscale.
        img = Image.open(image_path).convert('RGB')
        segm = Image.open(segm_path)
        assert segm.mode == "L"  # Ensure segmentation is in grayscale.
        assert img.size == segm.size  # Ensure dimensions of image and segmentation match.

        # Apply transformations to the image and segmentation data.
        img = self.img_transform(img)
        segm = self.segm_transform(segm)
        segm[segm > 0] = 1  # Wall is 0, everything else is 1.

        # Return a dictionary with transformed image data, segmentation label, and image name.
        return {'img_data': img[None], 'seg_label': segm, 'name': this_record['fpath_img'].split('/')[-1]}

    def __len__(self):
        # Return the number of samples in the test dataset.
        return self.num_sample
