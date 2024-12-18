import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.constants import ROOT_DATASET, LIST_SCENES
from utils.utils import count_files, move_and_rename_files, delete_files

IMAGE_TRAINING_ROOT = ROOT_DATASET + "/images/training"
IMAGE_VALIDATION_ROOT = ROOT_DATASET + "/images/validation"
IMAGE_TEST_ROOT = ROOT_DATASET + "/images/test"
ANNOTATION_TRAINING_ROOT = ROOT_DATASET + "/annotations/training"
ANNOTATION_VALIDATION_ROOT = ROOT_DATASET + "/annotations/validation"
ANNOTATION_TEST_ROOT = ROOT_DATASET + "/annotations/test"

SCENE_CATEGORIES_FILE = ROOT_DATASET + '/sceneCategories.txt'
OUTPUT_FILE = ROOT_DATASET + '/scene_categories_not_in_list.txt'

def main():
    # read the scene categories file and delete files with scenes not in LIST_SCENES
    with open(SCENE_CATEGORIES_FILE, 'r') as file, open(OUTPUT_FILE, 'w') as output:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            image_filename, scene_category = parts
            if scene_category not in LIST_SCENES:
                output.write(f"{scene_category}\n")
                delete_files(image_filename, IMAGE_TRAINING_ROOT, ANNOTATION_TRAINING_ROOT, IMAGE_VALIDATION_ROOT, ANNOTATION_VALIDATION_ROOT)
    
    # count the files in training and validation directories
    num_train_images = count_files(IMAGE_TRAINING_ROOT)
    num_val_images = count_files(IMAGE_VALIDATION_ROOT)
    total_images = num_train_images + num_val_images

    print(f"There are a total of {total_images} files with {num_train_images} training images and {num_val_images} validation images initially.")

    # calculate 10% of the total images for moving
    move_count = int(total_images * 0.1)
    move_count_val = int(abs(move_count - num_val_images)) # some images already exist in the validation set

    # move files from training to validation 
    print(f"Moving {move_count_val} files from training to validation for images and annotations...")
    move_and_rename_files(IMAGE_TRAINING_ROOT, ANNOTATION_TRAINING_ROOT, IMAGE_VALIDATION_ROOT, ANNOTATION_VALIDATION_ROOT, move_count_val, 'train', 'val2')

    # create test directories if they don't exist
    os.makedirs(IMAGE_TEST_ROOT, exist_ok=True)
    os.makedirs(ANNOTATION_TEST_ROOT, exist_ok=True)

    # move another 10% of files from training to test
    print(f"Moving {move_count} files from training to test for images and annotations...")
    move_and_rename_files(IMAGE_TRAINING_ROOT, ANNOTATION_TRAINING_ROOT, IMAGE_TEST_ROOT, ANNOTATION_TEST_ROOT, move_count, 'train', 'test')

    print("File moving and renaming process completed.")

if __name__ == "__main__":
    main()