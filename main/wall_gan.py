import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.constants import ROOT_DATASET, DEVICE, NUM_WORKERS, BATCH_PER_GPU
from utils.utils import create_data_loaders, not_None_collate, visualize_sample

IMAGE_TRAINING_ROOT = ROOT_DATASET + "/images/training"
IMAGE_VALIDATION_ROOT = ROOT_DATASET + "/images/validation"
IMAGE_TEST_ROOT = ROOT_DATASET + "/images/test"
ANNOTATION_TRAINING_ROOT = ROOT_DATASET + "/annotations/training"
ANNOTATION_VALIDATION_ROOT = ROOT_DATASET + "/annotations/validation"
ANNOTATION_TEST_ROOT = ROOT_DATASET + "/annotations/test"

def main():
    # init
    data_loaders = create_data_loaders(
    root_dataset=ROOT_DATASET,
    image_training_root=IMAGE_TRAINING_ROOT,
    annotation_training_root=ANNOTATION_TRAINING_ROOT,
    image_validation_root=IMAGE_VALIDATION_ROOT,
    annotation_validation_root=ANNOTATION_VALIDATION_ROOT,
    image_test_root=IMAGE_TEST_ROOT,
    annotation_test_root=ANNOTATION_TEST_ROOT,
    batch_per_gpu=BATCH_PER_GPU,
    num_workers=NUM_WORKERS,
    collate_fn_train=not_None_collate
    )

    # Access the loaders and iterators
    train_loader = data_loaders["train_loader"]
    train_iterator = data_loaders["train_iterator"]
    val_loader = data_loaders["val_loader"]
    val_iterator = data_loaders["val_iterator"]
    test_loader = data_loaders["test_loader"]
    test_iterator = data_loaders["test_iterator"]

    # Sample from the training loader
    batch_train = next(train_iterator)
    img_data_train, seg_label_train = batch_train[0]['img_data'], batch_train[0]['seg_label']

    # Check batch size and tensor shapes
    print(f"Training Batch - Image shape: {img_data_train.shape}, Mask shape: {seg_label_train.shape}")

    # Visualize the first sample in the training batch
    visualize_sample(img_data_train[0], seg_label_train[0], "Training Sample", "training_sample.png")

    # Sample from the validation loader
    batch_val = next(val_iterator)
    img_data_val, seg_label_val, name_val = batch_val[0]['img_data'], batch_val[0]['seg_label'], batch_val[0]['name']

    # Check tensor shapes and name
    print(f"Validation Sample - Name: {name_val}, Image shape: {img_data_val.shape}, Mask shape: {seg_label_val.shape}")

    # Visualize the validation sample
    visualize_sample(img_data_val, seg_label_val, name_val, "validation_sample.png")

    # Sample from the validation loader
    batch_test = next(test_iterator)
    img_data_test, seg_label_test, name_test = batch_test[0]['img_data'], batch_test[0]['seg_label'], batch_test[0]['name']

    # Check tensor shapes and name
    print(f"Test Sample - Name: {name_test}, Image shape: {img_data_test.shape}, Mask shape: {seg_label_test.shape}")

    # Visualize the validation sample
    visualize_sample(img_data_test, seg_label_test, name_test, "test_sample.png")

if __name__ == '__main__':
    main()