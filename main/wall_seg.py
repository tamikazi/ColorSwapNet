import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models_seg.models import SegmentationModule, build_encoder, build_decoder
from src.src_seg.train import create_optimizers, train_one_epoch, checkpoint
from src.src_seg.eval import validation_step
from utils.constants import ROOT_DATASET, CKPT_DIR_PATH_SEG, DEVICE, NUM_WORKERS, BATCH_PER_GPU, OPTIMIZER_PARAMETERS_SEG, NUM_EPOCHS_SEG
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

    net_encoder = build_encoder()
    net_decoder = build_decoder()

    # Creating criterion. In the dataset there are labels -1 which stand for "don't care", so should be ommited during training.
    crit = nn.NLLLoss(ignore_index=-1)

    # Creating Segmentation Module
    segmentation_module = SegmentationModule(net_encoder, net_decoder).to(DEVICE)

    # Set up optimizers
    nets = (net_encoder, net_decoder, crit)
    optimizers = create_optimizers(nets, OPTIMIZER_PARAMETERS_SEG)

    # Tensorboard initialization
    writer = SummaryWriter(os.path.join(CKPT_DIR_PATH_SEG, 'tensorboard'))
    writer.close()

    last_epoch = 0

    print('Starting training')
    # Main loop of certain number of epochs
    path_train_metadata = os.path.join(CKPT_DIR_PATH_SEG, 'training_metadata.pkl')
    if os.path.exists(path_train_metadata):
        with open(path_train_metadata, 'rb') as f:
            train_metadata = pickle.load(f)
    else:
        train_metadata = {'best_acc': 0, 'best_IOU': 0}

    for epoch in range(last_epoch, NUM_EPOCHS_SEG):
        print(f'Training epoch {epoch + 1}/{NUM_EPOCHS_SEG}...')
        train_one_epoch(segmentation_module, train_iterator, optimizers, epoch + 1, crit, writer)
        
        print(f'Starting evaluation after epoch {epoch + 1}')
        acc, IOU = validation_step(segmentation_module, val_loader, writer, epoch + 1)
        print('Evaluation Done!')
                
        if acc > train_metadata['best_acc']:
            train_metadata['best_acc'] = acc
            train_metadata['best_IOU'] = IOU
            save_best = True
            with open(path_train_metadata, 'wb') as f:
                pickle.dump(train_metadata, f)
            print(f'Epoch {epoch + 1} is the new best epoch!')
        else:
            save_best = False
        checkpoint(nets, epoch + 1, CKPT_DIR_PATH_SEG, save_best)

    writer.close()
    print('Training Done!')
    
    
if __name__ == '__main__':
    main()