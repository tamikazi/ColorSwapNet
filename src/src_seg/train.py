import os
import torch
import sys
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.constants import TOTAL_NUM_ITER_SEG, NUM_ITER_PER_EPOCH_SEG, OPTIMIZER_PARAMETERS_SEG, DEVICE
from utils.utils import pixel_acc, compute_dice, compute_jaccard


import matplotlib.pyplot as plt

def train_one_epoch(segmentation_module, iterator, optimizers, epoch, crit, writer):
    """
        Function for training model for only one epoch
    """

    segmentation_module.train()
    running_loss = 0.0
    running_pixel_acc = 0.0
    running_jaccard = 0.0
    running_dice = 0.0
    num_batches = 0

    for i in tqdm(range(NUM_ITER_PER_EPOCH_SEG)):
        # load a batch of data
        batch_data = next(iterator)[0]  # Because the batch size in the dataloader is 1, but the batch is created in TrainDataset
        segmentation_module.zero_grad()
        
        # adjust learning rate (learning rate "poly")  # TODO change to learning rate scheduler
        curr_iter = i + (epoch - 1) * NUM_ITER_PER_EPOCH_SEG
        lr_encoder, _ = adjust_learning_rate(optimizers, curr_iter)
                
        # forward pass
        pred = segmentation_module(batch_data)

        # Calculate loss and accuracy
        loss = crit(pred, batch_data['seg_label'].to(DEVICE))
        acc = pixel_acc(pred, batch_data['seg_label'].to(DEVICE))
        jaccard = compute_jaccard(pred, batch_data['seg_label'].to(DEVICE))
        dice = compute_dice(pred, batch_data['seg_label'].to(DEVICE))
               
        loss = loss.mean()
        acc = acc.mean()
        jaccard = jaccard.mean()
        dice = dice.mean()

        # Update running metrics
        running_loss += loss.item()
        running_pixel_acc += acc.item()
        running_jaccard += jaccard.item()
        running_dice += dice.item()
        num_batches += 1

        # Backward pass
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # Log metrics for each batch
        writer.add_scalar('Learning rate', lr_encoder, curr_iter)
        writer.add_scalar('Batch Loss', loss.data.item(), curr_iter)
        writer.add_scalar('Batch Pixel Accuracy', acc.data.item(), curr_iter)
        writer.add_scalar('Batch Jaccard', jaccard.data.item(), curr_iter)
        writer.add_scalar('Batch Dice', dice.data.item(), curr_iter)

    # Calculate averages
    epoch_loss = running_loss / num_batches
    epoch_pixel_acc = running_pixel_acc / num_batches
    epoch_jaccard = running_jaccard / num_batches
    epoch_dice = running_dice / num_batches

    print(f"Epoch {epoch}:")
    print(f"  Loss: {epoch_loss:.4f}")
    print(f"  Pixel Accuracy: {epoch_pixel_acc:.4f}")
    print(f"  Jaccard (IoU): {epoch_jaccard:.4f}")
    print(f"  Dice Coefficient: {epoch_dice:.4f}")

    # Log epoch metrics
    writer.add_scalar('Epoch Loss', epoch_loss, epoch)
    writer.add_scalar('Epoch Pixel Accuracy', epoch_pixel_acc, epoch)
    writer.add_scalar('Epoch Jaccard', epoch_jaccard, epoch)
    writer.add_scalar('Epoch Dice', epoch_dice, epoch)

    return epoch_loss, epoch_pixel_acc, epoch_jaccard, epoch_dice

def checkpoint(nets, epoch, checkpoint_dir_path, is_best_epoch): 
    """
        Function for saving encoder and decoder weights into a file
    """
    print('Saving checkpoints...')
    net_encoder, net_decoder, _ = nets

    dict_encoder = net_encoder.state_dict()
    dict_decoder = net_decoder.state_dict()
    
    torch.save(dict_encoder, os.path.join(checkpoint_dir_path, f'encoder_epoch_{epoch}.pth'))
    torch.save(dict_decoder, os.path.join(checkpoint_dir_path, f'decoder_epoch_{epoch}.pth'))
    
    previous_encoder_epoch = os.path.join(checkpoint_dir_path, f'encoder_epoch_{epoch - 1}.pth')
    if os.path.exists(previous_encoder_epoch):
        os.remove(previous_encoder_epoch)
        
    previous_decoder_epoch = os.path.join(checkpoint_dir_path, f'decoder_epoch_{epoch - 1}.pth')
    if os.path.exists(previous_decoder_epoch):
        os.remove(previous_decoder_epoch)
    
    if is_best_epoch:
        prev_best_models = [os.path.join(checkpoint_dir_path, x) for x in os.listdir(checkpoint_dir_path) if x.startswith('best_')]
        for model_path in prev_best_models:
            os.remove(model_path)
        torch.save(dict_encoder, os.path.join(checkpoint_dir_path, f'best_encoder_epoch_{epoch}.pth'))
        torch.save(dict_decoder, os.path.join(checkpoint_dir_path, f'best_decoder_epoch_{epoch}.pth'))


def group_weight(module):
    """
        Function for grouping weights and biases of a network into individual groups for training
    """
    group_decay = []
    group_no_decay = []

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.modules.conv._ConvNd)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    return [
        dict(params=group_decay),
        dict(params=group_no_decay, weight_decay=0),
    ]


def create_optimizers(nets, optim_parameters):
    """
        Creating individual optimizers for encoder and decoder
    """
    (net_encoder, net_decoder, crit) = nets
    
    optimizer_encoder = torch.optim.SGD(group_weight(net_encoder),
                                        lr=optim_parameters["LEARNING_RATE"],
                                        momentum=optim_parameters["MOMENTUM"],
                                        weight_decay=optim_parameters["WEIGHT_DECAY"])
    
    optimizer_decoder = torch.optim.SGD(group_weight(net_decoder),
                                        lr=optim_parameters["LEARNING_RATE"],
                                        momentum=optim_parameters["MOMENTUM"],
                                        weight_decay=optim_parameters["WEIGHT_DECAY"])
        
    return optimizer_encoder, optimizer_decoder


def adjust_learning_rate(optimizers, curr_iter):
    """
        Adjusting learning rate in each iteration
    """
    scale_running_lr = ((1 - curr_iter/TOTAL_NUM_ITER_SEG) ** 0.9)
    start_lr = OPTIMIZER_PARAMETERS_SEG["LEARNING_RATE"]

    lr_encoder = start_lr * scale_running_lr
    lr_decoder = start_lr * scale_running_lr

    (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer_encoder.param_groups:
        param_group['lr'] = lr_encoder
    for param_group in optimizer_decoder.param_groups:
        param_group['lr'] = lr_decoder

    return lr_encoder, lr_decoder