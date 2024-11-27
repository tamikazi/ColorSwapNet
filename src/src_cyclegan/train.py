# src/src_cyclegan/train.py

import torch
from torch.autograd import Variable
import itertools
from tqdm import tqdm

import os
from torchvision.utils import save_image

def masked_L1_loss(input, target, mask):
    return torch.mean(torch.abs(input - target) * mask)

def train_one_epoch(
    G_AB, G_BA, D_A, D_B,
    optimizer_G, optimizer_D_A, optimizer_D_B,
    criterion_GAN, dataloader, epoch, device, writer, lambda_cycle=5.0, lambda_identity=1.0):

    max_norm = 5.0  # You can adjust this value
    
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    for i, batch in enumerate(tqdm(dataloader)):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)
        mask = batch['mask'].to(device)

        # ----------------------
        #  Train Generators
        # ----------------------

        optimizer_G.zero_grad()

        # Identity loss
        if lambda_identity > 0:
            # G_AB(B) should be equal to B if real B is provided
            idt_B = G_AB(real_B, mask)
            loss_idt_B = masked_L1_loss(idt_B, real_B, mask) * lambda_identity

            # G_BA(A) should be equal to A if real A is provided
            idt_A = G_BA(real_A, mask)
            loss_idt_A = masked_L1_loss(idt_A, real_A, mask) * lambda_identity
        else:
            loss_idt_B = 0
            loss_idt_A = 0

        # GAN loss
        fake_B = G_AB(real_A, mask)
        pred_fake_B = D_B(fake_B)
        valid = torch.ones_like(pred_fake_B).to(device)
        loss_GAN_AB = criterion_GAN(pred_fake_B, valid)

        fake_A = G_BA(real_B, mask)
        pred_fake_A = D_A(fake_A)
        valid = torch.ones_like(pred_fake_A).to(device)
        loss_GAN_BA = criterion_GAN(pred_fake_A, valid)

        # Cycle loss
        recovered_A = G_BA(fake_B, mask)
        loss_cycle_A = masked_L1_loss(recovered_A, real_A, mask) * lambda_cycle

        recovered_B = G_AB(fake_A, mask)
        loss_cycle_B = masked_L1_loss(recovered_B, real_B, mask) * lambda_cycle

        # Total generator loss
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        loss_G.backward()

        # Optional: Log gradient norm before clipping for generators
        total_norm_G = 0
        for p in itertools.chain(G_AB.parameters(), G_BA.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_G += param_norm.item() ** 2
        total_norm_G = total_norm_G ** 0.5
        writer.add_scalar('Gradients/G_before_clipping', total_norm_G, epoch * len(dataloader) + i)

        # Gradient clipping for generators
        torch.nn.utils.clip_grad_norm_(itertools.chain(G_AB.parameters(), G_BA.parameters()), 20.0)

        # Optional: Log gradient norm after clipping for generators
        clipped_total_norm_G = 0
        for p in itertools.chain(G_AB.parameters(), G_BA.parameters()):
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clipped_total_norm_G += param_norm.item() ** 2
        clipped_total_norm_G = clipped_total_norm_G ** 0.5
        writer.add_scalar('Gradients/G_after_clipping', clipped_total_norm_G, epoch * len(dataloader) + i)

        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real_A = D_A(real_A)
        valid = torch.ones_like(pred_real_A).to(device)
        loss_D_real_A = criterion_GAN(pred_real_A, valid)

        # Fake loss
        pred_fake_A = D_A(fake_A.detach())
        fake = torch.zeros_like(pred_fake_A).to(device)
        loss_D_fake_A = criterion_GAN(pred_fake_A, fake)

        # Total loss
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

        loss_D_A.backward()

        # Optional: Log gradient norm before clipping for D_A
        total_norm_D_A = 0
        for p in D_A.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_D_A += param_norm.item() ** 2
        total_norm_D_A = total_norm_D_A ** 0.5
        writer.add_scalar('Gradients/D_A_before_clipping', total_norm_D_A, epoch * len(dataloader) + i)

        # Gradient clipping for discriminator D_A
        torch.nn.utils.clip_grad_norm_(D_A.parameters(), max_norm)

        # Optional: Log gradient norm after clipping for D_A
        clipped_total_norm_D_A = 0
        for p in D_A.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clipped_total_norm_D_A += param_norm.item() ** 2
        clipped_total_norm_D_A = clipped_total_norm_D_A ** 0.5
        writer.add_scalar('Gradients/D_A_after_clipping', clipped_total_norm_D_A, epoch * len(dataloader) + i)

        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        pred_real_B = D_B(real_B)
        valid = torch.ones_like(pred_real_B).to(device)
        loss_D_real_B = criterion_GAN(pred_real_B, valid)

        # Fake loss
        pred_fake_B = D_B(fake_B.detach())
        fake = torch.zeros_like(pred_fake_B).to(device)
        loss_D_fake_B = criterion_GAN(pred_fake_B, fake)

        # Total loss
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

        loss_D_B.backward()

        # Optional: Log gradient norm before clipping for D_B
        total_norm_D_B = 0
        for p in D_B.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_D_B += param_norm.item() ** 2
        total_norm_D_B = total_norm_D_B ** 0.5
        writer.add_scalar('Gradients/D_B_before_clipping', total_norm_D_B, epoch * len(dataloader) + i)

        # Gradient clipping for discriminator D_B
        torch.nn.utils.clip_grad_norm_(D_B.parameters(), max_norm)

        # Optional: Log gradient norm after clipping for D_B
        clipped_total_norm_D_B = 0
        for p in D_B.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                clipped_total_norm_D_B += param_norm.item() ** 2
        clipped_total_norm_D_B = clipped_total_norm_D_B ** 0.5
        writer.add_scalar('Gradients/D_B_after_clipping', clipped_total_norm_D_B, epoch * len(dataloader) + i)

        optimizer_D_B.step()

        # Logging losses
        global_step = epoch * len(dataloader) + i
        writer.add_scalar('Loss/G', loss_G.item(), global_step)
        writer.add_scalar('Loss/D_A', loss_D_A.item(), global_step)
        writer.add_scalar('Loss/D_B', loss_D_B.item(), global_step)
        # Log individual losses
        writer.add_scalar('Loss/GAN_AB', loss_GAN_AB.item(), global_step)
        writer.add_scalar('Loss/GAN_BA', loss_GAN_BA.item(), global_step)
        writer.add_scalar('Loss/Cycle_A', loss_cycle_A.item(), global_step)
        writer.add_scalar('Loss/Cycle_B', loss_cycle_B.item(), global_step)
        writer.add_scalar('Loss/Identity_A', loss_idt_A.item(), global_step)
        writer.add_scalar('Loss/Identity_B', loss_idt_B.item(), global_step)

        # Log images to TensorBoard every N batches
        if i % 1000 == 0:
            # Prepare images for logging
            def denormalize(tensor):
                return (tensor + 1) / 2

            real_A_img = denormalize(real_A[:4].cpu())
            real_B_img = denormalize(real_B[:4].cpu())
            fake_B_img = denormalize(fake_B[:4].cpu())
            fake_A_img = denormalize(fake_A[:4].cpu())
            recovered_A_img = denormalize(recovered_A[:4].cpu())
            recovered_B_img = denormalize(recovered_B[:4].cpu())

            epoch_tag = f'Epoch_{epoch}'

            writer.add_images(f'{epoch_tag}A/real', real_A_img, global_step)
            writer.add_images(f'{epoch_tag}A/fake', fake_A_img, global_step)
            writer.add_images(f'{epoch_tag}A/recovered', recovered_A_img, global_step)

            writer.add_images(f'{epoch_tag}B/real', real_B_img, global_step)
            writer.add_images(f'{epoch_tag}B/fake', fake_B_img, global_step)
            writer.add_images(f'{epoch_tag}B/recovered', recovered_B_img, global_step)

            # Save images to a folder
            save_images_folder = os.path.join('saved_images', f'epoch_{epoch}')
            os.makedirs(save_images_folder, exist_ok=True)

            # Save images with batch number in filenames
            save_image(real_A_img, os.path.join(save_images_folder, f'real_A_epoch_{epoch}_batch_{i}.png'))
            save_image(real_B_img, os.path.join(save_images_folder, f'real_B_epoch_{epoch}_batch_{i}.png'))
            save_image(fake_B_img, os.path.join(save_images_folder, f'fake_B_epoch_{epoch}_batch_{i}.png'))
            save_image(fake_A_img, os.path.join(save_images_folder, f'fake_A_epoch_{epoch}_batch_{i}.png'))
            save_image(recovered_A_img, os.path.join(save_images_folder, f'recovered_A_epoch_{epoch}_batch_{i}.png'))
            save_image(recovered_B_img, os.path.join(save_images_folder, f'recovered_B_epoch_{epoch}_batch_{i}.png'))

    return loss_G.item(), loss_D_A.item(), loss_D_B.item()

