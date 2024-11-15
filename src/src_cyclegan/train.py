# src/src_cyclegan/train.py

import torch
from torch.autograd import Variable
import itertools
from tqdm import tqdm

def train_one_epoch(
    G_AB, G_BA, D_A, D_B,
    optimizer_G, optimizer_D_A, optimizer_D_B,
    criterion_GAN, criterion_cycle, criterion_identity,
    dataloader, epoch, device, writer, lambda_cycle=10.0, lambda_identity=5.0
):
    G_AB.train()
    G_BA.train()
    D_A.train()
    D_B.train()

    for i, batch in enumerate(tqdm(dataloader)):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        # Adversarial ground truths
        valid = torch.ones((real_A.size(0), 1, 30, 30), requires_grad=False).to(device)
        fake = torch.zeros((real_A.size(0), 1, 30, 30), requires_grad=False).to(device)

        # ----------------------
        #  Train Generators
        # ----------------------

        optimizer_G.zero_grad()

        # Identity loss
        if lambda_identity > 0:
            # G_AB(B) should be equal to B if real B is provided
            idt_B = G_AB(real_B)
            loss_idt_B = criterion_identity(idt_B, real_B) * lambda_cycle * lambda_identity

            # G_BA(A) should be equal to A if real A is provided
            idt_A = G_BA(real_A)
            loss_idt_A = criterion_identity(idt_A, real_A) * lambda_cycle * lambda_identity
        else:
            loss_idt_B = 0
            loss_idt_A = 0

        # GAN loss
        fake_B = G_AB(real_A)
        pred_fake_B = D_B(fake_B)
        loss_GAN_AB = criterion_GAN(pred_fake_B, valid)

        fake_A = G_BA(real_B)
        pred_fake_A = D_A(fake_A)
        loss_GAN_BA = criterion_GAN(pred_fake_A, valid)

        # Cycle loss
        recovered_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recovered_A, real_A) * lambda_cycle

        recovered_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recovered_B, real_B) * lambda_cycle

        # Total generator loss
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real_A = D_A(real_A)
        loss_D_real_A = criterion_GAN(pred_real_A, valid)

        # Fake loss
        pred_fake_A = D_A(fake_A.detach())
        loss_D_fake_A = criterion_GAN(pred_fake_A, fake)

        # Total loss
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        pred_real_B = D_B(real_B)
        loss_D_real_B = criterion_GAN(pred_real_B, valid)

        # Fake loss
        pred_fake_B = D_B(fake_B.detach())
        loss_D_fake_B = criterion_GAN(pred_fake_B, fake)

        # Total loss
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

        loss_D_B.backward()
        optimizer_D_B.step()

        # Logging losses
        global_step = epoch * len(dataloader) + i
        writer.add_scalar('Loss/G', loss_G.item(), global_step)
        writer.add_scalar('Loss/D_A', loss_D_A.item(), global_step)
        writer.add_scalar('Loss/D_B', loss_D_B.item(), global_step)

        # Log images to TensorBoard every N batches
        if i % 100 == 0:
            # Prepare images for logging
            def denormalize(tensor):
                return (tensor + 1) / 2

            real_A_img = denormalize(real_A[:4].cpu())
            real_B_img = denormalize(real_B[:4].cpu())
            fake_B_img = denormalize(fake_B[:4].cpu())
            fake_A_img = denormalize(fake_A[:4].cpu())
            recovered_A_img = denormalize(recovered_A[:4].cpu())
            recovered_B_img = denormalize(recovered_B[:4].cpu())

            writer.add_images('A/real', real_A_img, global_step)
            writer.add_images('A/fake', fake_A_img, global_step)
            writer.add_images('A/recovered', recovered_A_img, global_step)

            writer.add_images('B/real', real_B_img, global_step)
            writer.add_images('B/fake', fake_B_img, global_step)
            writer.add_images('B/recovered', recovered_B_img, global_step)

    return loss_G.item(), loss_D_A.item(), loss_D_B.item()
