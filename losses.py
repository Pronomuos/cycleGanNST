import torch.nn as nn
from torch import ones, zeros
from torch.nn import functional as F


class CycleGanLoss(nn.Module):

    def __init__(self, discriminator_A, discriminator_B, gen_A2B, gen_B2A, device):
        super(CycleGanLoss, self).__init__()

        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B
        self.gen_A2B = gen_A2B
        self.gen_B2A = gen_B2A

        self.bce_loss = nn.BCELoss().to(device)  # losses for generator and discriminator
        self.mae_loss = nn.L1Loss().to(device)   # identity and cycle losses

    def backward(self, real_A, real_B):
        real_labels, fake_labels = ones(real_A.size(0)), zeros(real_A.size(0))

        def discriminator_loss(disc_real_results, disc_gen_results):
            real_loss = self.bce_loss(disc_real_results, real_labels)
            generated_loss = self.bce_loss(disc_gen_results, fake_labels)

            return (real_loss + generated_loss) * 0.5

        generated_B = self.gen_A2B.forward(real_A)
        generated_A = self.gen_B2A.forward(real_B)

        cycled_A = self.gen_B2A(generated_B)
        cycled_B = self.gen_A2B(generated_A)

        identical_A = self.gen_B2A(real_A)
        identical_B = self.gen_A2B(real_B)

        disc_real_A_results = self.discriminator_A(real_A)
        disc_real_B_results = self.discriminator_B(real_B)

        disc_gen_A_results = self.discriminator_A(generated_A)
        disc_gen_B_results = self.discriminator_B(generated_B)

        gen_A2B_loss = self.bce_loss(disc_gen_A_results, real_labels)
        gen_B2A_loss = self.bce_loss(disc_gen_B_results, real_labels)

        cycle_A_loss = self.mae_loss(cycled_A, real_A) * 10
        cycled_B_loss = self.mae_loss(cycled_B, real_B) * 10

        total_cycle_loss = cycle_A_loss + cycled_B_loss

        identity_A_loss = self.mae_loss(real_A, identical_A) * 0.5
        identity_B_loss = self.mae_loss(real_B, identical_B) * 0.5

        # Total generator loss = adversarial loss + cycle loss
        total_gen_A2B_loss = gen_A2B_loss + total_cycle_loss + identity_A_loss
        total_gen_B2A_loss = gen_B2A_loss + total_cycle_loss + identity_B_loss

        disc_x_loss = discriminator_loss(disc_real_A_results, disc_gen_A_results)
        disc_y_loss = discriminator_loss(disc_real_B_results, disc_gen_B_results)

        total_gen_A2B_loss.backward()
        total_gen_B2A_loss.backward()
        disc_x_loss.backward()
        disc_y_loss.backard()
