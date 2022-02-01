from PIL import Image
from dataset import NSTDataset
from losses import CycleGanLoss
from model import ResnetGenerator, PatchGanDiscriminator
from torchvision import transforms
from tqdm import tqdm

import torch.utils.data
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(17)
image_size = 256
epoch_n = 40

transform = transforms.Compose([
    transforms.Resize(int(image_size * 1.12), Image.BICUBIC),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

root_dir = '/Users/apple/PycharmProjects/cycleGan/data/monet2photo'
dataset = NSTDataset(root_dir=root_dir, train=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)


gen_A2B = ResnetGenerator().to(device)
gen_B2A = ResnetGenerator().to(device)
discriminator_A = PatchGanDiscriminator().to(device)
discriminator_B = PatchGanDiscriminator().to(device)

optimizer_gen_A2B = torch.optim.Adam(gen_A2B.parameters())
optimizer_gen_B2A = torch.optim.Adam(gen_B2A.parameters())
optimizer_disc_A = torch.optim.Adam(discriminator_A.parameters())
optimizer_disc_B = torch.optim.Adam(discriminator_B.parameters())

cycleGanLoss = CycleGanLoss(discriminator_A=discriminator_A, discriminator_B=discriminator_B,
                            gen_A2B=gen_A2B, gen_B2A=gen_B2A, device=device)

for epoch in range(epoch_n):
    p_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, batch in p_bar:

        optimizer_gen_A2B.zero_grad()
        optimizer_gen_B2A.zero_grad()
        optimizer_disc_A.zero_grad()
        optimizer_disc_B.zero_grad()

        cycleGanLoss.backward(batch['imageA'], batch['imageB'])

        optimizer_gen_A2B.step()
        optimizer_gen_B2A.step()
        optimizer_disc_A.step()
        optimizer_disc_B.step()

