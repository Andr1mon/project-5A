import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
from typing import Type, Union
from IPython.display import clear_output, display
from ipywidgets import Output
from tqdm.auto import trange
from numpy.random import randint
import os
import zipfile
from torch.utils.data import Dataset
from torchvision import datasets

#%matplotlib inline

# Initialisation des valeurs du générateur de nombres aléatoires
np.random.seed(14)
torch.manual_seed(14)
print(torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

# Classe pour la transformation d'image
class PreprocessImage:
    # Obtenir des informations sur les contours de l'image
    def edge_information(self, image):
        img_np = np.array(image*255).transpose(1, 2, 0).astype(np.uint8)
        canny = cv.Canny(img_np,100,200)
        tau = 2
        edge = (canny + 1) / tau
        edge = np.exp(edge * (-1))
        return torch.from_numpy(edge)

    # Obtenir des informations sur la chrominance de l'image
    def chrominance_information(self, image):
        new_img = image #* 255
        y = 0.299 * new_img[0] + 0.587 * new_img[1] + 0.114 * new_img[2]
        cb = 0.564*(new_img[2] - y)
        cr = 0.713*(new_img[0] - y)
        teta = 0.25
        cb_norm = torch.square(cb)
        cr_norm = torch.square(cr)
        chrominance = (cb_norm + cr_norm) / (teta ** 2) * (-1)
        chrominance = torch.exp(chrominance) * (-1) + 1
        return chrominance

    # Transformation de l'image
    def preprocess_cover(self, image):
        img_norm = torch.zeros(image.size())
        chrominance = self.chrominance_information(image)
        edge = self.edge_information(image)
        know = (chrominance + edge) / 2
        img_norm[0] = image[0] + know - 1
        img_norm[1] = image[1] + know - 1
        img_norm[2] = image[2] + know - 1
        return img_norm

# Classe du dataset
class ImageDataset(Dataset):
    # Initialisation des variables
    def __init__(self, path):
        self.path = path
        self.cover_files = os.listdir(f'{self.path}/covers')
        self.logo_files = os.listdir(f'{self.path}/logo')
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.preprocess = PreprocessImage()

    # Longueur du dataset
    def __len__(self):
        return len(self.cover_files)
    
    # Récupérer un élément du dataset
    def __getitem__(self, idx):
        cover_path = self.cover_files[idx]
        logo_path = self.logo_files[idx]
        cover = Image.open(f'{self.path}/covers/{cover_path}').convert('RGB')
        logo = Image.open(f'{self.path}/logo/{logo_path}')
        cover = self.transform(cover)
        logo = self.transform(logo)
        cover_norm = self.preprocess.preprocess_cover(cover)
        return cover, logo, cover_norm

# Initialisation des ensembles de test et d'entraînement
train_dataset = ImageDataset('dataset/train')
test_dataset = ImageDataset('dataset/test')
train_dataloader = torch.utils.data.DataLoader(
train_dataset, batch_size=16, shuffle=True, num_workers=0
)
test_dataloader = torch.utils.data.DataLoader(
test_dataset, batch_size=16, shuffle=False, num_workers=0
)

# Classe de l'encodeur
class Encoder(nn.Module):
# Initialisation des couches du réseau de neurones
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1_watermark = nn.Conv2d(in_channels=1, out_channels=16,
        kernel_size=3, padding=1)
        self.conv2_watermark = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv3_watermark = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv4_watermark = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv5_watermark = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv6_watermark = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv7_watermark = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv1_cover = nn.Conv2d(in_channels=3, out_channels=16,
        kernel_size=3, padding=1)
        self.conv2_cover = nn.Conv2d(in_channels=32, out_channels=16,
        kernel_size=3, padding=1)
        self.conv3_cover = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv4_cover = nn.Conv2d(in_channels=32, out_channels=16,
        kernel_size=3, padding=1)
        self.conv5_cover = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv6_cover = nn.Conv2d(in_channels=32, out_channels=16,
        kernel_size=3, padding=1)
        self.conv7_cover = nn.Conv2d(in_channels=16, out_channels=16,
        kernel_size=3, padding=1)
        self.conv8_cover = nn.Conv2d(in_channels=35, out_channels=64,
        kernel_size=3, padding=1)
        self.conv9_cover = nn.Conv2d(in_channels=64, out_channels=128,
        kernel_size=3, padding=1)
        self.conv9_1_cover = nn.Conv2d(in_channels=128, out_channels=256,
        kernel_size=3, padding=1)
        self.conv9_2_cover = nn.Conv2d(in_channels=256, out_channels=128,
        kernel_size=3, padding=1)
        self.conv10_cover = nn.Conv2d(in_channels=128, out_channels=64,
        kernel_size=3, padding=1)
        self.conv11_cover = nn.Conv2d(in_channels=64, out_channels=32,
        kernel_size=3, padding=1)
        self.conv12_cover = nn.Conv2d(in_channels=32, out_channels=16,
        kernel_size=3, padding=1)
        self.conv13_cover = nn.Conv2d(in_channels=16, out_channels=3,
        kernel_size=3, padding=1)
        self.activator = nn.ReLU()

    # Structure du réseau de neurones
    def forward(self, input):
        (cover, watermark, cover_orig) = input
        watermark = self.conv1_watermark(watermark)
        cover = self.conv1_cover(cover)
        cover = torch.cat([cover, watermark], 1)
        watermark = self.conv2_watermark(watermark)
        watermark = self.conv3_watermark(watermark)
        cover = self.conv2_cover(cover)
        cover = self.conv3_cover(cover)
        cover = torch.cat([cover, watermark], 1)
        watermark = self.conv4_watermark(watermark)
        watermark = self.conv5_watermark(watermark)
        cover = self.conv4_cover(cover)
        cover = self.conv5_cover(cover)
        cover = torch.cat([cover, watermark], 1)
        watermark = self.conv6_watermark(watermark)
        watermark = self.conv7_watermark(watermark)
        cover = self.conv6_cover(cover)
        cover = self.conv7_cover(cover)
        cover = torch.cat([cover, watermark, cover_orig], 1)
        cover = self.conv8_cover(cover)
        cover = self.activator(self.conv9_cover(cover))
        cover = self.activator(self.conv9_1_cover(cover))
        cover = self.activator(self.conv9_2_cover(cover))
        cover = self.activator(self.conv10_cover(cover))
        cover = self.activator(self.conv11_cover(cover))
        cover = self.activator(self.conv12_cover(cover))
        cover = self.conv13_cover(cover)
        return cover

# Classe du décodeur
class Decoder(nn.Module):
    # Initialisation des couches du réseau de neurones
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
        kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
        kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
        kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
        kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64,
        kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32,
        kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=16,
        kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=1,
        kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(16)
        self.activator = nn.ReLU()

    # Structure du réseau de neurones
    def forward(self, input):
        output = self.activator(self.bn1(self.conv1(input)))
        output = self.activator(self.bn2(self.conv2(output)))
        output = self.activator(self.bn3(self.conv3(output)))
        output = self.activator(self.bn4(self.conv4(output)))
        output = self.activator(self.bn5(self.conv5(output)))
        output = self.activator(self.bn6(self.conv6(output)))
        output = self.activator(self.bn7(self.conv7(output)))
        output = self.conv8(output)
        return output
    
# Classe du Discriminateur
class Discriminator(nn.Module):
    # Initialisation des couches du réseau de neurones
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
        kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
        kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
        kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
        kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256,
        kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.activator = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(256 * 64 * 64, 1)

    # Structure du réseau de neurones
    def forward(self, input):
        output = self.activator(self.bn1(self.conv1(input)))
        output = self.activator(self.bn2(self.conv2(output)))
        output = self.activator(self.bn3(self.conv3(output)))
        output = self.activator(self.bn4(self.conv4(output)))
        output = self.activator(self.bn5(self.conv5(output)))
        output = self.pool(output)
        output = output.view(-1, 256 * 64 * 64)
        output = self.fc(output)
        output = self.sigmoid(output)
        return output
    
# Classe du Simulateur d'Attaque
class Attack:
    # Flou Gaussien
    def gaussian(self, image, p=3):
        transform_gaussian = transforms.Compose([transforms.GaussianBlur(p)])
        return transform_gaussian(image)
    
    # Recadrage
    def cropping(self, image):
        crop = torch.ones(image.size()).to(device)
        a = randint(0,crop.shape[1]-40)
        c = randint(0,crop.shape[2]-40)
        crop[:,a:a+40,c:c+40] = 0
        return image * crop
    
    # Dropout de pixels
    def dropout(self, image, p=0.15):
        mask = np.random.choice([0,1],image.size()[1:],True,[p,1-p])
        mask = torch.from_numpy(mask).to(device)
        return image[:] * mask

    # Ajout de bruit sel et poivre
    def salt(self, image, p=0.2):
        salt = np.random.choice([0,1],image.size()[1:],True,[p/2,1-p/2])
        peper = np.random.choice([0,1],image.size()[1:],True,[1-p/2,p/2])
        salt = torch.from_numpy(salt).to(device)
        peper = torch.from_numpy(peper).to(device)
        return image[:] * salt + peper
    
    def medianFilter(self, image, p = 5):
        img_np = np.asarray(image.cpu().detach()).transpose(1,2,0)
        img_bl = cv.medianBlur(img_np, p)
        return transforms.ToTensor()(img_bl)
    
    def jpg(self, image, p=90):
        img_np = np.asarray(image.cpu().detach()*255).transpose(1,2,0)
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), p]
        result, encimg = cv.imencode('.jpg', img_np, encode_param)
        decimg = cv.imdecode(encimg, 1)
        return transforms.ToTensor()(decimg)
    
    # Attaque aléatoire
    def random_attack(self, image):
        attack = randint(0,7)
        if attack == 1:
            return self.gaussian(image)
        elif attack == 2:
            return self.cropping(image)
        elif attack == 3:
            return self.dropout(image)
        elif attack == 4:
            return self.salt(image)
        elif attack == 5:
            return self.medianFilter(image)
        elif attack == 6:
            return self.jpg(image)
        return image
    
# Classe de l'Auto-Encodeur
class AutoEncoder(nn.Module):
    # Initialisation des variables
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.discriminator = Discriminator()
        self.attack_class = Attack()
        self.alfa = 0.5
        self.beta = 0.5
        self.sigma = 0.001
        self.criterion = nn.MSELoss()

    # Encodage
    def encode(self, x, y, z):
        return self.encoder((x,y,z))

    # Décodage
    def decode(self, x):
        return self.decoder(x)

    # Vérification de la présence d'un watermark
    def discriminate(self, x):
        return self.discriminator(x)

    # Application d'attaques sur les images
    def attack(self, batch):
        noise_batch = torch.ones(batch.size()).to(device)
        for i in range(batch.size()[0]):
            noise_batch[i] = self.attack_class.random_attack(batch[i])
        return noise_batch

    # Calcul de l'erreur du modèle
    def compute_loss(
        self,
        cover: torch.Tensor,
        watermark: torch.Tensor,
        cover_norm: torch.Tensor
        ) -> torch.Tensor:
        encode_image = self.encode(cover_norm, watermark, cover)
        is_watermark = self.discriminate(encode_image)
        encode_loss = self.criterion(cover,encode_image)
        discriminate_loss = - torch.log(is_watermark + 0.0001).mean()
        noise_image = self.attack(encode_image)
        decode_image = self.decode(encode_image)
        not_watermark = self.discriminate(cover)
        decode_loss = self.criterion(watermark,decode_image)
        discriminate_loss = discriminate_loss - torch.log(1 -
        not_watermark + 0.0001).mean()
        loss = self.alfa * encode_loss + self.beta * decode_loss + self.sigma * discriminate_loss
        return loss

# Entraînement d'une époque
def train_epoch(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    number,
    verbose_num_iters: int = 32,
    device: torch.device = "cuda",
    ) -> list[float]:
    model.to(device)
    model.train()
    epoch_loss_trace = []
    
    display()
    out = Output()
    display(out)
    for i, batch in enumerate(train_dataloader):
        cover, logo, cover_norm = batch
        cover = cover.to(device)
        logo = logo.to(device)
        cover_norm = cover_norm.to(device)
        loss = model.compute_loss(cover, logo, cover_norm)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss_trace.append(loss.item())
        
        if (i + 1) % verbose_num_iters == 0:
            with out:
                clear_output(wait=True)
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title(f"Current epoch loss: {number}", fontsize=22)
                plt.xlabel("Iteration", fontsize=16)
                plt.ylabel("Reconstruction loss", fontsize=16)
                plt.grid()
                plt.plot(epoch_loss_trace)
                #plt.show()
                plt.savefig("training/epoch_" + str(number) + " batch_" + str(i + 1) + ".png")
                plt.close()
    out.clear_output()
    return epoch_loss_trace

# Entraînement du modèle
def train_model(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 5,
    verbose_num_iters: int = 32,
    device: torch.device = "cuda"
    ) -> None:
    loss_trace = []
    epoch_number = 1
    for epoch in trange(num_epochs, desc="Epoch: ", leave=True):
        epoch_loss_trace = train_epoch(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        number = epoch_number,
        verbose_num_iters = verbose_num_iters,
        device=device,
        )
        loss_trace += epoch_loss_trace
        torch.save(model, f"models/model_{epoch_number}.pt")
        epoch_number += 1
    plt.figure(figsize=(10, 5))
    plt.title("Total training loss", fontsize=22)
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Reconstruction loss", fontsize=16)
    plt.grid()
    plt.plot(loss_trace)
    #plt.show()
    plt.savefig("training/training.png")
    plt.close()
    model.eval()

model = AutoEncoder()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_model(model, train_dataloader, optimizer, 15, device=device)
