import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from model import ESPCN

class TagDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale_factor=4):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.lr_images = list(self.lr_dir.glob("*.png"))
        self.hr_images = list(self.hr_dir.glob("*.png"))
        self.scale_factor = scale_factor

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_path = self.lr_images[idx]
        hr_path = self.hr_images[idx]

        lr_img = Image.open(lr_path)
        hr_img = Image.open(hr_path)

        to_tensor = transforms.ToTensor()
        lr_tensor = to_tensor(lr_img)
        hr_tensor = to_tensor(hr_img)

        return lr_tensor, hr_tensor

def train_model(model, train_loader, num_epochs=100, device='cpu'):
    model = model.to(device)
    criterion = nn.MSELoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            loss = criterion(sr_imgs, hr_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch}/{num_epochs}, Loss: {avg_loss}')

        # if (epoch + 1) % 100 == 0:
        #     torch.save(model.state_dict(), f"apriltags_{epoch + 1}.pth")

    return model

dataset = TagDataset("dataset/low_res", "dataset/high_res")
train_loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

model = ESPCN(1, 64, 4)
trained = train_model(model, train_loader, num_epochs=200, device='mps')
torch.save(trained.state_dict(), "apriltags.pth")