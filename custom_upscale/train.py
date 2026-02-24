import os
import sys
import threading
import types
import torch
from PIL import Image
from torch.nn import MSELoss
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
from model import ESPCN
import wandb
import json
import cv2
import numpy as np

wandb.login()

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

project = "apriltag-upscaler"

train_dataset = TagDataset("dataset/train/low_res", "dataset/train/high_res")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)

valid_dataset = TagDataset("dataset/valid/low_res", "dataset/valid/high_res")
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=0)

best_loss = 1.0

def sweep_run():
    with wandb.init(project=project) as run:
        config = run.config
        valid_loss, sr_imgs, _ = train_model(config, "mps")
        valid_tags = detect_markers(config, sr_imgs)
        run.log({ 'valid_tags': valid_tags, 'valid_loss': valid_loss })
        run.finish(0)

def detect_markers(config, sr_imgs):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    params = detector.getDetectorParameters()

    params.aprilTagQuadDecimate = config.quad_decimate
    params.aprilTagQuadSigma = config.quad_sigma
    params.aprilTagMinClusterPixels = config.min_cluster_pixels
    params.aprilTagMaxNmaxima = config.max_nmaxima
    params.aprilTagMinWhiteBlackDiff = config.min_white_black_diff

    detector.setDetectorParameters(params)

    detected_imgs = 0

    for i in sr_imgs:
        i = i.permute(1,2,0).cpu().detach().numpy()
        i = (i * 255).astype(np.uint8)
        (corners, ids, rejected) = detector.detectMarkers(i)
        if ids is not None:
            detected_imgs += 1

    return detected_imgs / len(sr_imgs)

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.bad_epochs = 0
        self.last_loss = 1

    def update(self, loss):
        if self.bad_epochs >= self.patience:
            bad_epochs = 0
            return True
        elif loss - self.last_loss > self.min_delta:
            self.bad_epochs += 1
        self.last_loss = loss
        return False

def train_model(config, device, save=True):
    model = ESPCN(1, config, 4)
    model = model.to(device)
    criterion = nn.L1Loss()
    criterion.to(device)

    lr = config.lr

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr, weight_decay=config.weight_decay,
        betas=(config.beta1, config.beta2),
    )
    scheduler = lr_scheduler.StepLR(optimizer, config.lr_step, config.lr_gamma)

    earlyStop = EarlyStopping(5, 0.05)

    all_sr_imgs = []

    print("starting epochs")

    for epoch in range(config.epochs):
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
        print(f'epoch {epoch}/{config.epochs}, loss: {loss.item()}')

        scheduler.step()

        if earlyStop.update(loss.item()):
            print("stopping early")
            break

    model.eval()
    valid_loss = 0
    valid_tags = 0
    with torch.no_grad():
        for lr_imgs, hr_imgs in valid_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            sr_imgs = model(lr_imgs)
            for i in sr_imgs:
                all_sr_imgs.append(i.cpu())
            loss = criterion(sr_imgs, hr_imgs)
            valid_loss += loss.item()

    if save:
        torch.save(model.state_dict(), "./model.pth")

    return valid_loss / len(valid_loader), all_sr_imgs, model

def wandb_sweep():
    config = json.load(open("config.json"))
    sweep_id = wandb.sweep(config, project=project)
    wandb.agent(sweep_id, sweep_run)

def save_best():
    api = wandb.Api()
    runs = api.runs("apriltag-upscaler")
    print(runs)
    runs = sorted(runs, key=lambda r: r.summary.get('valid_tags', 0), reverse=True)
    config = runs[0].config
    best_run = types.SimpleNamespace(**config)
    json.dump(config, open("best_config.json", "w"))

    best_run.epochs = 16

    print("starting training")

    _, _, model = train_model(best_run, "cpu", True)

    image = cv2.imread(sys.argv[2])
    scale = 4

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_np = np.array(gray)
    img = Image.fromarray(img_np)

    target_size = (img.size[1] * scale, img.size[0] * scale)
    img_bicubic = img.resize(target_size)

    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_bicubic)
    img_tensor = img_tensor.to('cpu')

    with torch.no_grad():
        sr_tensor = model(img_tensor)

    sr_float = sr_tensor.cpu().numpy().transpose(1, 2, 0)
    sr_image = (sr_float * 255.0).astype(np.uint8)

    cv2.imshow("1_hr.png", sr_image)
    cv2.waitKey(0)

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sweep":
        wandb_sweep()
    elif mode == "train":
        save_best()