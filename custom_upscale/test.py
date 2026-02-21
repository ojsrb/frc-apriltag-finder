import sys
from types import SimpleNamespace
import torch
from model import ESPCN
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import json

scale = 4
image = sys.argv[1]
image = cv2.imread(image)

best_config = json.load(open("best_config.json"))
best_config = SimpleNamespace(best_config)

model = ESPCN(1, best_config, 4)
model.load_state_dict(torch.load("model.pth"))
model.eval()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_np = np.array(gray)
img = Image.fromarray(img_np)

target_size = (img.size[1] * scale, img.size[0] * scale)
img_bicubic = img.resize(target_size)

to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img_bicubic)

with torch.no_grad():
    sr_tensor = model(img_tensor)

sr_float = sr_tensor.numpy().transpose(1, 2, 0)
sr_image = (sr_float * 255.0).astype(np.uint8)

cv2.imshow("image", sr_image)
cv2.waitKey(0)