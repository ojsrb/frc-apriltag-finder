import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import custom_upscale.model as models

scale = 4

def run(image):
    model = models.ESPCN(1, 64, 4)
    model.load_state_dict(torch.load("custom_upscale/apriltags.pth"))
    model.to('cpu')
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

    # sr_img.show()

    return sr_image



