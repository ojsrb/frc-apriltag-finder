import cv2
import custom_upscale
import numpy as np
import os

print(os.listdir("./test/tags"))

for i in os.listdir('./test/tags'):
    if i.endswith(".png"):
        lr_img = cv2.imread(f'./test/tags/{i}')
        hr_img = custom_upscale.run(lr_img)
        cv2.imwrite(f'./test/tags/hr/{i}', hr_img)
