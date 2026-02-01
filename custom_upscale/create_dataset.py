import cv2
import numpy as np
import os
import random

hr_dir = 'dataset/high_res'
lr_dir = 'dataset/low_res'

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

def generate(id):

    tag_size = 96

    tag = cv2.aruco.generateImageMarker(dictionary, id, tag_size, 1)

    h, w = tag.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    offset = int(w * 0.3)
    pts2 = np.float32([
        [np.random.randint(0, offset), np.random.randint(0, offset)],
        [w - np.random.randint(0, offset), np.random.randint(0, offset)],
        [w - np.random.randint(0, offset), h - np.random.randint(0, offset)],
        [np.random.randint(0, offset), h - np.random.randint(0, offset)]
    ])

    perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
    tag = cv2.warpPerspective(tag, perspective_matrix, (w, h),
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))

    angle = np.random.normal(-60, 60)
    center = (tag_size // 2, tag_size // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    tag = cv2.warpAffine(tag, rotation_matrix, (tag_size, tag_size), borderValue=(255, 255, 255))
    return tag

def compress(image):
        tag_size = 24
        tag = cv2.resize(image, (tag_size, tag_size), interpolation=cv2.INTER_AREA)

        tag_float = tag.astype(np.float32)

        brightness = np.random.randint(low=-60, high=60)
        tag_float += brightness

        contrast = np.random.uniform(0.6, 1.4)
        tag_float = (tag_float - 127.5) * contrast + 127.5

        tag_float = np.clip(tag_float, 0, 255)
        tag = tag_float.astype(np.uint8)

        tag = cv2.GaussianBlur(tag, (3, 3), 0)
        # tag = cv2.resize(tag, (tag_size*4, tag_size*4), interpolation=cv2.INTER_CUBIC)

        return tag

ids = 33
variations = 100

for i in range(ids):
    for j in range(variations):
        tag = generate(i)
        compressed = compress(tag)
        filename = f'{i}_{j}.png'
        cv2.imwrite(os.path.join(hr_dir, filename), tag)
        cv2.imwrite(os.path.join(lr_dir, filename), compressed)