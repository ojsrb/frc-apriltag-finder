import cv2
import numpy as np
import os
import random

ids = range(23)
variations = range(500)
split = 0.8

img_size = 200

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

border_bits = 1

for id in ids:
    print("Processing ", id)
    for v in variations:
        if v > round(len(variations) * 0.8):
            dir = 'test'
        else:
            dir = 'train'

        background = np.ones((img_size, img_size, 3), np.uint8)

        tag_size = random.randint(20, 50)

        tag = cv2.aruco.generateImageMarker(dictionary, id, tag_size, border_bits)

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

        angle = np.random.uniform(-30, 30)
        center = (tag_size // 2, tag_size // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        tag = cv2.warpAffine(tag, rotation_matrix, (tag_size, tag_size))

        tag_float = tag.astype(np.float32)

        brightness = np.random.randint(low=-10, high=10)
        tag_float += brightness

        contrast = np.random.uniform(0.8, 1.2)
        tag_float = (tag_float - 127.5) * contrast + 127.5

        tag_float = np.clip(tag_float, 0, 255)
        tag = tag_float.astype(np.uint8)

        tag = cv2.GaussianBlur(tag, (3, 3), 0)

        y_off = 100 - (tag_size // 2)
        x_off = 100 - (tag_size // 2)

        background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

        # background[y_off:y_off+tag_size, x_off:x_off+tag_size] = tag

        #
        # if np.random.rand() > 0.7:
        #     noise = np.random.normal(0, 4, tag.shape).astype(np.uint8)
        #     tag = cv2.add(tag, noise)
        #
        # scale = random.uniform(0.2, 0.6)
        # new_size = int(img_size * scale)
        # tag = cv2.resize(tag, (new_size, new_size))
        #
        # tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
        #
        # max_x = img_size - new_size
        # max_y = img_size - new_size
        # pos_x = np.random.randint(0, max_x) if max_x > 0 else 0
        # pos_y = np.random.randint(0, max_y) if max_y > 0 else 0
        #
        # # Simple overlay: just replace the region (works because tag has white background)
        # # If you want transparency for white pixels, use the more complex masking above
        # background[pos_y:pos_y+new_size, pos_x:pos_x+new_size] = tag

        if not os.path.exists(f'./yolo/dataset/{dir}/{id}'):
            os.mkdir(f'./yolo/dataset/{dir}/{id}')
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        cv2.imwrite(f'./yolo/dataset/{dir}/{id}/{v}.png', tag)