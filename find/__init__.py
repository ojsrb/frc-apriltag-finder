from ultralytics import YOLO
import os
import random
import cv2
import sys


def detect(frame, show=False):

    model = YOLO("find/apriltag_detector.pt")
    result = model.predict(frame, conf=0.1, verbose=False)[0]

    annotated_frame = result.plot()
    if show:
        cv2.imshow("frame", annotated_frame)
    outputs = []

    for box in result.boxes:
        position = box.xyxy
        outputs.append(crop(frame, position))

    return outputs

def crop(frame, position):
    position = position.cpu().numpy().astype(int)
    x1 = position[0, 0]
    y1 = position[0, 1]
    x2 = position[0, 2]
    y2 = position[0, 3]

    position = (x1, y1)

    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return {
        "image": frame[y1:y2, x1:x2],
        "position": position,
    }

if __name__ == "__main__":
    model = YOLO("../find.pt")
    cap = cv2.VideoCapture('../test/videos/2025wila_qm57.mp4')

    frame = cv2.imread('../test/pictures/2025alhu_qm43_frame0000.jpg')

    # tags = detect(frame, show=True)

    # index = 0
    # for i in tags:
    #     cv2.imwrite(f'../test/tags/{index}.png', i["image"])
    #     print(index)
    #     index += 1

    while True:
        ret, frame = cap.read()
        detect(frame, show=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break