import find
import identify
import cv2
import PIL
from PIL import Image
import torch
import numpy as np
import json
import custom_upscale

class Tag:
    def __init__(self, id, img_pos, lr_img, hr_img):
        self.img_pos = img_pos
        self.id = id
        self.lr_img = lr_img
        self.hr_img = hr_img
        self.corner_hist = []
        self.avg_corners = []

        file = open("tags.json")
        tag_data = json.load(file)["tags"]

        for i in tag_data:
            if i["ID"] == id:
                tag_pos = i['pose']['translation']
                self.field_pos = (round(tag_pos['x'], 4), round(tag_pos['y'], 4), round(tag_pos['z'], 4))
                break

    def calc_average_corners(self):
        tl = []
        tr = []
        br = []
        bl = []

        for i in self.corner_hist:
            print(i)
            tl.append(i[0])
            tr.append(i[1])
            br.append(i[2])
            bl.append(i[3])

        average_corners = [
            average_positions(tl),
            average_positions(tr),
            average_positions(br),
            average_positions(bl)
        ]

        self.avg_corners = average_corners
        return average_corners

tags = []

def average_positions(list):
    x_list = []
    y_list = []
    for i in list:
        x_list.append(i[0])
        y_list.append(i[1])

    x_avg = round(sum(x_list) / len(x_list), 2)
    y_avg = round(sum(y_list) / len(y_list), 2)

    return (x_avg, y_avg)

def cropped2global(cropped_img_pos, pos, scale):
    x = (pos[0] // scale + cropped_img_pos[0])
    y = (pos[1] // scale + cropped_img_pos[1])
    return (x, y)

def globalize_corners(corners, img_pos, scale):
    global_corners = []
    global_corners.append(cropped2global(img_pos, corners[0][0], scale))
    global_corners.append(cropped2global(img_pos, corners[0][1], scale))
    global_corners.append(cropped2global(img_pos, corners[0][2], scale))
    global_corners.append(cropped2global(img_pos, corners[0][3], scale))
    return global_corners

def corners2center(corners):
    corner1 = corners[0][0]
    corner2 = corners[0][2]
    x = (corner1[0] + corner2[0]) // 2
    y = (corner1[1] + corner2[1]) // 2
    return (x, y)

def identify(results):

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    index = 0

    for i in results:
        pos = i["position"]
        og = i["image"]

        preds = custom_upscale.run(og)
        # print(preds.shape)

        # pretty = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        (corners, marker_ids, rejected) = detector.detectMarkers(preds)
        if marker_ids is not None:
            id = marker_ids[0]

            corners = corners[0]

            center = corners2center(corners)
            frame_position = cropped2global(center, pos, 4)

            preds = cv2.cvtColor(preds, cv2.COLOR_GRAY2BGR)

            found = False
            for i in tags:
                if i.id == id:
                    global_corners = globalize_corners(corners, pos, 4)
                    i.corner_hist.append(global_corners)
                    i.calc_average_corners()
                    i.corners = global_corners
                    found = True
                    break

            if not found:
                tags.append(Tag(id, frame_position, og, preds))

            print(tags[0].avg_corners)

        index += 1

    return tags

if __name__ == '__main__':
    video = cv2.VideoCapture('test/videos/camera_03.mp4')
    while True:
        ok, frame = video.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not ok:
            continue

        results = find.detect(frame)
        tags = identify(results)

        if tags != []:
            corners = tags[0].avg_corners
            if tags[0].avg_corners != []:
                frame = cv2.drawMarker(frame, (corners[0][0].astype(int), corners[0][1].astype(int), (0,255,0), cv2.MARKER_CROSS))

        cv2.imshow('video', frame)