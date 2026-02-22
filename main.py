import numpy as np
import find
import cv2
import json
import custom_upscale
import sys

# cofiguration

device = 'cpu'

class Tag:
    def __init__(self, id, img_pos, lr_img, hr_img):
        self.img_pos = img_pos
        self.id = id
        self.lr_img = lr_img
        self.hr_img = hr_img
        self.corner_hist = []
        self.corners = []
        self.center = ()

        file = open("tags.json")
        tag_data = json.load(file)["tags"]

        for i in tag_data:
            if i["ID"] == id:
                tag_pos = i['pose']['translation']
                self.field_pos = (round(tag_pos['x'], 4), round(tag_pos['y'], 4), round(tag_pos['z'], 4))
                break

    def add_corners(self, corners):
        # no clue why this need to be 16 (prob math or smth)
        # but here we are
        global_corners = globalize_corners(corners, self.img_pos, 4)
        self.corners = global_corners
        self.center = corners2center(global_corners)

tags = {}

def average_positions(list):
    x_list = []
    y_list = []
    for i in list:
        x_list.append(i[0])
        y_list.append(i[1])

    x_avg = int(sum(x_list) / len(x_list))
    y_avg = int(sum(y_list) / len(y_list))

    return (x_avg, y_avg)

def cropped2global(cropped_img_pos, pos, scale):
    x = (pos[0] // (scale ** 2)) + cropped_img_pos[0]
    y = (pos[1] // (scale ** 2)) + cropped_img_pos[1]
    return (int(x), int(y))

def globalize_corners(corners, img_pos, scale):
    global_corners = []
    global_corners.append(cropped2global(img_pos, corners[0], scale))
    global_corners.append(cropped2global(img_pos, corners[1], scale))
    global_corners.append(cropped2global(img_pos, corners[2], scale))
    global_corners.append(cropped2global(img_pos, corners[3], scale))
    return global_corners

def corners2center(corners):
    corner1 = corners[0]
    corner2 = corners[2]
    x = (corner1[0] + corner2[0]) // 2
    y = (corner1[1] + corner2[1]) // 2
    return (int(x), int(y))

def identify(results, save=False):

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    for index, result in enumerate(results):
        pos = result["position"]

        og = result["image"]

        # choose where to run the model
        # cpu should be fine
        preds = custom_upscale.run(og, device)
        # print(preds.shape)

        # pretty = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        (corners_list, marker_ids, rejected) = detector.detectMarkers(preds)
        preds = cv2.cvtColor(preds, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(preds, corners_list, marker_ids, (0,255,0))
        if marker_ids is not None:
            id = int(marker_ids[0][0])
            tag_corners = corners_list[0][0]

            center = corners2center(tag_corners)

            try:
                tags[id].add_corners(tag_corners)
            except KeyError:
                tags[id] = Tag(id, pos, lr_img=og, hr_img=preds)
                tags[id].add_corners(tag_corners)


    return tags

def main(frame):
    annotated_frame, results = find.detect(frame)
    tags = identify(results, True)

    if tags != []:
        # iterate over all tags
        for tag in tags.values():
            cv2.imwrite(f'output/{tag.id}.jpg', tag.hr_img)
            cv2.drawMarker(frame, tag.center, (0,255,0), cv2.MARKER_CROSS, 15, 2)

    cv2.imshow('video', frame)

if __name__ == '__main__':
    if sys.argv[1].endswith(".mp4"):
        cap = cv2.VideoCapture(sys.argv[1])
        while True:
            ret, frame = cap.read()
            main(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    elif sys.argv[1].endswith(".jpg") or sys.argv[1].endswith(".jpeg") or sys.argv[1].endswith(".png"):
        img = cv2.imread(sys.argv[1])
        main(img)
        cv2.waitKey(0)