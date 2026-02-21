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
        self.avg_corners = []
        self.avg_center = (0,0)

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
        self.avg_center = corners2center(average_corners)
        return average_corners

tags = []

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
    x = (pos[0] // scale + cropped_img_pos[0])
    y = (pos[1] // scale + cropped_img_pos[1])
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

    index = 0
    result_positions = [(0,0)]

    for index, result in enumerate(results):
        pos = result["position"]

        min_dist = 10

        x = pos[0]
        y = pos[1]
        tooClose = False
        for i in result_positions:
            tag_x = i[0]
            tag_y = i[1]

            x_dist = abs(tag_x - x)
            y_dist = abs(tag_y - y)
            print(x_dist, y_dist)

            if x_dist < min_dist or y_dist < min_dist:
                tooClose = True

        if tooClose:
            print("too close")
            continue
        else:
            result_positions.append((int(x),int(y)))

        og = result["image"]

        # choose where to run the model
        # cpu should be fine
        preds = custom_upscale.run(og, device)
        # print(preds.shape)

        # pretty = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        (corners_list, marker_ids, rejected) = detector.detectMarkers(preds)
        preds = cv2.cvtColor(preds, cv2.COLOR_GRAY2BGR)
        cv2.aruco.drawDetectedMarkers(preds, corners_list, marker_ids, (0,255,0))
        # if corners_list != []:
        #     corners = corners_list[0]
        #     for i in corners:
        #         cv2.drawMarker(preds, i, (0,255,0), cv2.MARKER_CROSS)
        if marker_ids is not None:
            tag_corners = corners_list[0][0]
            id = marker_ids[0]

            center = corners2center(tag_corners)
            frame_position = cropped2global(center, pos, 4)

            found = False
            tag = None
            for i in tags:
                if i.id == id:
                    tag = i
                    i.corner_hist.append(tag_corners)
                    i.calc_average_corners()
                    i.corners = tag_corners
                    found = True

                    break

            if not found:
                tags.append(Tag(id, frame_position, og, preds))
                tag = tags[-1]

            if save and tag.avg_corners is not None:
                frame = tag.hr_img
                center = tag.avg_center
                frame = cv2.drawContours(frame, np.array([tag.avg_corners]), -1, (0, 255, 0), 2)

                for i in tag.avg_corners:
                    cv2.drawMarker(preds, i, (0,255,0), cv2.MARKER_CROSS, 50, 1)

                for i in tag.corner_hist:
                    cv2.drawMarker(preds, (int(i[0][0]), int(i[0][1])), (0,255,0), cv2.MARKER_CROSS, 50, 10)

                cv2.imshow(f'output/{tag.id}.jpg', preds)


    return tags, result_positions

if __name__ == '__main__':
    video = cv2.VideoCapture(sys.argv[1])
    while True:
        ok, frame = video.read()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not ok:
            continue

        annotated_frame, results = find.detect(frame)
        tags, detections = identify(results, True)

        for i in detections:
            cv2.drawMarker(frame, i, (0,255,0), cv2.MARKER_CROSS, 30, 4)

        if tags != []:
            for index, tag in enumerate(tags):
                cv2.imwrite(f'output/{tag.id}.jpg', tag.hr_img)
                if tag.avg_corners != []:
                    corners = globalize_corners(tag.avg_corners, tag.img_pos, 4)
                    center = cropped2global(tag.img_pos, tag.avg_center, 4)
                    for i in corners:
                        corner = cropped2global(tag.img_pos, i, 4)
                        cv2.drawMarker(frame, i, (255,0,0), cv2.MARKER_CROSS, 30, 4)

        cv2.imshow('video', annotated_frame)