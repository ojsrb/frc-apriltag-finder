from ultralytics import YOLO
import cv2

def classify(image):
    model = YOLO('identify/yolo/find.pt')
    results = model.predict(image)
    return results[0].probs.top1

if __name__ == "__main__":
    model = YOLO('yolo/find.pt')

    img = cv2.imread('../test/tags/3.png')

    # metrics = model.val()
    # print(metrics.top5 * 100)

    results = model.predict(img)
    id = results[0].probs.top1
    print(id)
    dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    marker = cv2.aruco.generateImageMarker(dict, id, 100)
    cv2.imshow('marker', marker)
    cv2.imshow('img', img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
