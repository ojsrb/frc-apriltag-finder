from super_image import EdsrModel, ImageLoader
from PIL import Image
import cv2
for i in range(16):
    image = Image.open(f'./test/tags/{i}.png')

    model = EdsrModel.from_pretrained('eugenesiow/msrn', scale=3)
    inputs = ImageLoader.load_image(image)
    preds = model(inputs)

    ImageLoader.save_image(preds, f'./scaled_{i}.png')

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()

    picture = cv2.imread(f'scaled_{i}.png', cv2.IMREAD_COLOR)

    cv_file = cv2.FileStorage("calibration.yaml", cv2.FILE_STORAGE_READ)
    mtx = cv_file.getNode("K").mat()
    dst = cv_file.getNode("D").mat()
    cv_file.release()

    gray = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    (corners, marker_ids, rejected) = detector.detectMarkers(gray)
    if marker_ids:
        print(marker_ids)

    cv2.aruco.drawDetectedMarkers(picture, corners, marker_ids)