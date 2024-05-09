from faceboxes_pytorch.faceboxes_face_detector import FaceBoxesFaceDetector
import cv2

path = "test/1200px-Mona_Lisa_detail_face.jpg"


def test_get_facebox():
    image = cv2.imread(path)

    face_detector = FaceBoxesFaceDetector()

    confs, boxes = face_detector.get_faceboxes(image)

    box = boxes[0]
    # python3.7での動作を基準にテストをする
    assert (confs[0] - 0.99192) < 0.0001
    assert (box[0] - 220.99234) < 0.01
    assert (box[1] - 259.8105) < 0.01
    assert (box[2] - 814.88275) < 0.01
    assert (box[3] - 1161.3115) < 0.01
