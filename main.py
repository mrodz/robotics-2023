from detectors import default_detector
import cv2

def start_stream():
    video_stream = cv2.VideoCapture(0)

    previous_code = None

    while (True):
        ret, frame = video_stream.read()
        result = default_detector.id_from(frame)

        if result is not None:
            print(result)


if __name__ == "__main__":
    start_stream()
    # frame = cv2.imread('./images/images-2.png')
    # result = default_detector.id_from(frame)
    # print(result)
