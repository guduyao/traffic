from yolov3.yoloV3 import *
from yolov3.line import *
import recognition.main_1 as m
from recognition.main_1 import deHaze

img = cv2.imread('E:/pycharm/car/yolov3/test.jpg')
model = YOLO_V3()
model.load_net_weights()

def draw_img(img):
    lines = lines_detect(img)
    line_img = np.zeros_like(img)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

    img = model.detect(img, model, False)
    result = cv2.addWeighted(img, 1, line_img, 0.95, 1)


def draw_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        _, img = cap.read()
        lines = lines_detect(img)
        line_img = np.zeros_like(img)
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 10)

        img = model.detect(img, model, False)
        result = cv2.addWeighted(img, 1, line_img, 0.95, 1)
        cv2.imshow('img', result)
        cv2.waitKey(1)

if __name__ == '__main__':
    draw_img(img)