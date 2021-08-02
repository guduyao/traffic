from people.yolo import YOLO, detect_video
from PIL import Image
import cv2
import numpy as np

# 参数配置
yolo_test_args = {
    "model_path": 'E:/pycharm/car/people/model_data/yolo.h5',
    "anchors_path": 'E:/pycharm/car/people/model_data/yolo_anchors.txt',
    "classes_path": 'E:/pycharm/car/people/model_data/coco_classes.txt',
    "score" : 0.4,
    "iou" : 0.40,
    "model_image_size" : (416, 416),
    "gpu_num" : 1,
}

# 定义图片检测方式辅助函数
def test_image(img):

    image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    r_image = yolo_test.detect_image(image)
    image1 = cv2.cvtColor(np.array(r_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('test', image1)
    cv2.waitKey()
    #r_image.show()

#  定义视频方式检测辅助函数
def test_video():
    while(True):
        _,img = cap.read()
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) #opencv->pil
        r_image = yolo_test.detect_image(image)
        op_image = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR) #pil->opencv
        cv2.imshow('test',op_image)

        if cv2.waitKey(16)==ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

#  主函数
if __name__=='__main__':
    yolo_test = YOLO(**yolo_test_args)

    # 1 视频测试：两种模式：1.1读取本地视频；1.2打开摄像头
    #video_path = '1.mp4' #本地视频路径
    #cap = cv2.VideoCapture(video_path) # 读取本地视频方式
    cap = cv2.VideoCapture(0)  #打开摄像头方式
    test_video()

    # 2 测试图片
    #image_path = 'E:/pycharm/car/people/1.jpg' #图片路径
    #img = cv2.imread(image_path)
    #test_image(img)
