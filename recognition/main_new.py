from transportation.frcnn import FRCNN
import cv2
import numpy as np
import recognition.main_1 as ma
from yolov3.predict import draw_img
from PIL import ImageFont, Image, ImageDraw


def photo(img):
    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

    # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), str(c.car), (255, 255, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

    # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2charimg = cv2.resize(cv2charimg, (900, 600))
    cv2.imshow('test', cv2charimg)
    cv2.waitKey()



if __name__ == '__main__':

    frcnn = FRCNN()
    img = Image.open(r'E:/pycharm/car/recognition/test3_1.jpg')
    img2 = cv2.imread('E:/pycharm/car/recognition/test3_2.jpg')
    r_image = frcnn.detect_image(img)
    img = np.array(r_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    draw_img(img)
    c=ma.UI_main()
    c.pic(img2)
    photo(img)



