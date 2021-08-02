#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from transportation.frcnn import FRCNN
from PIL import Image
import numpy as np


frcnn = FRCNN()
img = Image.open(r'E:\pycharm\car\transportation\test\1.jpg')
r_image = frcnn.detect_image(img)
r_image.show()
# r_image = frcnn.detect_image(img)
# draw_img('E:/pycharm/car/transportation/save.jpg')
#draw_img(img)
#
#r_image.show()
# while True:

    # img = input('Input image filename:')
    # try:
    #     image = Image.open(img)
    # except:
    #     print('Open Error! Try again!')
    #     continue
    # else:
    #     r_image = frcnn.detect_image(image)
    #     draw_img(cv2.cvtColor(r_image,cv2.COLOR_RGB2BGR))
