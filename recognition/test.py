# -*- coding:utf-8 -*-
# @Datetime: 2020/12/23 10:39
# @Author: 铭仔
# @File: test.py
# @Software: PyCharm
import recognition.main_1 as m
import PIL
import numpy as np
from PIL import ImageFont,Image,ImageDraw
import cv2
from PIL import Image

if __name__ == '__main__':
    x=[]
    c = m.UI_main()
    #c.from_pic()
    img = cv2.imread('E:/pycharm/car/recognition/1.png')
    c.pic(img)



    print('c.car', c.car)
    a = c.car
    x.append(a)
    print('a', a)
    print(x)




    #img=cv2.putText(im,str(c.car),(0,40),font,1.2,(255,255,255),2)



    cv2img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
    pilimg = Image.fromarray(cv2img)

        # PIL图片上打印汉字
    draw = ImageDraw.Draw(pilimg)  # 图片上打印
    font = ImageFont.truetype("simhei.ttf", 40, encoding="utf-8")  # 参数1：字体文件路径，参数2：字体大小
    draw.text((0, 0), str(c.car), (255, 255, 0), font=font)  # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体

            # PIL图片转cv2 图片
    cv2charimg = cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)
    cv2.imshow('test', cv2charimg)
    cv2.waitKey()

