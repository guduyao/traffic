

from tkinter import ttk
from PIL import Image, ImageTk
import recognition.img_function as predict
import recognition.img_math as img_math
import recognition.main as m
import PIL
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import cv2
from PIL import Image


class UI_main():
    pic_path = ""  # 图片路径
    colorimg = 'white'  # 车牌颜色
    cameraflag = 0
    car = []
    f = []

    def __init__(self):
        # 车牌颜色

        self.color_ct2 = ttk.Label(background=self.colorimg,
                                   text="", width="4", font=('Times', '14'))

        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        pil_image_resized = im.resize((500, 400), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=pil_image_resized)
        return imgtk

    def pic(self, img_bgr):  # 车牌识别功能
        # img_bgr = img_math.img_read(pic_path)
        first_img, oldimg = self.predictor.img_first_pre(img_bgr)
        if not self.cameraflag:
            self.imgtk = self.get_imgtk(img_bgr)
        r_color, roi_color, color_color = self.predictor.img_only_color(oldimg, oldimg, first_img)
        print("|", color_color,
              r_color, "|")
        self.car.append(color_color + ' ' + r_color)
        '''font = cv2.FONT_HERSHEY_SIMPLEX
        img = cv2.putText(pic_path, 'color_color', (0, 40), font, 1.2, (255, 255, 255), 2)
        cv2.imshow('test', img)
        cv2.waitKey()'''
        return color_color, r_color

    # 来自图片--->打开系统接口获取图片绝对路径

    def photo(self, img):
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


def zmMinFilterGray(src, r=7):
    '''''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r - 1, 2 * r - 1)))


def guidedfilter(I, p, r, eps):
    '''''引导滤波，直接参考网上的matlab代码'''
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p

    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I

    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b


def getV1(m, r, eps, w, maxV1):  # 输入rgb图像，值范围[0,1]
    '''''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)  # 得到暗通道图像
    V1 = guidedfilter(V1, zmMinFilterGray(V1, 7), r, eps)  # 使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)  # 计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()

    V1 = np.minimum(V1 * w, maxV1)  # 对值范围进行限制

    return V1, A


def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    V1, A = getV1(m, r, eps, w, maxV1)  # 得到遮罩图像和大气光照
    for k in range(3):
        Y[:, :, k] = (m[:, :, k] - V1) / (1 - V1 / A)  # 颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))  # gamma校正,默认不进行该操作
    return Y


if __name__ == '__main__':
    x = []
    c = UI_main()
    img = cv2.imread('E:/pycharm/car/recognition/5.jpg')
    print('是否执行去雾处理 1:不去雾  2:去雾')
    chose = input()
    if chose == '1':
        c.pic(img)
        # print('c.car', c.car)
        c.photo(img)
    elif chose == '2':
        m = deHaze((img) / 255.0) * 255
        cv2.imwrite('test.jpg', m)
        img = cv2.imread('test.jpg')
        c.pic(img)
        # print('c.car', c.car)
        c.photo(img)
    else:
        print('chongxinshuru')


