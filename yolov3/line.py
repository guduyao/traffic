import cv2
import numpy as np


def lines_detect(img):
    lines_lst = []
    gauss_img = cv2.GaussianBlur(img, (5, 5), 0)
    canny = cv2.Canny(gauss_img, 50, 150)

    h, w = canny.shape

    roi = np.array([[(0, h), (int(w / 2), int(h / 2)), (w, h)]])
    mask = np.zeros_like(canny)

    cv2.fillPoly(mask, roi, (255, 255, 255))
    masked_img = cv2.bitwise_and(mask, canny)
    lines = cv2.HoughLinesP(masked_img, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                if not -0.1 < (y2 - y1) / (x2 - x1) < 0.1 and -10 < (y2 - y1) / (x2 - x1) < 10:
                    if np.abs(np.sqrt(y2 ** 2 + x2 ** 2) - np.sqrt(y1 ** 2 + x1 ** 2)) > 10:
                        lines_lst.append([x1, y1, x2, y2])

    return lines_lst


if __name__ == '__main__':
    img = cv2.imread('test.png')
    lines = lines_detect(img)
    line_img = np.zeros_like(img)
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 10)
    cv2.line(img, (100, 100), (150, 150), (255, 255, 0), 10)
    result = cv2.addWeighted(img, 1, line_img, 0.8, 1)

    cv2.imshow('result', result)
    cv2.waitKey()
    print(len(lines))
