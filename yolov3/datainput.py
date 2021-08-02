import os
from xml.etree import cElementTree as et
from tqdm import tqdm
import cv2 as cv
import numpy as np
import tensorflow as tf

img_path = r'D:\UA_DETRAC\Insight-MVT_Annotation_Train'
xml_path = r'D:\UA_DETRAC\DETRAC-Train-Annotations-XML'
SCALE_SIZE = [13, 26, 52]
CLASS_NUM = 4
ANCHOR_NUM_EACH_SCALE = 3
ANCHORS = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],
                   np.float32)


class IOUSameXY():
    def __init__(self, anchors, boxes):
        super(IOUSameXY, self).__init__()
        self.anchor_max = anchors / 2
        self.anchor_min = - self.anchor_max
        self.box_max = boxes / 2
        self.box_min = - self.box_max
        self.anchor_area = anchors[..., 0] * anchors[..., 1]
        self.box_area = boxes[..., 0] * boxes[..., 1]

    def calculate_iou(self):
        intersect_min = np.maximum(self.box_min, self.anchor_min)
        intersect_max = np.minimum(self.box_max, self.anchor_max)
        intersect_wh = np.maximum(intersect_max - intersect_min + 1.0, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # w * h
        union_area = self.anchor_area + self.box_area - intersect_area
        iou = intersect_area / union_area  # shape : [N, 9]

        return iou


class DataInput:
    def __init__(self, img_path=img_path, xml_path=xml_path):
        self.__img_path = img_path
        self.__xml_path = xml_path
        self.__all_img_dir_path = os.listdir(img_path)
        self.__xml_file_name = os.listdir(xml_path)
        self.__index = 0

    def preprocess_true_boxes(self, boxes, input_shape, anchors, num_classes):
        boxes_lst = []
        for box in boxes:
            assert (box[..., 4] < num_classes).all(), 'class id must be less than num_classes'
            # 一共有三个特征层数
            num_layers = len(anchors) // 3
            # 先验框
            # 678为116,90,  156,198,  373,326
            # 345为30,61,  62,45,  59,119
            # 012为10,13,  16,30,  33,23,
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]

            box = np.array(box, dtype='float32')
            input_shape = np.array(input_shape, dtype='int32')  # 416,416
            # 读出xy轴，读出长宽
            # 中心点(m,n,2)
            boxes_xy = (box[..., 0:2] + box[..., 2:4]) // 2
            boxes_wh = box[..., 2:4] - box[..., 0:2]
            # 计算比例
            box[..., 0:2] = boxes_xy / input_shape[::-1]
            box[..., 2:4] = boxes_wh / input_shape[::-1]

            # m张图
            m = box.shape[0]
            # 得到网格的shape为13,13;26,26;52,52
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
            # y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
            y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                               dtype='float32') for l in range(num_layers)]
            # [1,9,2]
            anchors = np.expand_dims(anchors, axis=0)
            anchor_maxes = anchors / 2.
            anchor_mins = -anchor_maxes
            # 长宽要大于0才有效
            valid_mask = boxes_wh[..., 0] > 0

            for b in range(m):
                # 对每一张图进行处理
                wh = boxes_wh[b, valid_mask[b]]
                if len(wh) == 0: continue
                # [n,1,2]
                wh = np.expand_dims(wh, -2)
                box_maxes = wh / 2.
                box_mins = -box_maxes

                # 计算真实框和哪个先验框最契合
                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
                box_area = wh[..., 0] * wh[..., 1]
                anchor_area = anchors[..., 0] * anchors[..., 1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                # 维度是(n) 感谢 消尽不死鸟 的提醒
                best_anchor = np.argmax(iou, axis=-1)

                for t, n in enumerate(best_anchor):
                    for l in range(num_layers):
                        if n in anchor_mask[l]:
                            # floor用于向下取整
                            i = np.floor(box[b, t, 0] * grid_shapes[l][1]).astype('int32')
                            j = np.floor(box[b, t, 1] * grid_shapes[l][0]).astype('int32')
                            # 找到真实框在特征层l中第b副图像对应的位置
                            k = anchor_mask[l].index(n)
                            c = box[b, t, 4].astype('int32')
                            y_true[l][b, j, i, k, 0:4] = box[b, t, 0:4]
                            y_true[l][b, j, i, k, 4] = 1
                            y_true[l][b, j, i, k, 5 + c] = 1
            boxes_lst.append(y_true)
        return boxes

    @staticmethod
    def resize416(img_h, img_w, x_min, y_min, x_max, y_max):
        """
        bbox in size (416,416)
        :param img_h: image h
        :param img_w: image w
        :param x_min: bbox x_min
        :param y_min: bbox y_min
        :param x_max: bbox x_max
        :param y_max: bbox y_max
        :return: x,y,w,h
        """

        x = x_min / img_w * 416
        w = x_max / img_w * 416
        y = y_min / img_h * 416
        h = y_max / img_h * 416

        x = int((x + w) - (w / 2))
        y = int((y + h) - (h / 2))
        return np.abs(x), np.abs(y), int(w), int(h)

    def __load_img(self, index):
        x = []
        img_shape = []
        print('Loading Dir in:' + os.path.join(self.__img_path, self.__all_img_dir_path[index]))
        for i in tqdm(os.listdir(os.path.join(self.__img_path, self.__all_img_dir_path[index]))):
            image = cv.imread(os.path.join(self.__img_path, self.__all_img_dir_path[index], i))
            img_shape.append(image.shape[:2])
            x.append(cv.resize(image, (416, 416)))

        x = tf.convert_to_tensor(np.array(x))

        label = []
        tree = et.parse(os.path.join(self.__xml_path, self.__xml_file_name[index]))
        root = tree.getroot()
        temp = {'car': 1, 'bus': 2, 'van': 3, 'others': 4}
        name_lst = []
        for img_num in range(2, len(root)):
            temp_lst = None
            h, w = img_shape[img_num - 2]
            for car_num in range(len(root[img_num][0])):
                left = int(float(root[img_num][0][car_num][0].get('left')))
                top = int(float(root[img_num][0][car_num][0].get('top')))
                width = int(float(root[img_num][0][car_num][0].get('width')))
                height = int(float(root[img_num][0][car_num][0].get('height')))
                vehicle_type = root[img_num][0][car_num][1].get('vehicle_type')
                left, top, width, height = self.resize416(h, w, left, top, width, height)
                name_lst.append(vehicle_type)
                if temp_lst is None:
                    temp_lst = np.array([left, top, width, height, temp[vehicle_type]])
                    temp_lst = np.expand_dims(temp_lst, axis=0)
                else:
                    box = np.expand_dims(np.array([left, top, width, height, temp[vehicle_type]]), axis=0)
                    temp_lst = np.concatenate((temp_lst, box), axis=0)

            label.append(temp_lst)
            # if label is None:
            #     label = temp_lst
            #     label = np.expand_dims(label, axis=0)
            # else:
            #     n_box = np.expand_dims(temp_lst, axis=0)
            #     label = np.vstack((label, n_box))

        print('==================================')
        print('- Labels:', set(name_lst))
        print('- Images shape:', x.shape)
        return x, label

    def next_batch(self):
        if self.__index > 59:
            self.__index = 0

        x, y = self.__load_img(self.__index)
        y = np.array(y)
        y = self.preprocess_true_boxes(y, (416, 416), ANCHORS, CLASS_NUM)
        self.__index += 1
        return x, y


if __name__ == '__main__':
    data = DataInput(img_path, xml_path)
    x, y = data.next_batch()
    for i in y:
        for j in i:
            print(j.shape)
    # img = np.array(x[300])
    # for i in y[300]:
    #     x, y, w, h = i[:4]
    #     cv.rectangle(img, (x - int(w /2), y - int(h/2)), (x + int(w/2), y + int(h/2)), (255, 255, 255), 1)
    #     print(i)
    # cv.imshow('img', img)
    # cv.waitKey()
