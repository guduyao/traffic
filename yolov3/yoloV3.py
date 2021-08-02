import numpy as np
from pathlib import Path
from yolov3.darknet53 import *
import cv2
import random
from bounding_box import bounding_box as bb
import imutils
from tqdm import tqdm

MASK = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
ANCHORS = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)],
                   np.float32)
INPUT_SIZE = 416
ANCHORS = ANCHORS / 416
IOU_THRES = 0.5
MAX_BOXES = 100
SCORE_THRES = 0.5
DECAY = 0.0005
MOMENTUM = 0.9
CHANNELS = 3
COLOR_LIST = ['navy', 'blue', 'aqua',
              'teal', 'olive', 'green',
              'lime', 'yellow', 'orange',
              'red', 'maroon', 'fuchsia',
              'purple', 'black', 'gray',
              'silver']

YOLOV3_LAYER_LIST = [
    'DarkNet-53',
    'YOLO_FEATURE1_LAYER',
    'YOLO_OUTPUT_0',
    'YOLO_FEATURE2_LAYER',
    'YOLO_OUTPUT_1',
    'YOLO_FEATURE3_LAYER',
    'YOLO_OUTPUT_2']
CLASS_FILE = "E:/pycharm/car/yolov3/class_name.txt"
WEIGHT_FILE = r"E:/pycharm/car/yolov3/yolov3.weights"
OUTPUT_DIR = Path("outputs/")
class_names = [c.strip() for c in open(CLASS_FILE).readlines()]
id_list = list(np.arange(0, 81))
COLOR_MAPPING = {}
for idx in id_list:  COLOR_MAPPING[idx] = random.choice(COLOR_LIST)


def load_weights(model, weights_file, layers):
    with open(weights_file, 'rb') as wf:
        major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

        counter = 1

        for layer_name in layers:
            sub_model = model.get_layer(layer_name)
            for i, layer in enumerate(sub_model.layers):
                if not layer.name.startswith('conv2d'):
                    continue
                batch_norm = None
                if i + 1 < len(sub_model.layers) and sub_model.layers[i + 1].name.startswith('batch_norm'):
                    batch_norm = sub_model.layers[i + 1]

                filters = layer.filters
                size = layer.kernel_size[0]
                in_dim = layer.get_input_shape_at(0)[-1]

                if batch_norm is None:
                    conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
                else:
                    # darknet [beta, gamma, mean, variance]
                    bn_weights = np.fromfile(
                        wf, dtype=np.float32, count=4 * filters)
                    # tf [gamma, beta, mean, variance]
                    bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

                # darknet shape (out_dim, in_dim, height, width)
                conv_shape = (filters, in_dim, size, size)
                conv_weights = np.fromfile(
                    wf, dtype=np.float32, count=np.product(conv_shape))
                # tf shape (height, width, in_dim, out_dim)
                conv_weights = conv_weights.reshape(
                    conv_shape).transpose([2, 3, 1, 0])

                if batch_norm is None:
                    layer.set_weights([conv_weights, conv_bias])
                else:
                    layer.set_weights([conv_weights])
                    batch_norm.set_weights(bn_weights)

            percent_comp = (counter / len(layers)) * 100

            print('-- Loading weights. Please Wait...{:.2f}% Complete'.format(percent_comp),
                  end='\r', flush=True)
            counter += 1

        assert len(wf.read()) == 0, 'failed to read all data'
        print("\n-- Weights Loaded Successfully ... ")


def yolo_layer(filters, name=None):
    def output(x_in):
        if isinstance(x_in, tuple):
            inputs = keras.Input(x_in[0].shape[1:]), keras.Input(x_in[1].shape[1:])
            x, x_skip = inputs
            x = DarkNet_conv(x, filters, 1)
            x = keras.layers.UpSampling2D(2, interpolation="bilinear")(x)
            x = keras.layers.Concatenate()([x, x_skip])
        else:
            x = inputs = keras.Input(x_in.shape[1:])
        x = DarkNet_conv(x, filters, 1)
        x = DarkNet_conv(x, filters * 2, 3)
        x = DarkNet_conv(x, filters, 1)
        x = DarkNet_conv(x, filters * 2, 3)
        x = DarkNet_conv(x, filters, 1)
        return keras.Model(inputs, x, name=name)(x_in)

    return output


def yolo_output_layer(filters, num_anchors, num_classes, name=None):
    def output(x_in):
        x = inputs = keras.Input(x_in.shape[1:])
        x = DarkNet_conv(x, filters * 2, 3)
        x = DarkNet_conv(x, num_anchors * (num_classes + 5), 1, bn=False)
        x = keras.layers.Lambda(
            lambda x: tf.reshape(x, shape=(-1, tf.shape(x)[1], tf.shape(x)[2], num_anchors, num_classes + 5))
        )(x)

        return tf.keras.Model(inputs, x, name=name)(x_in)

    return output


def yolo_V3(size=INPUT_SHAPE, mask=MASK, num_class=NUM_CLASSES):
    """
    :param size:  Tuple-> yoloV3 input size
    :param mask:  yolo-v3 anchors mask
    :param num_class: classifier numbers
    :return: keras.Model, return output_1,output_2,output_3
    """
    x = inputs = keras.Input(shape=size, name="INPUT-416x416x3")
    feat1, feat2, feat3 = DarkNet53(name="DarkNet-53")(x)

    x = yolo_layer(512, name="YOLO_FEATURE1_LAYER")(feat3)
    output_1 = yolo_output_layer(512, len(mask[0]), num_class, "YOLO_OUTPUT_0")(x)

    x = yolo_layer(256, name="YOLO_FEATURE2_LAYER")((x, feat2))
    output_2 = yolo_output_layer(256, len(mask[1]), num_class, "YOLO_OUTPUT_1")(x)

    x = yolo_layer(128, name="YOLO_FEATURE3_LAYER")((x, feat1))
    output_3 = yolo_output_layer(128, len(mask[2]), num_class, "YOLO_OUTPUT_2")(x)

    return keras.Model(inputs, outputs=(output_1, output_2, output_3), name="YOLO_V3")
def yolo_V3(size=INPUT_SHAPE, mask=MASK, num_class=NUM_CLASSES):
    """
    :param size:  Tuple-> yoloV3 input size
    :param mask:  yolo-v3 anchors mask
    :param num_class: classifier numbers
    :return: keras.Model, return output_1,output_2,output_3
    """
    x = inputs = keras.Input(shape=size, name="INPUT-416x416x3")
    feat1, feat2, feat3 = DarkNet53(name="DarkNet-53")(x)

    x = yolo_layer(512, name="YOLO_FEATURE1_LAYER")(feat3)
    output_1 = yolo_output_layer(512, len(mask[0]), num_class, "YOLO_OUTPUT_0")(x)

    x = yolo_layer(256, name="YOLO_FEATURE2_LAYER")((x, feat2))
    output_2 = yolo_output_layer(256, len(mask[1]), num_class, "YOLO_OUTPUT_1")(x)

    x = yolo_layer(128, name="YOLO_FEATURE3_LAYER")((x, feat1))
    output_3 = yolo_output_layer(128, len(mask[2]), num_class, "YOLO_OUTPUT_2")(x)

    return keras.Model(inputs, outputs=(output_1, output_2, output_3), name="YOLO_V3")


def cvt_boxes(pred, t_xy, t_wh, anchors):
    grid_size = tf.shape(pred)[1:3]

    # create grid offsets
    grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
    # out: (grid_size,grid_size),(grid_size,grid_size)
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
    # out: (grid_size, grid_size, 2) -> (grid_size, grid_size, 1, 2)

    # sigmoid and add offset & scale with anchors
    # Normalize
    box_xy = (tf.sigmoid(t_xy) + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    wh = tf.exp(t_wh) * anchors

    # [x1, y1, x2, y2]: Normalized
    box_x1y1 = box_xy - wh / 2
    box_x2y2 = box_xy + wh / 2

    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)
    return bbox


def decode_predict(predict, anchors, num_class=NUM_CLASSES, input_dim=416):
    xy, wh, confidence, class_probs = tf.split(predict, [2, 2, 1, num_class], axis=-1)
    confidence = tf.sigmoid(confidence)
    class_probs = tf.sigmoid(class_probs)
    bbox = cvt_boxes(predict, xy, wh, anchors)
    return tf.concat([bbox, confidence, class_probs], axis=-1)


def nms(outputs, max_boxes=MAX_BOXES, iou_threshold=IOU_THRES, score_threshold=SCORE_THRES):
    boxes, conf, probs = [], [], []

    # extract values from the outputs
    # outputs: [3,batch_size,grid_size,grid_size,anchors,(Normalized:[x1,y1,x2,y2]+ojectness+num_classes)]
    for o in outputs:
        boxes.append(tf.reshape(o[..., 0:4], (tf.shape(o[..., 0:4])[0], -1, tf.shape(o[..., 0:4])[-1])))
        conf.append(tf.reshape(o[..., 4:5], (tf.shape(o[..., 4:5])[0], -1, tf.shape(o[..., 4:5])[-1])))
        probs.append(tf.reshape(o[..., 5:], (tf.shape(o[..., 5:])[0], -1, tf.shape(o[..., 5:])[-1])))

    bbox = tf.concat(boxes, axis=1)  # [..., 4]
    confidence = tf.concat(conf, axis=1)  # [..., 1]
    class_probs = tf.concat(probs, axis=1)  # [..., num_classes]

    # calculate the scores
    scores = confidence * class_probs

    # Compute Non-Max Supression
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        # [batch_size, num_boxes, num_classes, 4]
        boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
        # [batch_size, num_boxes, num_classes]
        scores=tf.reshape(scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])),
        max_output_size_per_class=max_boxes,
        max_total_size=max_boxes,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold)

    # shapes: [batch_size, max_boxes, 4], [batch_size, max_boxes], ..., [batch_size]
    return boxes, scores, classes, valid_detections


class YOLO_V3(keras.Model):
    def __init__(self, size=INPUT_SHAPE, num_class=NUM_CLASSES, anchors=ANCHORS, mask=MASK, maxbox=MAX_BOXES,
                 iou_threshold=IOU_THRES, score_threshold=SCORE_THRES):
        super(YOLO_V3, self).__init__()
        self.yolo_v3 = yolo_V3(size, mask, num_class)
        self.anchors = anchors
        self.mask = mask
        self.classes = num_class
        self.max_boxes = maxbox
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def call(self, inputs, training=None, mask=None):
        return self.yolo_v3(inputs, training=True)

    def load_net_weights(self):
        load_weights(model=self.yolo_v3, weights_file=WEIGHT_FILE, layers=YOLOV3_LAYER_LIST)


    def detect(self, img, model, display=True, verbose=1, **kwargs):
        wh = np.flip(img.shape[0:2])
        im = tf.cast(img / 255, tf.float32)
        im = tf.image.resize(im, size=[INPUT_SIZE, INPUT_SIZE])
        im = tf.expand_dims(im, axis=0)
        boxes, scores, classes, nums = model.predict(im, **kwargs)
        boxes, objectness, classes, nums = boxes[0], scores[0], classes[0], nums[0]
        if verbose > 0: print("-- Number of objects detected: ", int(nums))
        counter = 1
        for i in range(nums):
            x1y1 = tuple(np.array(boxes[i][0:2] * wh).astype(np.int32))
            x2y2 = tuple(np.array(boxes[i][2:4] * wh).astype(np.int32))
            cls_id = int(classes[i])
            label = class_names[cls_id] + ': {:.1f}'.format(objectness[i])
            bb.add(img, x1y1[0], x1y1[1], x2y2[0], x2y2[1], label=label, color=COLOR_MAPPING[cls_id])
            percent_comp = (counter / nums) * 100
            if verbose > 0:
                print('-- Drawing Predicitons. Please Wait...{:.2f}% Complete'
                      .format(percent_comp), end='\r', flush=True)
            counter += 1

        if display:
            cv2.imshow('img', img)
            cv2.waitKey()

        elif not display:
            return img


    def predict(self,
                x,
                batch_size=None,
                verbose=0,
                steps=None,
                callbacks=None,
                max_queue_size=10,
                workers=1,
                use_multiprocessing=False):
        super(YOLO_V3, self).predict(x=x, verbose=verbose)
        outputs = self(x, training=False)
        output_0, output_1, output_2 = outputs
        preds_0 = keras.layers.Lambda(
            lambda x: decode_predict(x, self.anchors[self.mask[0]], self.classes))((output_0))

        preds_1 = keras.layers.Lambda(
            lambda x: decode_predict(x, self.anchors[self.mask[1]], self.classes))((output_1))

        preds_2 = keras.layers.Lambda(
            lambda x: decode_predict(x, self.anchors[self.mask[2]], self.classes))((output_2))

        outputs = keras.layers.Lambda(
            lambda x: nms(x, self.max_boxes, self.iou_threshold, self.score_threshold)
        )((preds_0[:3], preds_1[:3], preds_2[:3]))

        return outputs

    @staticmethod
    def detect_video(video: str, model, output: str, **kwargs):
        """Draw Bounding box over VideoFile"""
        # initialize the video stream, pointer to the output video file,
        # and frame dimensions
        vs = cv2.VideoCapture(video)
        writer = None
        (W, H) = (None, None)

        # try to determin total number of images in the frame
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
            total = int(vs.get(prop))
            print("[INFO] {} total frames in video".format(total))
        # an error occurred while trying to determine the total
        # number of frames in the video file
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            total = -1

        print('[INFO] Drawing Predicitons', end="\t")
        if total > 0: bar = tqdm(range(total), leave=False)
        # loop over frames from the video file stream
        while True:

            # read the next frame from the file
            (grabbed, frame) = vs.read()

            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:     break

            # if the frame dimensions are empty, grab them
            if W is None or H is None:   (H, W) = frame.shape[:2]

            # Draw the bboxs on the image
            frame = model.detect(frame, model, display=False, **kwargs, verbose=0)

            # check if the video writer is None
            if writer is None:
                # initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*"MP4V")
                writer = cv2.VideoWriter(output, fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            # write the output frame to disk
            writer.write(frame)
            bar.update(1)

        # release the file pointers
        print("[INFO] cleaning up...")
        print(f"[INFO] output saved to {output} ... ")
        writer.release()
        vs.release()


if __name__ == '__main__':
    yolo = yolo_V3()
