import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2020', 'train'), ('2020', 'val'), ('2020', 'test')]

wd = getcwd()
# 类别的名称
classes = ["red_light", "green_light", "yello_light"]


# convert_annotation用于打开标签和图片文件，生成对应的2020_train.txt，每一行对应其图片位置及其真实框的位置
def convert_annotation(year, image_id, list_file):
    in_file = open('E:/pycharm/car/transportation/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    if root.find('object')==None:
        return
    list_file.write('%s/VOCdevkit/VOC%s/Dataset/%s.jpg'%(wd, year, image_id))
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        convert_annotation(year, image_id, list_file)
    list_file.close()