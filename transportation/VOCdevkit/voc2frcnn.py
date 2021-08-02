import os
import random
# xmlfilepath用于打开标签路径； saveBasePath用于保存生成的txt文件
xmlfilepath = r'E:/pycharm/car/transportation/VOCdevkit/VOC2020/Annotations'
saveBasePath = r"E:/pycharm/car/transportation/VOCdevkit/VOC2020/ImageSets/Main/"
"""
    train_percent：训练的数据集的比例  trainval_percent：训练所占比例，剩下的为测试集
"""
train_percent = 1
trainval_percent = 1
# 打开标签文件
temp_xml = os.listdir(xmlfilepath)
total_xml = []
for xml in temp_xml:
    if xml.endswith(".xml"):
        total_xml.append(xml)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
# 打印数据集的大小和训练数据集的大小
print("train and val size", tv)
print("traub suze", tr)
# 创建对应的txt文件
ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()