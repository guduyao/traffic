1、本代码使用VOC格式进行训练。
2、训练前将标签文件放在VOCdevkit文件夹下的VOC2020文件夹下的Annotation中。
3、训练前将图片文件放在VOCdevkit文件夹下的VOC2020文件夹下的Dataset中。
4、在训练前利用voc2frcnn.py文件生成对应的txt。
5、再运行根目录下的voc_annotation.py，运行前需要将classes改成你自己的classes。
例：
classes = ["hands0", "hands1", "hands2", "hands3", "hands4", "hands5"]
6、就会生成对应的2020_train.txt，每一行对应其图片位置及其真实框的位置。
7、在训练前需要修改model_data里面的classes.txt文件，需要将类别改成你自己的类别。
8、同时需要修改train.py里面的NUM_CLASSES，修改成需要分的类的个数；BACKBONE为需要的主干特征提取网络。
9、运行train.py即可开始训练。