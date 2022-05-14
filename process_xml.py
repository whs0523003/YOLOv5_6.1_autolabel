import os
import shutil
import random
import xml.etree.ElementTree as ET

'''
把用labelimg标注得到的xml格式的标签文件转换成Ultralytics-YOLOv5官方使用的训练数据格式

parent
├── yolov5_6.1_official
    └── current_path
        └── output_file (自动生成)
        └── input_file  (需自己创建)
            └── images (存放所有jpg图片)
                └── img1.jpg
                └── img2.jpg
                └── ...
            └── labels (存放使用labelimg标注得到的xml标签)
                └── img1.xml
                └── img2.xml
                └── ...
        └── process_xml.py
  
1.current_path指process_xml.py工作的路径，可以直接在YOLOv根路径下执行，也可以在根路径下新建文件夹在文件夹内执行
2.需要图片和xml格式标签文件
3.图片需要是jpg格式
4.图片名称需要与其xml标签名称一致
5.如果已经生成了文件再执行可能会报错，删除datasets文件夹再运行即可
'''

# ================================ 参数 ================================
input_file = './raw/'  # 输入要处理的文件夹路径，需要自己创建
output_file = './datasets/'  # 输出的txt格式文件的文件夹路径
dataset_name = './csgo_data'  # 要创立的数据集名称
classes = ['gangster', 'police']  # 数据类别

trainval_test = 0  # 指定(训练集+验证集)与测试集的比例，0则表示不划分测试集
train_val = 0.9      # 指定训练集与验证集的比例
# =====================================================================


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    try:
        in_file = open (input_file + 'labels/%s.xml' % (image_id), 'r', encoding='utf-8')
    except:
        print('输入文件路径有误！')
        return

    if not os.path.exists (output_file):
        os.makedirs (output_file)

        if not os.path.exists (output_file + dataset_name):
            os.makedirs (output_file + dataset_name)
            os.makedirs (output_file + dataset_name + '/images')
            os.makedirs (output_file + dataset_name + '/labels')

    out_file = open(output_file + dataset_name + '/labels/%s.txt' % (image_id), 'w')

    shutil.copy(input_file + '/images/%s.jpg' % (image_id) ,
                output_file + dataset_name + '/images/%s.jpg' % (image_id))

    tree = ET.parse (in_file)
    root = tree.getroot ()
    size = root.find ('size')
    w = int (size.find ('width').text)
    h = int (size.find ('height').text)

    for obj in root.iter ('object'):
        difficult = obj.find ('difficult').text
        cls = obj.find ('name').text
        if cls not in classes or int (difficult) == 1:
            continue
        cls_id = classes.index (cls)
        xmlbox = obj.find ('bndbox')
        b = (float (xmlbox.find ('xmin').text), float (xmlbox.find ('xmax').text),
             float (xmlbox.find ('ymin').text), float (xmlbox.find ('ymax').text))
        bb = convert ((w, h), b)

        out_file.write (str (cls_id) + " " + " ".join ([str (a) for a in bb]) + '\n')

    in_file.close()
    out_file.close()

def TrianValTest_divide():
    random.seed (0)
    if not os.path.exists(output_file + dataset_name + '/train'):
        os.makedirs(output_file + dataset_name + '/train')
        os.makedirs(output_file + dataset_name + '/train' + '/images')
        os.makedirs(output_file + dataset_name + '/train' + '/labels')

    if not os.path.exists(output_file + dataset_name + '/val'):
        os.makedirs(output_file + dataset_name + '/val')
        os.makedirs (output_file + dataset_name + '/val' + '/images')
        os.makedirs (output_file + dataset_name + '/val' + '/labels')

    if not os.path.exists(output_file + dataset_name + '/test') and trainval_test != 0:
        os.makedirs(output_file + dataset_name + '/test')
        os.makedirs (output_file + dataset_name + '/test' + '/images')
        os.makedirs (output_file + dataset_name + '/test' + '/labels')

    all = os.listdir(input_file + '/images')
    picked = random.sample (os.listdir (input_file + '/images'),
                            int(len (all) * train_val))
    rest = list(set(all) ^ set(picked))

    if trainval_test == 0:
        for i in picked:
            shutil.copy(output_file + dataset_name + '/images/%s' % (i),
                        output_file + dataset_name + '/train/images/%s' % (i))

            shutil.copy (output_file + dataset_name + '/labels/%s.txt' % (i[:-4]),
                         output_file + dataset_name + '/train/labels/%s.txt' % (i[:-4]))

        for i in rest:
            shutil.copy (output_file + dataset_name + '/images/%s' % (i),
                         output_file + dataset_name + '/val/images/%s' % (i))

            shutil.copy (output_file + dataset_name + '/labels/%s.txt' % (i[:-4]),
                         output_file + dataset_name + '/val/labels/%s.txt' % (i[:-4]))

    else:
        for i in rest:
            shutil.copy (output_file + dataset_name + '/images/%s' % (i),
                         output_file + dataset_name + '/test/images/%s' % (i))

            shutil.copy (output_file + dataset_name + '/labels/%s.txt' % (i[:-4]),
                         output_file + dataset_name + '/test/labels/%s.txt' % (i[:-4]))

        all_trainval = picked
        picked_trainval = random.sample (all_trainval,
                                int (len (all_trainval) * trainval_test))
        rest_trainval = list (set (all_trainval) ^ set (picked_trainval))

        for i in picked_trainval:
            shutil.copy (output_file + dataset_name + '/images/%s' % (i),
                         output_file + dataset_name + '/train/images/%s' % (i))

            shutil.copy (output_file + dataset_name + '/labels/%s.txt' % (i[:-4]),
                         output_file + dataset_name + '/train/labels/%s.txt' % (i[:-4]))

        for i in rest_trainval:
            shutil.copy (output_file + dataset_name + '/images/%s' % (i),
                         output_file + dataset_name + '/val/images/%s' % (i))

            shutil.copy (output_file + dataset_name + '/labels/%s.txt' % (i[:-4]),
                         output_file + dataset_name + '/val/labels/%s.txt' % (i[:-4]))

    shutil.rmtree(output_file + dataset_name + '/images')
    shutil.rmtree(output_file + dataset_name + '/labels')


if __name__ == '__main__':
    for i in range(len(os.listdir(input_file + '/labels'))):  # 遍历raw-labels文件夹长度
        convert_annotation ((os.listdir(input_file + '/labels')[i])[:-4])  # 取原本xml文件名称的.xml前的字符

    TrianValTest_divide()

    print('done')
