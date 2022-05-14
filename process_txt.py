from xml.etree import ElementTree as ET
import os

'''
将Ultralytics-YOLOv5官方使用标签标注形式(txt)转换成labelimg使用的xml格式

目的是将官方YOLOv5预测的目标框信息反向写成labelimg所需的xml格式，
以实现利用YOLOv5的预测给新数据自动打标签，并在可以在labelimg上人工调整

结合process_xml.py使用

parent
├── yolov5_6.1_official
    └── current_path
        └── labels_xml (自动生成)
        └── labels_txt (需自己创建)
            └── img1.txt 
            └── img2.txt 
            └── ...    
        └── process_txt.py

1.current_path指process_xml.py工作的路径，可以直接在YOLOv根路径下执行，也可以在根路径下新建文件夹在文件夹内执行
2.需要txt格式标签文件
3.把需要处理的txt文件全部放在labels_txt文件夹里
4.处理完以后会在同级下生成labels_xml文件夹并保存结果
5.如果已经生成了文件再执行可能会报错，删除labels_xml文件夹再运行即可
'''

# ================================ 参数 ================================
dict_classes = {'0': 'gangster', '1': 'police'}  # 类别
# =====================================================================


def convert(data):
    dw = 640
    dh = 640

    x = ((float(data[1]) * dw * 2) - (float(data[3]) * dh)) / 2
    w = ((float(data[1]) * dw * 2) + (float(data[3]) * dh)) / 2
    y = ((float(data[2]) * dw * 2) - (float(data[4]) * dh)) / 2
    h = ((float(data[2]) * dw * 2) + (float(data[4]) * dh)) / 2

    xmin = int(x)
    xmax = int(w)
    ymin = int(y)
    ymax = int(h)
    return (xmin, ymin, xmax, ymax)


# 读取txt文件
for i in os.listdir('./labels_txt'):
    f = open('./labels_txt/'+'%s.txt' % (i[:-4]), 'r')
    lines = f.readlines()

    # 创建根节点annotation
    root = ET.Element("annotation")

    # 创建子节点folder
    folder = ET.SubElement(root, 'folder')
    folder.text = 'dataset'

    # 创建子节点filename
    filename = ET.SubElement(root, 'filename')
    filename.text = i[:-4] + '.jpg'

    # 创建子节点path
    path = ET.SubElement(root, 'path')
    path.text = 'F:\\1\\leetcode\dataset\\' + i[:-4] + '.jpg'

    # 创建子节点source
    source = ET.SubElement(root, 'source')

    # 创建子节点的子节点database
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # 创建子节点size
    size = ET.SubElement(root, 'size')

    # 创建子节点的子节点width
    width = ET.SubElement(size, 'width')
    width.text = '640'

    # 创建子节点的子节点database
    height = ET.SubElement(size, 'height')
    height.text = '640'

    # 创建子节点的子节点database
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'

    # 创建子节点
    segmented = ET.SubElement(root, 'segmented')
    segmented.text = '0'

    for line in lines:
        data = line.strip().split(' ')

        # 创建子节点object
        object = ET.SubElement(root, 'object')

        # 创建子节点的子节点name
        name = ET.SubElement(object, 'name')
        name.text = dict_classes[data[0]]

        # 创建子节点的子节点pose
        pose = ET.SubElement (object, 'pose')
        pose.text = 'Unspecified'

        # 创建子节点的子节点truncated
        truncated = ET.SubElement (object, 'truncated')
        truncated.text = '0'

        # 创建子节点的子节点difficult
        difficult = ET.SubElement (object, 'difficult')
        difficult.text = '0'

        # 创建子节点的子节点bndbox
        bndbox = ET.SubElement (object, 'bndbox')

        xmin_value, ymin_value, xmax_value, ymax_value = convert(data)

        # 创建子节点的子节点的子节点xmin
        xmin = ET.SubElement (bndbox, 'xmin')
        xmin.text = str(xmin_value)

        # 创建子节点的子节点的子节点ymin
        ymin = ET.SubElement (bndbox, 'ymin')
        ymin.text = str(ymin_value)

        # 创建子节点的子节点的子节点xmin
        xmax = ET.SubElement (bndbox, 'xmax')
        xmax.text = str(xmax_value)

        # 创建子节点的子节点的子节点xmin
        ymax = ET.SubElement (bndbox, 'ymax')
        ymax.text = str(ymax_value)

    ET.dump(root)

    tree = ET.ElementTree(root)

    if not os.path.exists('./labels_xml'):
        os.makedirs('./labels_xml')

    tree.write('./labels_xml/%s.xml'%i[:-4], encoding='utf-8', short_empty_elements=False)