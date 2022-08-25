import os
import shutil
import random

'''
将图片和txt格式的标签进行训练集，验证集，测试集的划分

parent
├── yolov5_6.1_official
    └── current_path
        └── output_file (自动生成)
        └── input_file  (需自己创建)
            └── images (存放所有jpg图片)
                └── img1.jpg
                └── img2.jpg
                └── ...
            └── labels (存放Ultralytics-YOLOv5官方使用的txt标签)
                └── img1.txt
                └── img2.txt
                └── ...
        └── process_trainval.py

1.current_path指process_trainval.py工作的路径，可以直接在YOLOv5根路径下执行，也可以在根路径下新建文件夹在文件夹内执行
2.需要图片和txt格式标签文件
3.图片需要是jpg格式
4.图片名称需要与其txt标签名称一致
5.如果已经生成了文件再执行可能会报错，删除datasets文件夹再运行即可
'''

# ================================ 参数 ================================
input_file = './input_file/'  # 输入要处理的文件夹路径，需要自己创建
output_file = './datasets/'  # 输出的txt格式文件的文件夹路径
dataset_name = './csgo_data'  # 要创立的数据集名称

trainval_test = 0  # 指定(训练集+验证集)与测试集的比例，0则表示不划分测试集
train_val = 0.9  # 指定训练集与验证集的比例

seed_num = 0  # 随机种子数
# =====================================================================


def TrianValTest_divide():
    random.seed(seed_num)
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
    picked = random.sample(os.listdir(input_file + '/images'), int(len (all) * train_val))
    rest = list(set(all) ^ set(picked))

    if trainval_test == 0:
        for i in picked:
            shutil.copy(input_file + '/images/%s' % (i),
                        output_file + dataset_name + '/train/images/%s' % (i))

            shutil.copy(input_file + '/labels/%s.txt' % (i[:-4]),
                        output_file + dataset_name + '/train/labels/%s.txt' % (i[:-4]))

        for i in rest:
            shutil.copy(input_file + '/images/%s' % (i),
                        output_file + dataset_name + '/val/images/%s' % (i))

            shutil.copy(input_file + '/labels/%s.txt' % (i[:-4]),
                        output_file + dataset_name + '/val/labels/%s.txt' % (i[:-4]))

    else:
        for i in rest:
            shutil.copy(input_file + '/images/%s' % (i),
                        output_file + dataset_name + '/test/images/%s' % (i))

            shutil.copy(input_file + '/labels/%s.txt' % (i[:-4]),
                        output_file + dataset_name + '/test/labels/%s.txt' % (i[:-4]))

        all_trainval = picked
        picked_trainval = random.sample(all_trainval, int(len(all_trainval) * trainval_test))
        rest_trainval = list(set(all_trainval) ^ set(picked_trainval))

        for i in picked_trainval:
            shutil.copy(input_file + '/images/%s' % (i),
                        output_file + dataset_name + '/train/images/%s' % (i))

            shutil.copy(input_file + '/labels/%s.txt' % (i[:-4]),
                        output_file + dataset_name + '/train/labels/%s.txt' % (i[:-4]))

        for i in rest_trainval:
            shutil.copy(input_file + '/images/%s' % (i),
                        output_file + dataset_name + '/val/images/%s' % (i))

            shutil.copy(input_file + '/labels/%s.txt' % (i[:-4]),
                        output_file + dataset_name + '/val/labels/%s.txt' % (i[:-4]))

    # shutil.rmtree(output_file + dataset_name + '/images')
    # shutil.rmtree(output_file + dataset_name + '/labels')

if __name__ == '__main__':
    TrianValTest_divide()

    print('Done!')