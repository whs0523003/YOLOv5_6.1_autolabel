import os
from PIL import Image

# 用于批量裁剪图片尺寸
# 仅适用于1920*1080分辨率
path = input('请输入文件夹路径(结尾加上/)：')

# 获取该目录下所有文件，存入列表中
filelist = os.listdir(path)

for i in filelist:
    # 获得每张图片路径
    img_path = os.getcwd() + '/' + path + i
    # 获得图片
    img = Image.open(img_path)
    # 裁剪图片
    img_c = img.crop((640, 220, 1280, 860))
    img_c.save('dataset/' + i)


