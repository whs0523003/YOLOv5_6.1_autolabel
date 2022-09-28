import os
import numpy as np
import cv2
import pyautogui

'''
每隔几秒保存一下屏幕的画面，保存图片命名为img1，img2，img3...，用于制作数据集
仅适用于1920*1080分辨率
'''


# ==================== 参数 ====================
i = 1  # 首张图片的名称，若输入10则以img10，img11，img12...顺序命名图片
second = 3000  # 每隔多少毫秒ms保存一次图片
# =============================================


# 在当前py文件所在位置新建文件夹raw_dataset
folder = 'raw_dataset'
if not os.path.exists(folder):  # 判断是否存在文件夹如果不存在则创建文件夹
    os.makedirs(folder)

while True:
    img = pyautogui.screenshot()
    img_np = np.array(img)

    # 1080P分辨率
    if img_np.shape[0] == 1080 and img_np.shape[1] == 1920:
        # 选屏幕中间640*640大小部分
        img_np = img_np[220:860, 640:1280]

    # 2k分辨率
    if img_np.shape[0] == 1440 and img_np.shape[1] == 2560:
        # 选屏幕中间960*960大小部分
        img_np = img_np[240:1200, 800:1760]

    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    cv2.imshow('s', frame)

    cv2.imwrite('raw_dataset/'+str(i)+'.jpg', frame)

    cv2.waitKey(second)
    i += 1

    cv2.destroyAllWindows()