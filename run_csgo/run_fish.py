import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import pyautogui

# 把当前工作环境从Game_Jump文件夹中切换到主文件夹yolov5中
last_path = '\\'.join(os.getcwd().split('\\')[:-1])  # 获得上一级目录
sys.path.append(last_path)  # 添加主路径到path路径中
os.chdir(last_path)  # 更改工作路径

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

'''
运行开始检测并自瞄
'''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.engine', help='model path(s)')
    parser.add_argument('--data', type=str, default='data/csgo.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

@torch.no_grad()
def main():
    # Load model
    device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(opt.imgsz, s=stride)  # check image size
    cudnn.benchmark = True  # set True to speed up constant image size inference

    while True:
        img0 = np.array(pyautogui.screenshot (region=(640, 220, 640, 640)))  # 1980 * 1080
        # img0 = np.array(pyautogui.screenshot(region=(640, 290, 320, 320)))  # 1600 * 900

        img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        bs = 1  # batch_size

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        t1 = time_sync()
        im = torch.from_numpy(img).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(im, augment=opt.augment, visualize=opt.visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes,
                                   opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # 存放所有框中的目标
        fish_centers = []
        x, y = 0, 0

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            im0 = img0

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    c = int(cls)  # integer class
                    label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # 如果检测到目标
                    if c == 0:
                        screen_size = 320
                        x_center = xywh[0] * screen_size
                        y_center = xywh[1] * screen_size
                        distance = ((screen_size / 2 - x_center) ** 2 + (screen_size / 2 - y_center) ** 2) ** 0.5

                        # 保存目标点信息
                        fish_centers.append((x_center, y_center, distance))

        # 找最近的一个目标，头
        try:
            fish_centers = np.array(head_centers)
            idx = int(np.where(head_centers[:, 2:] == min(head_centers[:, 2:]))[0])
            x, y = head_centers[idx][:2]

        except:
            pass

        # 移动鼠标
        if x != 0 and y != 0:
            xoffset = (x-screen_size/2) * 2.85
            yoffset = (y-screen_size/2) * 2.85
            # yoffset = 0  # 不直接锁头，自己手动调整
            pyautogui.moveRel(xoffset, yoffset)

        # =================== 以下可以省略 ===================
        # 在图上标记结果
        im0 = annotator.result ()
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

        # 打印时间信息
        t = tuple (x / seen * 1E3 for x in dt)  # speeds per image
        print((f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t))

        # 展示图片
        cv2.imshow('1', im0)
        cv2.waitKey(1)


if __name__ == "__main__":
    opt = parse_opt()
    main()
