import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

'''
利用YOLOv5的预测在新数据上框选目标框，并将获得的目标框信息保存并写入指定的img_path文件夹里的label_txt文件夹里

保存的目标框信息是YOLOv5官方使用的txt格式，还需要使用process_txt转换成xml格式才可以在labelimg中使用
'''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='./raw/', help='image path')
    parser.add_argument('--weights', nargs='+', type=str, default='csgo_yolov5s.pt', help='model path(s)')
    parser.add_argument('--data', type=str, default='data/csgo.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
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

    if not os.path.exists(opt.img_path + 'labels_txt'):
        os.makedirs(opt.img_path + 'labels_txt')

    for image in os.listdir (opt.img_path + 'images/'):
        img0 = cv2.imread (os.getcwd () + opt.img_path + 'images/' + image)
        img = letterbox (img0, imgsz, stride=stride, auto=True)[0]
        img = img.transpose ((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray (img)
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

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            im0 = img0

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=opt.line_thickness, example=str(names))

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if True:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)  # label format

                    if True:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                    print('Result:%s'%image, c, str(xywh))
                    out_file = open(opt.img_path + '/labels_txt/%s.txt' % (image)[:-4], 'a')
                    out_file.write (str(c) + " " + " ".join ([str (a) for a in xywh]) + '\n')


if __name__ == "__main__":
    opt = parse_opt()
    main()

