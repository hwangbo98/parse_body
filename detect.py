import argparse
import time
from pathlib import Path
import json
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels').mkdir(parents=True, exist_ok=True)
    # (save_dir/ 'images').mkdir(parents=True, exist_ok=True) 
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    # if trace:
    #     model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        print(opt.classes, opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        mid_cls = 0
        big_cls = 0
        data = {}
        data['item_info'] = []
        
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            print(i)
            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            # save_path = str(save_dir / 'images' / p.stem) + f'_{frame}' + '.jpg'
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # for *xyxy, conf, cls in reversed(det):
                #     if save_txt:  # Write to file
                #         # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist() 
                #         line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                #         with open(txt_path + '.txt', 'a') as f:
                #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                #     if save_img or view_img:  # Add bbox to image
                #         label = f'{names[int(cls)]} {conf:.2f}'
                #         plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                count = 0
                # iteminfo = {}
                # iteminfo['item_info'] = []
                cls_list = []
                for k in range(len(det)) :
                    # print(f' det = {int(det[k][5])}')
                    cls_list.append(int(det[k][5]) + 1)
                print(f' class in this images : {cls_list}')
                result = {}
                result['top'] = []
                result['bottom'] = []
                result['shoes'] = []
                result['head'] = []
                # for cls_num in cls_list :
                #     if 0 < cls_num < 14 or cls_num == 27 or cls_num == 29 or cls_num == 31 :
                #         print(f'Top clothes is available')
                #         # top = []
                #     elif 13 < cls_num < 24 or cls_num == 30 or cls_num == 32 :
                #         print(f'Bottom clothes is available')
                #         # bottom = []
                #     elif 36 < cls_num < 42 :
                #         print(f'Shoes is available.')
                #         # shoes = []
                #     elif 52 < cls_num < 57 or 63 < cls_num < 66 :
                #         print(f'Head accesory is available.')
                #         # head = []

                # exit()
                for *xyxy, conf, cls in reversed(det):
                    
                    # if save_txt:  # Write to file
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) ).view(-1).tolist() 
                    if cls < 7 :
                        big_cls = 1
                        mid_cls = int(cls.item()) + 1
                    elif cls < 13 :
                        big_cls = 2
                        mid_cls = int(cls.item()) + 1
                    elif cls < 21 :
                        big_cls = 3
                        mid_cls = int(cls.item()) + 1
                    elif cls < 23 :
                        big_cls = 4
                        mid_cls = int(cls.item()) + 1
                    elif cls < 25 :
                        big_cls = 5
                        mid_cls = int(cls.item()) + 1
                    elif cls < 30 :
                        big_cls = 6
                        mid_cls = int(cls.item()) + 1
                    elif cls < 36 :
                        big_cls = 7
                        mid_cls = int(cls.item()) + 1
                    elif cls < 41 :
                        big_cls = 8
                        mid_cls = int(cls.item()) + 1
                    elif cls < 52 :
                        big_cls = 9
                        mid_cls = int(cls.item()) + 1
                    elif cls < 56 :
                        big_cls = 10
                        mid_cls = int(cls.item()) + 1
                    elif cls < 63 :
                        big_cls = 11
                        mid_cls = int(cls.item()) + 1
                    elif cls < 65 :
                        big_cls = 12
                        mid_cls = int(cls.item()) + 1
                    
                    if 0 < mid_cls < 14 or mid_cls == 27 or mid_cls == 29 or mid_cls == 31 :
                        print(f'Top clothes is available')
                        sub_top = []
                        sub_top.append(int(xyxy[0].item()))
                        sub_top.append(int(xyxy[1].item()))
                        sub_top.append(int(xyxy[2].item()))
                        sub_top.append(int(xyxy[3].item()))
                        print(f'sub_top = {sub_top}')
                        result['top'].append(sub_top)
                    elif 13 < mid_cls < 24 or mid_cls == 30 or mid_cls == 32 :
                        print(f'Bottom clothes is available')
                        sub_bottom = []
                        sub_bottom.append(int(xyxy[0].item()))
                        sub_bottom.append(int(xyxy[1].item()))
                        sub_bottom.append(int(xyxy[2].item()))
                        sub_bottom.append(int(xyxy[3].item()))
                        print(f'sub_bottom = {sub_bottom}')
                        result['bottom'].append(sub_bottom)
                    elif 36 < mid_cls < 42 :
                        print(f'Shoes is available.')
                        sub_shoes = []
                        sub_shoes.append(int(xyxy[0].item()))
                        sub_shoes.append(int(xyxy[1].item()))
                        sub_shoes.append(int(xyxy[2].item()))
                        sub_shoes.append(int(xyxy[3].item()))
                        print(f'sub_shoes = {sub_shoes}')
                        result['shoes'].append(sub_shoes)
                    elif 52 < mid_cls < 57 or 63 < mid_cls < 66 :
                        print(f'Head accesory is available.')
                        sub_head = []
                        sub_head.append(int(xyxy[0].item()))
                        sub_head.append(int(xyxy[1].item()))
                        sub_head.append(int(xyxy[2].item()))
                        sub_head.append(int(xyxy[3].item()))
                        print(f'sub_head = {sub_head}')
                        result['head'].append(sub_head)
                        # item = {}
                        
                        # # line = (cls, *xyxy, conf) if opt.save_conf else (big_cls, mid_cls, *xyxy)  # label format
                        # line = "-1:" + "-1:" + str(big_cls) + ":" + str(cls.item() + 1) + ":"
                        # item['item_id'] = line
                        # # with open(txt_path + '.txt', 'a') as f:
                        # #     f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        # bound = {}
                        # bound['lt_x'] = int(xyxy[0].item())
                        # bound['lt_y'] = int(xyxy[1].item())
                        # bound['rb_x'] = int(xyxy[2].item())
                        # bound['rb_y'] = int(xyxy[3].item())
                        # print(bound)
                        # item['bounding_box'] = bound
                        # iteminfo['item_info'].append(item)
                        # #data['item_info'].append({"item_id" : line})
                        # if save_img or view_img:  # Add bbox to image
                        #     label = f'{names[int(cls)]} {conf:.2f}'
                        #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
                # with open(txt_path + '.json', 'a') as f :
                #     json.dump(iteminfo, f)
                # print(f'Top = {top}, Bottom = {bottom}, shoes = {shoes}')
                print(result)
                final_result = {}
                
                #check if it is exist 
                if len(result['top']) > 1 :
                    top_size = 0
                    top_idx = 0
                    for k, top_value in enumerate(result['top']) :
                        area = (top_value[2] - top_value[0]) * (top_value[3] - top_value[1])
                        if area >= top_size :
                            top_size = area
                            top_idx = k
                    final_result['top'] = result['top'][top_idx]
                elif len(result['top']) == 1 :
                    final_result['top'] = result['top']
                else :
                    final_result['top'] = [-1,-1,-1,-1]

                if len(result['bottom']) > 1 :
                    bottom_size = 0
                    bottom_idx = 0
                    for k, bottom_value in enumerate(result['bottom']) :
                        area = (bottom_value[2] - bottom_value[0]) * (bottom_value[3] - bottom_value[1])
                        if area >= bottom_size :
                            bottom_size = area
                            bottom_idx = k

                    final_result['bottom'] = result['bottom'][top_idx]
                elif len(result['bottom']) == 1 :
                    final_result['bottom'] = result['bottom']
                else :
                    final_result['bottom'] = [-1,-1,-1,-1]

                if len(result['shoes']) > 1 :
                    shoes_size = 0
                    shoes_idx = 0
                    for k, shoes_value in enumerate(result['shoes']) :
                        area = (shoes_value[2] - shoes_value[0]) * (shoes_value[3] - shoes_value[1])
                        if area >= shoes_size :
                            shoes_size = area
                            shoes_idx = k

                    final_result['shoes'] = result['shoes'][top_idx]
                elif len(result['shoes']) == 1 :
                    final_result['shoes'] = result['shoes']
                else :
                    final_result['shoes'] = [-1,-1,-1,-1]

                if len(result['head']) > 1 :
                    shoes_size = 0
                    head_idx = 0
                    for k, head_value in enumerate(result['head']) :
                        area = (head_value[2] - head_value[0]) * (head_value[3] - head_value[1])
                        if area >= head_size :
                            head_size = area
                            head_idx = k

                    final_result['head'] = result['head'][top_idx]
                elif len(result['head']) == 1 :
                    final_result['head'] = result['head']
                else :
                    final_result['head'] = [-1,-1,-1,-1]
                # for k, top_cl in enumerate(top) :
                #     top_

            print(f'final_result = {final_result}')
            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #         print(f" The image with the result is saved in: {save_path}")
            #     else:  # 'video' or 'stream'
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                 save_path += '.mp4'
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer.write(im0)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
