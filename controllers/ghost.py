import sys
import time
from pathlib import Path
import os
import os.path
from os import path
import torch
import shutil

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, strip_optimizer
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized

inboundfilename = ''
init = True

from flask import Flask, request
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)

@torch.no_grad()
def runInference(inboundfilefullpath,
                 weights='weights/best.pt',  # model.pt path(s)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=10,  # maximum detections per image
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 update=False,  # update all models
                 line_thickness=3,  # bounding box thickness (pixels)
                 hide_labels=False,  # hide labels
                 hide_conf=False,  # hide confidences
                 half=True,  # use FP16 half-precision inference
                 ):

    global init
    global model
    global stride
    global dataset
    imgsz = 416

    results = ''

    # Directories
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    if init:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names

        if half:
    	    model.half()  # to FP16

    dataset = LoadImages(inboundfilefullpath, img_size=imgsz, stride=stride)
    init = False

    # Run inference
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False, visualize=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    print(label + ('%g ' * 5) % (cls, *xyxy) + '|')
                    results = results + " " + label + ('%g ' * 5) % (cls, *xyxy) + '|'
                    plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

    print(results)

    #run cleanup in bg
    deleteinbound(inboundfilefullpath)

    return results

#def deleteallinbound(source):
#    source_dir = "/home/bob/inference/inbound"
#    target_dir = "/home/bob/inference/inbound/done"
#    file_names = os.listdir(source_dir)
#    os.remove(os.path.join(source_dir, source))
#    for file_name in file_names:
#        shutil.move(os.path.join(source_dir, file_name), target_dir)

def deleteinbound(path):
    print("deleting " + path)
    os.remove(path)

@app.route("/print_filename", methods=['POST','PUT'])
def print_filename():
    file = request.files['file']
    filename=secure_filename(file.filename)
    file.save('/home/bob/inference/inbound/' + filename)
    print('saved: ' + '/home/bob/inference/inbound/' + filename)
    print('running inference: ' + '/home/bob/inference/inbound/' + filename)
    result = runInference('/home/bob/inference/inbound/' + filename)
    print('done inference: ' + '/home/bob/inference/inbound/' + filename)
    print('result: ' + result)

    return result

if __name__=="__main__":
    app.run(host='192.168.1.30', port=76, debug=False, threaded=True)
