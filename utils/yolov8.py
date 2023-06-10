from ultralytics.yolo.utils.ops import  non_max_suppression
from ultralytics.nn.autobackend import AutoBackend

from ultralytics.yolo.utils.torch_utils import select_device
import torch

class Yolov8Engine:
    def __init__(self, weights, device_str, classes, conf_thres, iou_thres, agnostic_nms, augment, half):
        
        self.half = half
        if device_str=='cuda':
            device = select_device(0)
        else:
            device ='cpu'

        self.model = AutoBackend(weights, device=device, dnn=False, fp16=half) #attempt_load(weights, map_location=device)

        if half:
            self.model.half()
       
        self.classes = classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.augment = augment
        self.agnostic_nms = agnostic_nms
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    def infer(self, img):
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img, augment=self.augment)[0]
        pred = self.nms(pred)
        return img, pred

    def nms(self, pred):
        out = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        return out

    def get_names(self):
        return self.model.names if hasattr(self.model, 'module') else self.model.names