from ultralytics.yolo.utils.ops import  non_max_suppression
from ultralytics.nn.autobackend import AutoBackend

from ultralytics.yolo.utils.torch_utils import select_device

class Yolov8Engine:
    def __init__(self, weights, device_str, classes, conf_thres, iou_thres, agnostic_nms, augment, half):
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

    def infer(self, img):
        pred = self.model(img, augment=self.augment)[0]
        pred = self.nms(pred)
        return pred

    def nms(self, pred):
        out = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        return out

    def get_names(self):
        return self.model.names if hasattr(self.model, 'module') else self.model.names