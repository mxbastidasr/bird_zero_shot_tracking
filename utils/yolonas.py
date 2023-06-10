import torch
from super_gradients.training import models
import numpy as np

class YoloNasEngine:
    def __init__(self, weights, device_str, classes, conf_thres, iou_thres, agnostic_nms, augment, half):
       
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.model = models.get("yolo_nas_l", pretrained_weights="coco").to(device)

        self.classes = classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.augment = augment
        self.agnostic_nms = agnostic_nms

    def infer(self, img):
        pred =  list(self.model.predict(img,self.conf_thres))[0].prediction
        bboxes_xyxy = np.round(pred.bboxes_xyxy).astype(int)
        confidence = pred.confidence
        class_id = pred.labels.astype(int)
        return [(bboxes_xyxy, confidence, class_id)]#pred_list

    def nms(self, pred):
        out = None#non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        return out

    def get_names(self):
        img_dummy = np.zeros((224,224,3))
        names = list(self.model.predict(img_dummy))[0].class_names
        return names