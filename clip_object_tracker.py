#python clip_object_tracker.py --weights models/yolov8x.pt --source "data/video/hummingbird_hq_AdobeExpress.mp4" --detection-engine yolonas --save-txt --animate --clip-labels "hummingbird" "flower" "bird feeder" --animate_over iou
#https://github.com/roboflow/zero-shot-object-tracking

import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import pandas as pd

import numpy as np

from utils.datasets import LoadImages
from utils.general import xyxy2xywh, increment_path, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

# deep sort imports
from deep_sort import  nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import generate_clip_detections as gdet

from utils.yolov8 import Yolov8Engine
from utils.yolonas import YoloNasEngine
from clip_zero_shot_classifier import ClipClassifier
from utils.jaccard_frames import get_color_for, jaccard_consecutive_frames, marking_pauses_on_video, jaccard_animation

import logging

logging.getLogger("super_gradients").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
classes = []

names = []


class DetectionAndTracking:
    def __init__(self, opt) -> None:
        self.opt = opt
        self.nms_max_overlap = opt.nms_max_overlap
        self.max_cosine_distance = opt.max_cosine_distance
        self.nn_budget = opt.nn_budget
        self.exist_ok = opt.exist_ok
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print('choosen dedvice: ',self.device)
        self.half = self.device != "cpu"

        
        self.model = ClipClassifier(device=self.device, labels=opt.clip_labels)
        
        self.encoder = gdet.create_box_encoder(self.model, model_name = opt.detection_engine)
        # calculate cosine distance metric
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget)
        # load yolov7 model here
        
        if opt.detection_engine == "yolov8":
            self.yolov8_engine = Yolov8Engine(opt.weights, self.device, opt.classes, opt.confidence, opt.overlap, opt.agnostic_nms, opt.augment, self.half)
            self.names = self.yolov8_engine.get_names()
        elif opt.detection_engine == "yolonas":
            self.yolonas_engine = YoloNasEngine(opt.weights, self.device, opt.classes, opt.confidence, opt.overlap, opt.agnostic_nms, opt.augment, self.half)
            self.names = self.yolonas_engine.get_names()

        elif opt.detection_engine == "clip":
            self.names = self.model.labels
        
        else:
            raise(f'{opt.detection_engine} detection engine not found')
        
        # initialize trackerself.tracker = Tracker(self.metric)
        
        self.weights, self.view_img, self.save_txt, self.imgsz = opt.weights, opt.view_img, opt.save_txt, opt.img_size

        # Initialize
        self.device = select_device(self.device)
        self.half =self. device.type != 'cpu'  # half precision only supported on CUDA

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None

        self.frames_detection_df = pd.DataFrame(columns=["frame", "track", "class", "bbox"])
        

    def detect(self, source, project, name, save_img=True):

        t0 = time_synchronized()
         # Directories
        save_dir = Path(increment_path(Path(project) / name,
                        exist_ok=self.exist_ok))  # increment run
        (save_dir / 'labels' if self.save_txt else save_dir).mkdir(parents=True,
                                                            exist_ok=True)  # make dir
        dataset = LoadImages(source, img_size=self.imgsz, detection_engine=self.opt.detection_engine)
        self.frame_count = 0
        if self.opt.detection_engine != "yolonas":
            img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        else:
            img = np.zeros((224,224,3))

        if self.opt.detection_engine == "yolov8":
            _ = self.yolov8_engine.infer(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        
        if self.opt.detection_engine == "yolonas":
            _ = self.yolonas_engine.infer(img) if self.device.type != 'cpu' else None  # run once

        for path, img, im0s, vid_cap in tqdm(dataset):
            
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            # choose between prediction engines 
            if self.opt.detection_engine == "yolov8":
                img, pred = self.yolov8_engine.infer(img)
            elif self.opt.detection_engine == "yolonas":
                pred = self.yolonas_engine.infer(img)
            elif self.opt.detection_engine == "clip":
                pred = self.model.detect(img, patch_size=64)
            else:
                raise Exception('')

            # Process detections

            for i, det in enumerate(pred):  # detections per image
               
                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg

                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
               
                if len(det):
                    # Print results
                    # Rescale boxes from img_size to im0 size
                    if self.opt.detection_engine == "yolov8": 
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Transform bboxes from tlbr to tlwh
                        trans_bboxes = det[:, :4].clone()
                        trans_bboxes[:, 2:] -= trans_bboxes[:, :2]
                        bboxes = trans_bboxes[:, :4].cpu()

                        confs = det[:, 4]
                        class_nums = det[:, -1].cpu()
                        classes = class_nums
                    elif self.opt.detection_engine == "yolonas":
                        bboxes = det[0]
                        confs = det[1]
                        classes = det[2]
    
                    elif self.opt.detection_engine == "clip":
                        bboxes = det[:, :4].cpu()
                        confs = det[:, 4]
                        classes = det[:, -1].cpu()
                    
                    # encode yolo detections and feed to tracker
                    features, clf_dict = self.encoder(im0, bboxes)
                
                    if clf_dict is None:
                        detections = [Detection(bbox, conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                            bboxes, confs, classes, features)]
                    else: 
                        classes = [x['label'] for x in clf_dict]
                        confs = [x['score'] for x in clf_dict]
                        detections = [Detection(bbox, conf, class_num, feature) for bbox, conf, class_num, feature in zip(
                            bboxes, confs, classes, features)]
                    
                    if clf_dict is not None:
                        class_nums = np.array(classes)
                    else:
                        class_nums = np.array([d.class_num for d in detections])
                    
                    # Call the tracker
                    self.tracker.predict()
                    self.tracker.update(detections)

                    # update tracks
                    im0 = self.update_tracks(save_img, im0, gn)

                # Stream results
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video'
                        if self.vid_path != save_path:  # new video
                            self.vid_path = save_path
                            if isinstance(self.vid_writer, cv2.VideoWriter):
                                self.vid_writer.release()  # release previous video writer

                            fourcc = 'mp4v'  # output video codec
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            self.vid_writer = cv2.VideoWriter(
                                save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

                self.frame_count = self.frame_count+1

        if self.save_txt:
            base_path= os.path.join(save_dir, 'labels')
            if not os.path.isdir(base_path):
                os.makedirs(base_path)
            logging.info('calculating jaccard coefficient for consecutive frames')
            jaccard_df, pause_range = jaccard_consecutive_frames(save_dir,p.name, self.frames_detection_df, pause_th = self.opt.pause_th)
          
            self.frames_detection_df = self.frames_detection_df.merge(jaccard_df, how="outer", on=["frame", "track"])
            df_path = os.path.join(base_path, f'full_labels.csv')
            self.frames_detection_df.to_csv(df_path)
            logging.info('Marking pauses on video')
            marking_pauses_on_video(source,self.vid_writer,self.frames_detection_df)
            if self.opt.animate:
                logging.info('Creating animation')
                save_animation = save_path.split('.')[0] + "_animation.mp4"
                jaccard_animation(save_path, df_path,save_animation, pause_range, animate_over=self.opt.animate_over)
            self.frames_detection_df = pd.DataFrame(columns=["frame", "track", "class", "bbox"])

           

            if self.opt.info: 
                print(f"Results saved to {save_dir}{s}")
        if self.opt.info:
            print(f'Done. ({time.time() - t0:.3f}s)')

    def update_tracks(self, save_img, im0, gn):
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if self.opt.detection_engine == "yolov8":
                xyxy = track.to_tlbr()
            elif self.opt.detection_engine == "yolonas":
                xyxy = track.to_tlwh()
            class_num = track.class_num
            bbox = xyxy
            
            if isinstance(class_num,str):
                class_name = class_num
            else:
                class_name = names[int(class_num)] if opt.detection_engine == "yolov7" or "yolov8" else class_num
            if self.opt.info:
                print("Tracker ID: {}, Class: {}, BBox Coords (xmin, ymin, xmax, ymax): {}".format(
                    str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            if self.save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                entry = pd.DataFrame.from_dict({
                "frame": [self.frame_count],
                "track":  [track.track_id],
                "class": [class_num],
                "bbox": [(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))],
                })

                self.frames_detection_df = pd.concat([self.frames_detection_df, entry], ignore_index=True)
                
            if save_img or self.view_img:  # Add bbox to image
                label = f'{class_name} #{track.track_id}'
                im0 = plot_one_box(xyxy, im0, label=label,
                            color=get_color_for(label), line_thickness=self.opt.thickness)
        return im0

    def __call__(self, source, project, name, save_img=True) -> None:
        self.tracker = Tracker(self.metric)
        self.detect(source, project, name, save_img=save_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='models/yolov8x.pt', help='model.pt path(s)')
    parser.add_argument('--names', type=str,
                        default='coco.names', help='yolov8 names file, file path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--confidence', type=float,
                        default=0.40, help='object confidence threshold')
    parser.add_argument('--overlap', type=float,
                        default=0.30, help='IOU threshold for NMS')
    parser.add_argument('--thickness', type=int,
                        default=3, help='Thickness of the bounding box strokes')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--nms_max_overlap', type=float, default=1.0,
                        help='Non-maxima suppression threshold: Maximum detection overlap.')
    parser.add_argument('--max_cosine_distance', type=float, default=0.4,
                        help='Gating threshold for cosine distance metric (object appearance).')
    parser.add_argument('--nn_budget', type=int, default=None,
                        help='Maximum size of the appearance descriptors allery. If None, no budget is enforced.')
    parser.add_argument('--info', action='store_true',
                        help='Print debugging info.')
    parser.add_argument("--detection-engine", default="yolov8", help="Which engine you want to use for object detection (yolov8.")
    parser.add_argument("--clip-labels", nargs='+', default=["hummingbird", "flower", "leaves", "feeder"])
    parser.add_argument('--pause_th', type=float,
                        default=0.2, help='pause marking threshold, represents a percentage from the mean (mean - mean*pause_th)')
    parser.add_argument('--animate', action='store_true',
                        help='animate results')
    parser.add_argument('--animate_over', type=str,
                        default='iou', help='choose over which variable animate: iou or centroide')
    opt = parser.parse_args()
    print(opt)
   
    video_detection = DetectionAndTracking(opt)
    with torch.no_grad():  
        video_detection(opt.source, opt.project, opt.name, save_img=True)
        

