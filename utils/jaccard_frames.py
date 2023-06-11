import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from pathlib import Path

vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def get_color_for(class_num):
    colors = [
        "#4892EA",
        "#00EEC3",
        "#FE4EF0",
        "#F4004E",
        "#FA7200",
        "#EEEE17",
        "#90FF00",
        "#78C1D2",
        "#8C29FF"
    ]

    num = hash(class_num) # may actually be a number or a string
    hex = colors[num%len(colors)]

    # adapted from https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    rgb = tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    return rgb

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def jaccard_consecutive_frames(project_path, file_name, df, pause_th = 0.01):
    df = df[df['class'].isin(["hummingbird"])==True]
    df = df.reset_index(drop=True)
    tracks = list(np.unique(df['track'].values))
    
    jaccard_distance = []
    for track in tracks:
        df_track = df[df['track']==track]
        df_track = df_track.reset_index(drop=True)
        iou_anterior = 0
        for idx in range(len(df_track)):
            frame = int(df_track['frame'][idx]) 
            bbox_A=df_track['bbox'][idx]
            bbox_B=df_track['bbox'][idx+1]
            iou_actual = bb_intersection_over_union(bbox_A, bbox_B)
            variance = abs(iou_actual - iou_anterior)
            iou_anterior = iou_actual
            pause = 1 if variance <= pause_th else np.nan
            jaccard_distance.append((track, frame, iou_actual, variance, pause))
            if idx>=len(df_track)-2:
                break
    jaccard_df = pd.DataFrame(jaccard_distance, columns=["track", "frame", "iou", "variance", "pause"])

    for key, grp in jaccard_df.groupby(['track']):
        fig, ax = plt.subplots()
        color_label = f'hummingbird #{key}'
        color_track_bgr = np.array(list(get_color_for(color_label)))/255

        ax = grp.plot(ax=ax, kind='line', x='frame', y='iou', c=color_track_bgr, label=key)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if not os.path.isdir(os.path.join(project_path,"figures")):
            os.makedirs(os.path.join(project_path,"figures"))

        plt.savefig(os.path.join(project_path,f"figures/iou_{file_name}_{color_label}.png"))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(file_name)
    return jaccard_df