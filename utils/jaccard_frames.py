import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval

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

def jaccard_consecutive_frames(project_path, file_name, df):
    color_palette = plt.cm.get_cmap('Pastel1', 50)
    try:
        df = df[df['class'].isin([" class: hummingbird"])==True]
        df = df.reset_index(drop=True)
        #df = df.replace({"^\s*|\s*$":""}, regex=True)
        tracks = list(np.unique(df['track'].values))
        jaccard_distance = []
        for track in tracks:
            df_track = df[df['track']==track]
            df_track = df_track.reset_index(drop=True)
            for idx in range(len(df_track)):
                frame = int(df_track['frame'][idx].split(': ')[-1])
                bbox_A=literal_eval(df_track['bbox'][idx].split(': ')[-1])
                bbox_B=literal_eval(df_track['bbox'][idx+1].split(': ')[-1])
                iou = bb_intersection_over_union(bbox_A, bbox_B)
                jaccard_distance.append((track.split(': ')[-1], frame, iou))
                if idx>=len(df_track)-2:
                    break
        jaccard_df = pd.DataFrame(jaccard_distance, columns=["track", "frame", "iou"])
        fig, ax = plt.subplots()
    
        for key, grp in jaccard_df.groupby(['track']):
            ax = grp.plot(ax=ax, kind='line', x='frame', y='iou', c=np.random.rand(3,), label=key)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        plt.savefig(os.path.join(project_path,f"iou_{file_name}.png"))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(file_name)
    except:
        pass