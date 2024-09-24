import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
import time
import glob
from pathlib import Path
from utils.plots import plot_one_box
from tqdm import tqdm

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
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def jaccard_consecutive_frames(project_path, file_name, df, pause_th= 0.9):
    df = df[df['class'].isin(["hummingbird"])==True]
    df = df.reset_index(drop=True)
    tracks = list(np.unique(df['track'].values))
    bbox = df['bbox'].values
    w = [(value[2]-value[0]) for value in bbox]
    h =  [(value[3]-value[1]) for value in bbox]

    cx = [(value[0] + w[idx]/2)/200 for idx, value in enumerate(bbox)]
    cy = [(value[1] + h[idx]/2)/200 for idx, value in enumerate(bbox)]

    magnitud_orig = [(cx[idx]**2 + cy[idx]**2)**(1/2) for idx in range (len(cx))]
    diff = list(np.diff(np.array(magnitud_orig))) + [0]
    
    jaccard_distance = []
    for track in tracks:
        df_track = df[df['track']==track]
        df_track = df_track.reset_index(drop=True)
        jaccard_distance.append(0)
        for idx in range(len(df_track)):

            bbox_A=df_track['bbox'][idx]
            bbox_B=df_track['bbox'][idx+1]
            iou_actual = bb_intersection_over_union(bbox_A, bbox_B)
           
            jaccard_distance.append(iou_actual)
            if idx>=len(df_track)-2:
                break
    smoth_iou = smooth(np.array(jaccard_distance),3)
    diff_iou = np.array(list(np.diff(smoth_iou)) + [0])

    init_frame = 0
    plat = []
    jump_frame=60

    for i in range(0, len(diff_iou), jump_frame):  # Step by 10
        chunk = diff_iou[init_frame:init_frame+jump_frame]
        
        if len(chunk) < jump_frame:
            # If the chunk has fewer than 10 elements, break the loop
            break

        mean = np.mean(np.abs(chunk))

        plat += [0.5 if 0 <= abs(value) <= mean*pause_th else 0.0 for value in chunk]
        
        init_frame += jump_frame  # Move to the next set of 10 values


    res = []
    idx = 0

    while idx < (len(plat)):
        strt_pos = idx
        val = plat[idx]

        # Getting last position
        while (idx < len(plat) and plat[idx] == val):
            idx += 1
        end_pos = idx - 1

        # Appending in format [element, start, end position]
        res.append((val, strt_pos, end_pos))
    
    
 
    jaccard_df=pd.DataFrame()

    jaccard_df['track'] = df['track']
    jaccard_df['frame'] = df['frame']
    jaccard_df['iou'] = jaccard_distance
    jaccard_df['diff_iou'] = diff_iou
    jaccard_df['cx'] = pd.Series(cx)
    jaccard_df['cy'] = pd.Series(cy)
    jaccard_df['magnitude_centroid'] = pd.Series(magnitud_orig)
    jaccard_df['dy_magnitud'] = diff
    # Check if the lengths are different
    if len(plat) < len(jaccard_df):
        # Pad the list with 0s to match the DataFrame length
        plat.extend([0] * (len(jaccard_df) - len(plat)))

    jaccard_df['dy_platteau'] = plat

    pause_range = [(init, fin) for (value, init, fin) in res if (fin-init>=10) and (value==0.5)]
    pause = np.array([None for _ in range(len(plat))])
    for (init, fin) in pause_range:
        pause[init : fin] = 1
    jaccard_df['pause'] = pause
    
    for key, grp in jaccard_df.groupby(['track']):
        fig, ax = plt.subplots()
        color_label = f'hummingbird #{key}'
        color_bgr = get_color_for(color_label)
        color_track_bgr = np.array(list((color_bgr[2],color_bgr[1],color_bgr[0])))/(255,255,255)

        ax = grp.plot(ax=ax, kind='line', x='frame', y='iou', c=color_track_bgr, label=key)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if not os.path.isdir(os.path.join(project_path,"figures")):
            os.makedirs(os.path.join(project_path,"figures"))
        pauses_frames = grp[grp['pause'].notna()]['frame'].values
        pauses_consec  = np.split(pauses_frames, np.where(np.diff(pauses_frames) != 1)[0]+1)
        pauses_range = []
        for consec_group in pauses_consec:
            try:
                pauses_range.append((consec_group[0],consec_group[-1]))
            except:
                pass
        for (init, fin) in pauses_range:
            plt.axvspan(init,fin, color='yellow', alpha=0.3, label ="pause detection")
        plt.savefig(os.path.join(project_path,f"figures/iou_{file_name}_{color_label}.png"))
        plt.legend(by_label.values(), by_label.keys())
        plt.title(file_name)
        plt.close()
    return jaccard_df, pause_range

def marking_pauses_on_video(video_path, vid_writer, data_frame, show_img=False):
    p = str(Path(video_path))  # os-agnostic
    p = os.path.abspath(p)  # absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception('ERROR: %s does not exist' % p)
    data_frame = data_frame[['frame','track','class','bbox','pause']]
    data_frame_group = data_frame.groupby(['frame'])

    videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
    cap = cv2.VideoCapture(videos[0])
    frame_count = 0
    org = (40, 40)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            
            try:
                group_content = data_frame_group.get_group(frame_count)
                for value in group_content.values.tolist():
                    _,track, class_label, bbox, pause = value
                    label = f'{class_label} #{track}'
                    
                    color = get_color_for(label)
                    
                    frame = plot_one_box(bbox, frame, label=label,
                            color=color, line_thickness=3)
                    
                    if class_label=='hummingbird' and pause == 1:
                        org = (50,50 + 10*track)
                        frame = cv2.putText(frame, f'Pause hummingbird {track}', org, font, 1, color, 3, cv2.LINE_AA)
            except Exception as e:
                pass
            
            vid_writer.write(frame)
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                break
            frame_count += 1
        else: break
    cap.release()
    vid_writer.release()  # release previous video writer



def reformating_path(str_path):
    p = str(Path(str_path))  # os-agnostic
    p = os.path.abspath(p)  # absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception('ERROR: %s does not exist' % p)

    videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
    return videos[0]

def jaccard_animation(video_path, df_path, save_animation, pause_range, animate_over='centroide'):

    df = pd.read_csv(df_path)
    df = df[df['class'].isin(["hummingbird"])==True]
    df = df.reset_index(drop=True)
    video_path = reformating_path(video_path)
    
    tracks = np.unique(df["track"].values)

    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fig, ax = plt.subplots(nrows=len(tracks)+1,ncols=1, figsize=(10,10))
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0/fps
    
    track_df_list, iou_list, dydx = [], [], []
    x_data = [ [] for _ in range(len(tracks)) ]
    y_data = [ [] for _ in range(len(tracks)) ]
   
    for track_idx, track in enumerate(tracks):
    
        color_label = f'hummingbird #{track}'
        ax[track_idx+1].set(xlim=[0, num_frames], ylim=[0, 5], xlabel='Frames', ylabel=color_label)

        track_df_list.append(df[df['track']==track][['frame','iou', 'dy_platteau']])#'iou', 'dy/dx']])
        #iou_temp = [[idx, 0] for idx in range(num_frames-1)  if idx not in track_df_list[track_idx]['frame'].values]  +  track_df_list[track_idx].values.tolist()
        #iou_temp = [iou_data[1] for iou_data in iou_temp]
        #iou_list.append(track_df_list[track_idx]['dy_platteau'])#(iou_temp)
        if animate_over=='iou':
            dydx.append([value for value in track_df_list[track_idx] ['iou']]) 
        elif animate_over=='centroide':
            dydx.append([value for value in track_df_list[track_idx] ['dy_platteau']])
        
    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = ax[0].imshow(frame)
  
    def animate(i):
        for track_idx, track in enumerate(tracks):
            try:
                color_label = f'hummingbird #{track}'
                color_bgr = get_color_for(color_label)
                color_track_bgr = np.array(list((color_bgr[2],color_bgr[1],color_bgr[0])))/(255,255,255)
                x_data[track_idx].append(i)
                #y_data[track_idx].append((iou_list[track_idx][i]))

                y_data[track_idx].append((dydx[track_idx][i]))
                
                for (init, fin) in pause_range:
                    plt.axvspan(init,fin, color='yellow', alpha=0.3)
                #dy_data[track_idx].append((dydx[track_idx][i]))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    im.set_array(frame)
                ax[track_idx+1].plot(x_data[track_idx], y_data[track_idx], color=color_track_bgr)
                #ax[track_idx+1].plot(x_data[track_idx], dy_data[track_idx], color='red')
                if animate_over=='iou':
                    ax[track_idx+1].legend(['iou'])
                elif animate_over=='centroide':
                    ax[track_idx+1].legend(['centroide'])
            except:
                pass
        time.sleep(frame_time)
        return ax, im,

    ani = FuncAnimation(fig, func=animate, frames=num_frames, interval=frame_time)
    Writer = writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(save_animation,writer=writer)
    plt.close()
    cap.release()