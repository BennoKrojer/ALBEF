import json
import os
from tqdm import tqdm
import requests
from pytube import YouTube
import traceback
from decord import VideoReader
import cv2


jsonl = open('/home/mila/b/benno.krojer/scratch/moment-retrieval/highlight_train_release.jsonl', 'r').readlines()
for line in tqdm(jsonl):
    d = json.loads(line)
    video_str = d['vid']
    video = video_str.split('_')
    start = video[-2]
    end = video[-1]
    video_id = '_'.join(video[:-2])
    start = int(float(start))
    end = int(float(end))
    # cut video
    try:
        vr = VideoReader('/home/mila/b/benno.krojer/scratch/moment-retrieval/videos/'+video_id+'.mp4')
        fps = vr.get_avg_fps()
        # round frame rate to nearest integer
        fps = int(round(fps))
        if fps == 30:
            print('fps is 30', video_str)
            continue
        start = start * fps
        end = end * fps
        for i in range(len(vr)):
            if i % fps == 0 and i >= start and i <= end:
                frame = vr[i].asnumpy() 
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                os.makedirs('/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/'+video_str, exist_ok=True)
                print("writing frame", i, "of", len(vr), "to", '/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/'+video_str+'/'+str(i//fps)+'.jpg')
                cv2.imwrite('/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/'+video_str+'/'+str(i//fps)+'.jpg', frame)
    except:
        print("error with", video_id)
        traceback.print_exc()