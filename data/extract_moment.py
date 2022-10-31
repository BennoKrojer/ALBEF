import json
import os
from tqdm import tqdm
import requests
import traceback
from glob import glob
from decord import VideoReader
import cv2

#extract 1 frame per second

for file in tqdm(glob('/home/mila/b/benno.krojer/scratch/moment-retrieval/video_segments/*.mp4')):
    name = file.split('/')[-1][:-4]
    # if os.path.exists('/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/'+name+'/'+'1.jpg'):
    #     continue

    vr = VideoReader(file)
    # get frame rate of video
    fps = vr.get_avg_fps()
    # round frame rate to nearest integer
    fps = int(round(fps))
    for i in range(len(vr)):
        if i % fps == 0:
            frame = vr[i].asnumpy() 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            os.makedirs('/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/'+name, exist_ok=True)
            print("writing frame", i, "of", len(vr), "to", '/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/'+name+'/'+str(i)+'.jpg')
            cv2.imwrite('/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/'+name+'/'+str(i//30)+'.jpg', frame)