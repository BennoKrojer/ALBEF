import json
import os
from tqdm import tqdm
import requests
from pytube import YouTube
import traceback
from moviepy.editor import *


jsonl = open('/home/mila/b/benno.krojer/scratch/moment-retrieval/highlight_val_release.jsonl', 'r').readlines()
i = 0
for line in tqdm(jsonl):
    d = json.loads(line)
    video = d['vid'].split('_')
    start = video[-2]
    end = video[-1]
    video_id = '_'.join(video[:-2])
    start = int(float(start))
    end = int(float(end))
    if os.path.exists('/home/mila/b/benno.krojer/scratch/moment-retrieval/video_segments/'+video_id+'_'+str(start)+'_'+str(end)+'.mp4'):
        continue
    # url = f'https://www.youtube.com/watch?v={video_id}?start={start}&end={end}&version=3'
    url = f'https://www.youtube.com/watch?v={video_id}&version=3'
    if not os.path.exists('/home/mila/b/benno.krojer/scratch/moment-retrieval/videos/'+video_id+'.mp4'):
        try:
            yt = YouTube(url)
            yt.streams.filter(file_extension='mp4').first().download(output_path='/home/mila/b/benno.krojer/scratch/moment-retrieval/videos', filename=video_id+'.mp4')
        except Exception:
            traceback.print_exc()
            print(f'Failed to download {url}')
            continue
    # cut video

    # os.system('ffmpeg -i /home/mila/b/benno.krojer/scratch/moment-retrieval/videos/'+video_id+'.mp4 -ss '+str(start)+' -to '+str(end)+' -c copy /home/mila/b/benno.krojer/scratch/moment-retrieval/video_segments/'+video_id+'_'+str(start)+'_'+str(end)+'.mp4')

    # video = VideoFileClip('/home/mila/b/benno.krojer/scratch/moment-retrieval/videos/'+video_id+'.mp4')#.subclip(start+1, end-1)
    # video = video.subclip(start, end)
    # video.write_videofile('/home/mila/b/benno.krojer/scratch/moment-retrieval/videos/'+video_id+'_'+str(start)+'_'+str(end)+'.mp4', fps=1)

    # os.system(f'rm /home/mila/b/benno.krojer/scratch/moment-retrieval/videos/{video_id}.mp4')