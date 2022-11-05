import json
import os
from tqdm import tqdm
import requests
from pytube import YouTube
import traceback
from decord import VideoReader
import cv2
import sys

split = sys.argv[1]

jsonl = open(f'/home/mila/b/benno.krojer/scratch/moment-retrieval/highlight_{split}_release.jsonl', 'r').readlines()

dataset = []

for line in tqdm(jsonl):
    d = json.loads(line)
    video_str = d['vid']
    video = video_str.split('_')
    start = video[-2]
    end = video[-1]
    video_id = '_'.join(video[:-2])
    start = int(float(start))
    end = int(float(end))

    query = d['query']
    windows = d['relevant_windows']

    inside = []
    outside = []
    not_allowed = []
    for window in windows:
        begin_window = window[0] + start
        end_window = window[1] + start
        inside += list(range(begin_window+2, end_window-1))
        not_allowed += list(range(begin_window-1, end_window+2))
    
    for frame in range(start, end+1):
        if frame in not_allowed:
            continue
        outside.append(frame)
    
    print(inside)
    print(outside)
    for pos_frame in inside:
        for neg_frame in outside:
            #check if jpg exists
            if os.path.exists(f'/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/{video_str}/{pos_frame}.jpg') and os.path.exists(f'/home/mila/b/benno.krojer/scratch/moment-retrieval/frames/{video_str}/{neg_frame}.jpg'):
                example = {
                    'query': query,
                    'pos_frame': pos_frame,
                    'neg_frame': neg_frame,
                    'video': video_str
                }
                dataset.append(example)

print(len(dataset), "examples")

with open(f'/home/mila/b/benno.krojer/scratch/moment-retrieval/{split}.json', 'w') as f:
    json.dump(dataset, f, indent=4)