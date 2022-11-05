from operator import is_
import os
import json
from pathlib import Path
from functools import partial
from tqdm import tqdm
from numpy import random


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizerFast
from PIL import Image

class MomentClassificationDataset(Dataset):

    def __init__(self, transform, split, debug=False):
        data_dir = '/home/mila/b/benno.krojer/scratch/moment-retrieval'
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.debug = debug
        
        self.data = self.load_data(data_dir, split)

    def load_data(self, data_dir, split):
        split_file = os.path.join(data_dir, f'{split}.json')
        with open(split_file) as f:
            json_file = json.load(f)

        dataset = []
        for i, row in tqdm(enumerate(json_file), total=len(json_file)):
            if self.debug and i > 130:
                break
            pos_id = str(row['pos_frame'])
            neg_id = str(row['neg_frame'])
            sentence = row['query']
            video = row['video']
            # get two different images
            pos_file = os.path.join(data_dir, "frames", video, pos_id+".jpg")
            neg_file = os.path.join(data_dir, "frames", video, neg_id+".jpg")
            dataset.append((pos_file, neg_file, sentence, 0))
            dataset.append((neg_file, pos_file, sentence, 1))

        return dataset
    
    def __getitem__(self, idx):
        file0, file1, text, target = self.data[idx]
        image0 = self.transform(Image.open(file0).convert('RGB'))
        image1 = self.transform(Image.open(file1).convert('RGB'))

        return image0, image1, text, target, 1, ''


    
    def __len__(self):
        return len(self.data)