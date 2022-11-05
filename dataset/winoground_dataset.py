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

class WinogroundClassificationDataset(Dataset):

    def __init__(self, transform, split, debug=False):
        data_dir = '/home/mila/b/benno.krojer/scratch/winoground'
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.debug = debug
        
        self.data = self.load_data(data_dir, split)

    def load_data(self, data_dir, split):
        split_file = os.path.join(data_dir, "annotations", f'{split}.json')
        with open(split_file) as f:
            json_file = json.load(f)

        dataset = []
        for i, row in tqdm(enumerate(json_file), total=len(json_file)):
            img_id = row["id"]
            caption0 = row["caption_0"]
            caption1 = row["caption_1"]
            # get two different images
            img0 = os.path.join(data_dir, "images", f'ex_{img_id}_img_0.jpg')
            img1 = os.path.join(data_dir, "images", f'ex_{img_id}_img_1.jpg')
            dataset.append((img0, img1, caption0, caption1))
        if self.debug:
            dataset = dataset[:120]

        return dataset
    
    def __getitem__(self, idx):
        img0, img1, text0, text1 = self.data[idx]
        image0 = self.transform(Image.open(img0).convert('RGB'))
        image1 = self.transform(Image.open(img1).convert('RGB'))

        return image0, image1, text0, text1

    def __len__(self):
        return len(self.data)