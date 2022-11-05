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
from glob import glob

class NaturalistClassificationDataset(Dataset):

    def __init__(self, transform, split, debug=False):
        data_dir = '/home/mila/b/benno.krojer/scratch/neural-naturalist'
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
            if self.debug and i > 100:
                break
            img0 = row["img1_id"]
            img1 = row["img2_id"]
            img0 = f'{data_dir}/images/{img0}'
            img1 = f'{data_dir}/images/{img1}'
            dataset.append(([img0,img1], row['description'], 0))
            dataset.append(([img1,img0], row['description'], 1))
        
        return dataset
    
    def __getitem__(self, idx):
        img, text, target = self.data[idx]
        file0, file1 = img
        image0 = self.transform(Image.open(file0).convert('RGB'))
        image1 = self.transform(Image.open(file1).convert('RGB'))

        return image0, image1, text, target, 1, ''


    
    def __len__(self):
        return len(self.data)