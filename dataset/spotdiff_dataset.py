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

class SpotdiffClassificationDataset(Dataset):

    def __init__(self, transform, split, debug=False):
        data_dir = '/home/mila/b/benno.krojer/scratch/spotdiff'
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
            img_id = row["img_id"]
            text = row["sentences"]
            for sent in text:
                # get two different images
                image0_file = os.path.join(data_dir, "resized_images", img_id+".png")
                image1_file = os.path.join(data_dir, "resized_images", img_id+"_2.png")
                dataset.append((image0_file, image1_file, sent, 1))
                dataset.append((image1_file, image0_file, sent, 0))
                # img = torch.stack(images, dim=0)
        if self.debug:
            dataset = dataset[:120]

        return dataset
    
    def __getitem__(self, idx):
        file0, file1, text, target = self.data[idx]
        image0 = self.transform(Image.open(file0).convert('RGB'))
        image1 = self.transform(Image.open(file1).convert('RGB'))

        return image0, image1, text, target, 1, ''


    
    def __len__(self):
        return len(self.data)