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

class ClevrClassificationDataset(Dataset):

    def __init__(self, transform, split, debug=False):
        data_dir = '/home/mila/b/benno.krojer/scratch/clevr_change'
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.debug = debug
        self.captions =  self.load_captions(data_dir)
        self.data = self.load_data(data_dir, split)
        self.transform = transform

    def load_captions(self, data_dir):
        fname = os.path.join(data_dir, "data", "change_captions.json")
        with open(fname) as f:
            captions = json.load(f)
        return captions


    def load_data(self, data_dir, split):
        split_file = os.path.join(data_dir, "data", 'splits.json')
        with open(split_file) as f:
            json_file = json.load(f)
        
        data = json_file[split]

        dataset = []
        for i, img_id in tqdm(enumerate(data), total=len(data)):
            for text in self.captions[f"CLEVR_default_{img_id:06d}.png"]:
                # get two different images
                image0_file = os.path.join(data_dir, "data", "nsc_images", f"CLEVR_nonsemantic_{img_id:06d}.png")
                image1_file = os.path.join(data_dir, "data", "sc_images", f"CLEVR_semantic_{img_id:06d}.png")
                
                dataset.append(([image0_file,image1_file], text, 1))
                dataset.append(([image1_file,image0_file], text, 0))
        
        if self.debug:
            dataset = dataset[:120]
        return dataset

    
    def __getitem__(self, idx):
        img, text, target = self.data[idx]
        file0, file1 = img
        image0 = self.transform(Image.open(file0).convert('RGB'))
        image1 = self.transform(Image.open(file1).convert('RGB'))

        return image0, image1, text, target, 1, ''
    
    def __len__(self):
        return len(self.data)