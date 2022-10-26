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

class ClevrChangeClassificationDataset(Dataset):

    def __init__(self, data_dir, split, config, transform):
        super().__init__()
        assert split in ['train', 'val']

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
            text = self.captions[f"CLEVR_default_{img_id:06d}.png"][-1]
            # get two different images
            image0_file = os.path.join(data_dir, "data", "nsc_images", f"CLEVR_nonsemantic_{img_id:06d}.png")
            image1_file = os.path.join(data_dir, "data", "sc_images", f"CLEVR_semantic_{img_id:06d}.png")
            # print(text, image0_file, image1_file)
            if random.rand() > 0.5:
                target = 1
                images = [image0_file,image1_file]
            else:
                target = 0
                images = [image1_file,image0_file]
            
            dataset.append((images, text, target))
        

        return dataset
    
    def __getitem__(self, idx):
        (image1_file, image0_file), text, target = self.data[idx] 
        image_0 = self.transform(Image.open(image0_file).convert('RGB'))
        image_1 = self.transform(Image.open(image1_file).convert('RGB'))
        img = torch.stack([image_0, image_1], dim=0)
        return img, text, target
    
    def __len__(self):
        return len(self.data)