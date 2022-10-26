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

    def __init__(self, data_dir, split, config, transform):
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.data = self.load_data(data_dir, split)

    def load_data(self, data_dir, split):
        split_file = os.path.join(data_dir, "annotations", f'{split}.json')
        with open(split_file) as f:
            json_file = json.load(f)

        dataset = []
        for i, row in tqdm(enumerate(json_file), total=len(json_file)):
            img_id = row["img_id"]
            text = row["sentences"]
            text = ". ".join(text)
            # get two different images
            image0_file = os.path.join(data_dir, "resized_images", img_id+".png")
            image1_file = os.path.join(data_dir, "resized_images", img_id+"_2.png")
            if random.rand() > 0.5:
                target = 1
                images = [image0_file,image1_file]
            else:
                target = 0
                images = [image1_file,image0_file]
            
            img = torch.stack(images, dim=0)
            dataset.append((img, text, target))

        return dataset
    
    def __getitem__(self, idx):
        img, text, target, is_video = self.data[idx]
        file0, file1 = img
        image0 = self.transform(Image.open(file0).convert('RGB'))
        image1 = self.transform(Image.open(file1).convert('RGB'))
        imgs = [image0, image1]
        return imgs, text, target


    
    def __len__(self):
        return len(self.data)