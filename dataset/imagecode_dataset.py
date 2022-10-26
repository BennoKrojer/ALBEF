import json
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from dataset.utils import pre_caption

# from transformers import BertTokenizerFast
from PIL import Image

class PairedImageCoDeDataset(Dataset):

    def __init__(self, transform, data_dir, split, video_only=False, max_words=40, debug=False, static_only=False):
        super().__init__()
        self.transform = transform
        self.max_words = max_words
        image_root = '/network/scratch/b/benno.krojer/dataset/games'
        self.debug = debug
        self.data = self.load_data(Path(data_dir), image_root, split, video_only, static_only)

    def load_data(self, data_dir, img_path, split, video_only=False, static_only=False):
        with open(data_dir / f'{split}_data.json') as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                img_idx = int(img_idx)
                static = 'open-images' in img_dir
                target_file = img_files[img_idx]
                for i, file in enumerate(img_files):
                    if i == img_idx:
                        continue
                    if i < img_idx:
                        pair = [target_file, file]
                        label = 0
                    else:
                        pair = [file, target_file]
                        label = 1
                    if video_only:
                        if not static:
                            dataset.append((img_dir, pair, label, text))
                    elif static_only:
                        if static:
                            dataset.append((img_dir, pair, label, text))
                    else:
                        dataset.append((img_dir, pair, label, text))
        if self.debug:
            dataset = dataset[:100]
        return dataset
    
    def __getitem__(self, idx):
        img_dir, pair, label, text = self.data[idx]
        
        images = [self.transform(Image.open(img_file).convert('RGB')) for img_file in pair]
        img1, img2 = images
        sentence = pre_caption(text, self.max_words)

        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img1, img2, sentence, label, is_video
    
    def __len__(self):
        return len(self.data)

class ImageCoDeDataset(Dataset):

    def __init__(self, transform, data_dir, split, video_only=False, max_words=40, debug=False, static_only=False):
        super().__init__()
        self.transform = transform
        self.max_words = max_words
        image_root = '/network/scratch/b/benno.krojer/dataset/games'
        self.debug = debug
        self.data = self.load_data(Path(data_dir), image_root, split, video_only, static_only)

    def load_data(self, data_dir, img_path, split, video_only=False, static_only=False):
        with open(data_dir / f'{split}_data.json') as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                if video_only:
                    if not static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                elif static_only:
                    if static:
                        dataset.append((img_dir, img_files, int(img_idx), text))
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))
        
        if self.debug:
            dataset = dataset[:100]
        return dataset
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.data[idx]
        target_file = img_files[img_idx]

        target_img = self.transform(Image.open(target_file).convert('RGB'))

        img0 = []
        img1 = []
        labels = []

        for i, file in enumerate(img_files):
            img = self.transform(Image.open(file).convert('RGB'))
            if i == img_idx:
                continue
            if i < img_idx:
                img0.append(target_img)
                img1.append(img)
                labels.append(torch.tensor(0))
            else:
                img0.append(img)
                img1.append(target_img)
                labels.append(torch.tensor(1))
        
        img0 = torch.stack(img0)
        img1 = torch.stack(img1)
        labels = torch.stack(labels)
        sentence = pre_caption(text, self.max_words)
        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img0, img1, sentence, labels, is_video, img_dir
    
    def __len__(self):
        return len(self.data)

class InferenceImageCoDeDataset(Dataset):

    def __init__(self, transform, data_dir, split, video_only=False, max_words=40, debug=False, static_only=False):
        super().__init__()
        self.transform = transform
        self.max_words = max_words
        image_root = '/network/scratch/b/benno.krojer/dataset/games'
        self.debug = debug
        self.data = self.load_data(Path(data_dir), image_root, split, video_only, static_only)

    def load_data(self, data_dir, img_path, split, video_only=False, static_only=False):
        with open(data_dir / f'{split}_data.json') as f:
            json_file = json.load(f)

        dataset = []
        for img_dir, data in json_file.items():
            img_files = list((Path(f'{img_path}/{img_dir}')).glob('*.jpg'))
            img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
            for img_idx, text in data.items():
                static = 'open-images' in img_dir
                img0 = []
                img1 = []
                pairs = []
                for i in range(len(img_files)):
                    for j in range(i+1, len(img_files)):
                        f1 = img_files[i]
                        f2 = img_files[j]
                        img0.append(f1)
                        img1.append(f2)
                        pairs.append((i,j))

                if video_only:
                    if not static:
                        dataset.append((img_dir, img0, img1, pairs, int(img_idx), text))
                elif static_only:
                    if static:
                        dataset.append((img_dir, img0, img1, pairs, int(img_idx), text))
                else:
                    dataset.append((img_dir, img0, img1, pairs, int(img_idx), text))
        if self.debug:
            dataset = dataset[:100]
        return dataset
    
    def __getitem__(self, idx):
        img_dir, img0, img1, pairs, img_idx, text = self.data[idx]
        img0 = [self.transform(Image.open(f).convert('RGB')) for f in img0]
        img1 = [self.transform(Image.open(f).convert('RGB')) for f in img1]
        img0 = torch.stack(img0)
        img1 = torch.stack(img1)

        sentence = pre_caption(text, self.max_words)
        
        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img0, img1, pairs, sentence, img_idx, is_video, img_dir
    
    def __len__(self):
        return len(self.data)
