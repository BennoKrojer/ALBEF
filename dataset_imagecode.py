import json
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset
from torchvision import transforms
# from transformers import BertTokenizerFast
from PIL import Image

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
# pretrain_transform = transforms.Compose([                        
#         transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
#         transforms.RandomHorizontalFlip(),
#         RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
#                                             'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
#         transforms.ToTensor(),
#         normalize,
#     ])    
# train_transform = transforms.Compose([                        
#         transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
#         transforms.RandomHorizontalFlip(),
#         RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
#                                             'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
#         transforms.ToTensor(),
#         normalize,
#     ])  
# test_transform = transforms.Compose([
#     transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
#     transforms.ToTensor(),
#     normalize,
#     ])   

class ImageCoDeDataset(Dataset):

    def __init__(self, data_dir, split, config, image_transform=None, text_transform=None, video_only=False, quarters=False):
        super().__init__()
        assert split in ['train', 'valid']
        self.quarters = quarters

        if image_transform is not None:
            self.image_transform = image_transform
        else:
            self.image_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])
        
        # if text_transform is not None:
        self.text_transform = text_transform
        # else:
        #     self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        #     self.text_transform = partial(default_text_transform, tokenizer=self.tokenizer)

        self.data = self.load_data(Path(data_dir), '/network/scratch/b/benno.krojer/dataset/games', split, video_only)

    @staticmethod
    def load_data(data_dir, img_path, split, video_only=False):
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
                else:
                    dataset.append((img_dir, img_files, int(img_idx), text))
        
        return dataset
    
    def __getitem__(self, idx):
        img_dir, img_files, img_idx, text = self.data[idx]
        
        images = [Image.open(img_file).convert('RGB') for img_file in img_files]

        if self.quarters:
            all_images = []
            for img in images:
                img_h = img.size[1] // 2
                img_w = img.size[0] // 2
                img1 = img.crop((0, 0, img_w, img_h))
                img2 = img.crop((img_w, 0, img_w * 2, img_h))
                img3 = img.crop((0, img_h, img_w, img_h * 2))
                img4 = img.crop((img_w, img_h, img_w * 2, img_h * 2))
                all_images = all_images + [img, img1, img2, img3, img4]
            images = all_images

        images = [self.image_transform(img) for img in images]
        img = torch.stack(images, dim=0)
        
        # txt = self.text_transform(text)
        is_video = torch.tensor(1 if 'open-images' not in img_dir else 0)
        
        return img, text, img_idx, is_video, img_dir
    
    def __len__(self):
        return len(self.data)