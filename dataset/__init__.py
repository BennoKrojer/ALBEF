import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset.nlvr_dataset import nlvr_dataset
from dataset.imagecode_dataset import ImageCoDeDataset, PairedImageCoDeDataset, InferenceImageCoDeDataset
from dataset.spotdiff_dataset import SpotdiffClassificationDataset
from dataset.clevr_change_dataset import ClevrClassificationDataset
from dataset.img_edit_dataset import ImgEditClassificationDataset
from dataset.moment_dataset import MomentClassificationDataset
from dataset.naturalist_dataset import NaturalistClassificationDataset
from dataset.svo_dataset import SVOClassificationDataset
from dataset.randaugment import RandomAugment

def create_dataset(dataset, config, fullset=False):
    
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
    pretrain_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])    
    train_transform = transforms.Compose([                        
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
            transforms.ToTensor(),
            normalize,
        ])  
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])   

    if dataset=='nlvr':   
        train_dataset = nlvr_dataset('train', train_transform)  
        val_dataset = nlvr_dataset('dev', test_transform)  
        # test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])                
        return train_dataset, val_dataset
               
    elif dataset=='imagecode':
        if config['random_pair_sampling']:
            train_dataset = PairedImageCoDeDataset(train_transform, '../imagecode/data', 'train', video_only=config['video_only'], debug=config['debug'])
        else:
            train_dataset = ImageCoDeDataset(train_transform, '../imagecode/data', 'train', video_only=config['video_only'], debug=config['debug'])
        val_dataset = PairedImageCoDeDataset(test_transform, '../imagecode/data','valid', video_only=config['video_only'], debug=config['debug'])
        # inference_val_dataset = InferenceImageCoDeDataset(test_transform, '../imagecode/data','valid', video_only=config['video_only'], debug=config['debug'])
        return train_dataset, val_dataset#, inference_val_dataset
    
    elif dataset=='spotdiff':
        train_dataset = SpotdiffClassificationDataset(train_transform, 'train', debug=config['debug'])
        val_dataset = SpotdiffClassificationDataset(test_transform, 'val', debug=config['debug'])
        return train_dataset, val_dataset

    elif dataset == 'clevr':
        train_dataset = ClevrClassificationDataset(train_transform, 'train', debug=config['debug'])
        val_dataset = ClevrClassificationDataset(test_transform, 'val', debug=config['debug'])
        return train_dataset, val_dataset

    elif dataset == 'img-edit':
        train_dataset = ImgEditClassificationDataset(train_transform, 'train', debug=config['debug'])
        val_dataset = ImgEditClassificationDataset(test_transform, 'val', debug=config['debug'])
        return train_dataset, val_dataset
    
    elif dataset == 'moment':
        train_dataset = MomentClassificationDataset(train_transform, 'train', debug=config['debug'])
        val_dataset = MomentClassificationDataset(test_transform, 'val', debug=config['debug'])
        return train_dataset, val_dataset

    elif dataset == 'naturalist':
        train_dataset = NaturalistClassificationDataset(train_transform, 'train', debug=config['debug'])
        val_dataset = NaturalistClassificationDataset(test_transform, 'val', debug=config['debug'])
        return train_dataset, val_dataset

    elif dataset == 'svo':
        train_dataset = SVOClassificationDataset(train_transform, 'train', debug=config['debug'])
        val_dataset = SVOClassificationDataset(test_transform, 'val', debug=config['debug'])
        return train_dataset, val_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers     


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )              
        loaders.append(loader)
    return loaders    