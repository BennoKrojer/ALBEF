import argparse
from nis import match
import os
from tabnanny import check
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.model_imagecode import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from dataset_imagecode import ImageCoDeDataset


def evaluate(model, data_loader, tokenizer, device):    
    correct = 0
    total = 0
    for i,(image, text, target, is_video, img_dir) in enumerate(data_loader):
        image = image.to(device,non_blocking=True)   
        target = target.to(device,non_blocking=True)
        image = image.flatten(end_dim=1)
        text_ = []
        for t in text:
            text_ += [t]*10
        text_input = tokenizer(text_, padding='longest', return_tensors="pt").to(device)  #TODO: max_length
        matching_score = model(image, text_input) #TODO: deal cleanly with these 2-dim output as softmax probabilities and for finetuning later
        matching_score = torch.nn.functional.softmax(matching_score, dim=1)[:,1]
        matching_score = matching_score.reshape(-1, 10)
        pred = torch.argmax(matching_score, dim=1)
        correct += (pred == target).sum()
        total += target.shape[0]
        print(correct/total)
    return correct/total
        


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    # train_dataset, val_dataset, test_dataset = create_dataset('re', config)
    data_dir = '../imagecode/data'
    train_dataset = ImageCoDeDataset(data_dir, 'train', config)
    val_dataset = ImageCoDeDataset(data_dir, 'valid', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader = create_loader([train_dataset, val_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']],
                                                          num_workers=[4,4],
                                                          is_trains=[True, False], 
                                                          collate_fns=[None,None])   
       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        if checkpoint == 'ALBEF.pth':
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        # m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)   
        # state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped 
        
        for key in list(state_dict.keys()):
            if 'bert' in key:
                encoder_key = key.replace('bert.','')         
                state_dict[encoder_key] = state_dict[key] 
                del state_dict[key]                
        msg = model.load_state_dict(state_dict,strict=False)  
        
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)  
        
    
    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    print("Start training")
    start_time = time.time()    
    for epoch in range(0, max_epoch):
        if args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)
            accuracy = evaluate(model, val_loader, tokenizer, device)
            print(accuracy)            
        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        dist.barrier()     
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/imagecode')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
