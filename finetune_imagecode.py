import argparse
from nis import match
import os
from statistics import mode
from tabnanny import check
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
from tqdm import tqdm
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
# from torchmetrics.functional import accuracy
from dataset_imagecode import ImageCoDeDataset
import wandb

wandb.init(project='ALBEF-finetune', settings=wandb.Settings(start_method='fork'))


def evaluate(model, data_loader, tokenizer, device):    
    correct = 0
    correct2 = 0
    correct3 = 0
    correct5 = 0
    total = 0
    for i,(image, text, target, is_video, img_dir) in enumerate(tqdm(data_loader)):
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
        # correct2 += accuracy(matching_score, target, top_k=2) * target.shape[0]
        # correct3 += accuracy(matching_score, target, top_k=3) * target.shape[0]
        # correct5 += accuracy(matching_score, target, top_k=5) * target.shape[0]
        correct2 = 0
        correct3 = 0
        correct5 = 0
        total += target.shape[0]

    return correct/total, correct2/total, correct3/total, correct5/total
        
def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, args):  
    model.train()
    step_size = 100
    warmup_iterations = warmup_steps*step_size
    total_loss = 0
    for i,(image, text, target, is_video, img_dir) in enumerate(tqdm(data_loader)):
        image = image.to(device,non_blocking=True)   
        image = image.flatten(end_dim=1)
        target = target.to(device,non_blocking=True)
        if args.binary_cross_entropy:
            target = F.one_hot(target, num_classes=10).flatten()
        text_ = []
        for t in text:
            text_ += [t]*10
        text_input = tokenizer(text_, padding='longest', max_length=30, return_tensors="pt").to(device)  #TODO: max_length
        matching_score = model(image, text_input) #TODO: deal cleanly with these 2-dim output as softmax probabilities and for finetuning later
        if not args.binary_cross_entropy:
            matching_score = matching_score[:,1]
            matching_score = matching_score.reshape(-1, 10)
        ce_loss = F.cross_entropy(matching_score, target)
        total_loss += ce_loss.item()
        ce_loss.backward()

        if i%args.grad_accumulation == 0:   
            optimizer.step()
            optimizer.zero_grad()

        epoch_cond = True if args.scheduler_always else (epoch==0)
        if epoch_cond and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)

        if i != 0 and i%step_size==0:
            wandb.log({'Loss': total_loss/step_size})
            total_loss = 0


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
    test_dataset = ImageCoDeDataset(data_dir, 'test', config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[args.batchsize, args.batchsize, args.batchsize],
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None, None, None])   
       
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        if args.checkpoint == 'ALBEF.pth' or args.evaluate:
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
        if not args.evaluate:
            train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, args)
        with torch.no_grad():
            if args.evaluate:
                accuracy, accuracy2, accuracy3, accuracy5 = evaluate(model, test_loader, tokenizer, device)   
                print('TEST ACCURACY: ', accuracy)
            accuracy, accuracy2, accuracy3, accuracy5 = evaluate(model, val_loader, tokenizer, device)
            print('VALIDATION ACCURACY: ', accuracy)
            print(accuracy2)
            print(accuracy3)
            print(accuracy5)
            wandb.log({'Accuracy': accuracy})
            
              

        if args.evaluate: 
            break
           
        lr_scheduler.step(epoch+warmup_steps+1)  
        torch.cuda.empty_cache()
        if accuracy > best:
            best = accuracy
            best_epoch = epoch
            wandb.log({'Best Accuracy': best})
            save_obj = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        'epoch': epoch,
                    }
            torch.save(save_obj, os.path.join('checkpoints', 'finetune_refcoco_checkpoint_best.pth'))  

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

    with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
        f.write("best epoch: %d"%best_epoch)               

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()     
    parser.add_argument('--config', default='./configs/Retrieval_flickr.yaml')
    parser.add_argument('--output_dir', default='output/imagecode')        
    parser.add_argument('--checkpoint', default='')   
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--lr', type=float, default=4e-6)
    parser.add_argument('--decay', type=float, default=0.01)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--scheduler_always', type=str)
    parser.add_argument('--binary_cross_entropy', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--job_id', type=str)
    args = parser.parse_args()
    args.scheduler_always = args.scheduler_always == 'True'
    args.binary_cross_entropy = args.binary_cross_entropy == 'True'

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config['optimizer']['lr'] = args.lr
    config['optimizer']['weight_decay'] = args.decay

    wandb.config.update(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
