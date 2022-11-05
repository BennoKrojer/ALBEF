import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_imagecode import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from collections import defaultdict

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
import wandb


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  
 
    for i,(image0, image1, text, targets, is_video, img_dir) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))        

        loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha) 

        loss.backward()  
        if i%config['grad_accumulation'] == 0:
            optimizer.step()
            optimizer.zero_grad()
              
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


def train_hard_neg(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size  

    for i,(img0, img1, text, targets, is_video, img_dir) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        img0 = img0.flatten(0,1)
        img1 = img1.flatten(0,1)
        texts = []
        for t in text:
            texts += [t]*9
        targets = targets.flatten()
        images = torch.cat([img0, img1], dim=0)
        images, targets = images.to(device), targets.to(device)   

        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  

        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))        

        loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha)    
        
        loss.backward()  
        if i%config['grad_accumulation'] == 0:
            optimizer.step()
            optimizer.zero_grad()
       
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)
            wandb.log({'loss': loss.item()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    for image0, image1, text, targets, is_video, img_dir in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  

        prediction = model(images, text_inputs, targets=targets, train=False)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        video_accuracy = ((pred_class.cuda() == targets.cuda()) * is_video.cuda()).sum() / is_video.sum()
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))
        if not torch.isnan(video_accuracy):
            metric_logger.meters['video_acc'].update(video_accuracy.item(), n=image0.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    

@torch.no_grad()
def evaluate_fullset(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    acc = 0
    vid_acc = 0
    total = 0
    vid_total = 0

    for i,(img0, img1, pairs, text, targets, is_video, img_dir) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        #TODO: this will assume batchsize=1 for now

        img0 = img0.flatten(0,1)
        img1 = img1.flatten(0,1)
        texts = []
        for t in text:
            texts += [t]*img0.shape[0]
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  
        targets = targets.flatten()
        images = torch.cat([img0, img1], dim=0)
        images, targets = images.to(device), targets.to(device)   

        prediction = model(images, text_inputs, targets=targets, train=False)  
 
        _, pred_class = prediction.max(1)
        scores = defaultdict(int)
        for pair, single_pred in zip(pairs, pred_class):
            if single_pred == 1:
                scores[int(pair[0])] += 1
            else:
                scores[int(pair[1])] += 1

        scores = sorted(scores.items(), key = lambda x: x[1], reverse=True)
        pred_class = scores[0][0]
        acc += targets==pred_class
        total += 1
        if is_video:
            vid_acc += pred_class == targets
            vid_total += 1
    
    return acc/total, vid_acc/vid_total


    
def main(args, config):    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")
    datasets = create_dataset('imagecode', config, fullset=True)
    
    samplers = [None, None]

    train_loader, val_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']]*2,
                                              num_workers=[4,4],is_trains=[True,False,False], collate_fns=[None,None])

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
            
        msg = model.load_state_dict(state_dict,strict=False)

        if config['distill']:
            model.copy_params()
            
        print('load checkpoint from %s'%args.checkpoint)
        print(msg)

    model = model.to(device)   
    
    model_without_ddp = model
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)  
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if config['random_pair_sampling']:
                train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            else: 
                train_stats = train_hard_neg(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
        else:
            train_stats = {}

        # if args.inference_eval:
        #     acc, vid_acc = evaluate_fullset(model, fullset_val_loader, tokenizer, device, config)
        val_stats = evaluate(model, val_loader, tokenizer, device, config)

        wandb.log({'Val_acc': float(val_stats['acc']), 'Val_video_acc': float(val_stats['video_acc'])})

        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                        }
                       
            if float(val_stats['acc'])>best:
                # save_obj = {
                #     'model': model_without_ddp.state_dict(),
                #     'optimizer': optimizer.state_dict(),
                #     'lr_scheduler': lr_scheduler.state_dict(),
                #     'config': config,
                #     'epoch': epoch,
                # }
                # torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch)) 
                best = float(val_stats['acc'])
                wandb.log({'Best Val Acc': best})
                best_epoch = epoch
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        lr_scheduler.step(epoch+warmup_steps+1)  
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/imagecode.yaml')
    parser.add_argument('--checkpoint', default='pretrain_model_nlvr.pth', type=str)
    parser.add_argument('--output_dir', default='output/imagecode')
    parser.add_argument('--evaluate', action='store_true')     
    parser.add_argument('--debug', action='store_true') 
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--grad_accumulation', default=128, type=int)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--min_lr', type=float, default=0.000003)
    parser.add_argument('--warmup_lr', type=float, default=0.000002)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--video_only', type=str, default='False')
    parser.add_argument('--random_pair_sampling', type=str, default='False')
    parser.add_argument('--aug_prob', type=float, default=0.3)
    parser.add_argument('--distill', type=str, default='True')
    parser.add_argument('--pretrained_cls_head', type=str, default='True')
    parser.add_argument('--inference_eval', action='store_true')
    parser.add_argument('--job_id', type=str, default="")
    args = parser.parse_args()

    if args.debug:
        wandb.init(project='Debug-can-be-deleted', settings=wandb.Settings(start_method='fork'))
    else:
        wandb.init(project='Multi-Image-Transfer-ALBEF', settings=wandb.Settings(start_method='fork'))

    args.video_only = args.video_only == 'True'
    args.random_pair_sampling = args.random_pair_sampling == 'True'
    args.pretrained_cls_head = args.pretrained_cls_head == 'True'
    args.distill = args.distill == 'True'

    wandb.config.update(args)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config['grad_accumulation'] = args.grad_accumulation
    config['optimizer']['lr'] = args.lr
    config['schedular']['lr'] = args.lr
    config['schedular']['min_lr'] = args.min_lr
    config['schedular']['warmup_lr'] = args.warmup_lr
    config['max_epoch'] = args.max_epoch
    config['video_only'] = args.video_only
    config['random_pair_sampling'] = args.random_pair_sampling
    config['aug_prob'] = args.aug_prob
    config['distill'] = args.distill
    config['pretrained'] = args.checkpoint
    config['pretrained_cls_head'] = args.pretrained_cls_head
    config['debug'] = args.debug

    if not config['random_pair_sampling']:
        config['batch_size'] = config['batch_size'] // 8
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
