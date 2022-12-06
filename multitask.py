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
from dataset.multitask_loader import MultitaskDataloader, DataLoaderWithTaskname


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

    for i,(task_name, image0, image1, text, targets, is_video, img_dir) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(config['tasks']) == 1 and task_name in ['moment', 'clevr'] and i > 15000:
            break
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))        

        loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha, task=task_name)

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

    for i,(task_name, img0, img1, text, targets, is_video, img_dir) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(config['tasks']) == 1 and task_name in ['moment', 'clevr'] and i > 15000:
            break
        if task_name == 'imagecode':
            img0 = img0.flatten(0,1)
            img1 = img1.flatten(0,1)
            texts = []
            for t in text:
                texts += [t]*9
            targets = targets.flatten()
        else:
            texts = text
            
        images = torch.cat([img0, img1], dim=0)
        images, targets = images.to(device), targets.to(device)   

        text_inputs = tokenizer(texts, padding='longest', return_tensors="pt").to(device)  

        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))        

        loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha, task=task_name)    
        
        loss.backward()  
        if i%config['grad_accumulation'] == 0:
            optimizer.step()
            optimizer.zero_grad()
       
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())

        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)
            # wandb.log({'loss': loss.item()})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}    


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config, taskname):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50
    for i, (image0, image1, text, targets, is_video, img_dir) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if taskname in ['moment', 'clevr'] and i % 10 != 0:
            continue
        images = torch.cat([image0, image1], dim=0)
        images, targets = images.to(device), targets.to(device)   
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)  

        prediction = model(images, text_inputs, targets=targets, train=False, task=taskname)
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        video_accuracy = ((pred_class.cuda() == targets.cuda()) * is_video.cuda()).sum() / is_video.sum()
        
        metric_logger.meters[f'{taskname}_acc'].update(accuracy.item(), n=image0.size(0))
        if not torch.isnan(video_accuracy):
            metric_logger.meters[f'{taskname}_video_acc'].update(video_accuracy.item(), n=image0.size(0))
                
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
    if args.distributed:
        utils.init_distributed_mode(args)    
    
    # print number of GPUs
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs:", num_gpus)

    # reduce grad_accumulation
    if num_gpus>1:
        config['grad_accumulation'] = config['grad_accumulation']//num_gpus

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating dataset")

    dataloaders_train = {}
    dataloaders_val = {}
    print('Tasks: ' + str(args.tasks))
    for task in args.tasks:
        if task == 'imagecode' and not config['random_pair_sampling']:
            batch_size = config['batch_size'] // 8
        else:
            batch_size = config['batch_size']
        print('Loading dataset for task', task)
        datasets = create_dataset(task, config)

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()            
            samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
        else:
            samplers = [None, None]

        train_loader, val_loader = create_loader(datasets, [None, None], batch_size=[batch_size]*2,
                                                    num_workers=[4,4],is_trains=[True,False], collate_fns=[None,None])
        dataloaders_train[task] = DataLoaderWithTaskname(task, train_loader)
        dataloaders_val[task] = val_loader

    multi_loader_train = MultitaskDataloader(dataloaders_train, sample_ratios=args.sample_ratios)
    
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, tasks=args.tasks)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']
        if args.load_optimizer:
            optim_state_dict = checkpoint['optimizer']
            scheduler_state_dict = checkpoint['lr_scheduler']
            print('Loading optimizer and scheduler')
        else:
            optim_state_dict = None
            scheduler_state_dict = None
        
        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
            
        msg = model.load_state_dict(state_dict,strict=False)

        if config['distill']:
            model.copy_params()


        print('load checkpoint from %s'%args.checkpoint)
        print(msg)
    else:
        optim_state_dict = None
        scheduler_state_dict = None


    model = model.to(device)   
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module    
    
    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model, state_dict=optim_state_dict)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer, state_dict=scheduler_state_dict)
    
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    
    print("Start training")
    start_time = time.time()
    best0 = 0
    best1 = 0
    best_epoch = 0

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            if config['random_pair_sampling']:
                train_stats = train(model, multi_loader_train, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
            else: 
                train_stats = train_hard_neg(model, multi_loader_train, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)  
        else:
            train_stats = {}

        if len(args.tasks) == 2:
            val_stats = evaluate(model, dataloaders_val[args.tasks[0]], tokenizer, device, config, args.tasks[0])
            val_stats.update(evaluate(model, dataloaders_val[args.tasks[1]], tokenizer, device, config, args.tasks[1]))
            wandb.log({f'{args.tasks[0]}_Val_acc': float(val_stats[f'{args.tasks[0]}_acc'])})
            wandb.log({f'{args.tasks[1]}_Val_acc': float(val_stats[f'{args.tasks[1]}_acc'])})
            if f'{args.tasks[0]}_vid_acc' in val_stats:
                wandb.log({f'{args.tasks[0]}_Val_vid_acc': float(val_stats[f'{args.tasks[0]}_vid_acc'])})
            if f'{args.tasks[1]}_vid_acc' in val_stats:
                wandb.log({f'{args.tasks[1]}_Val_vid_acc': float(val_stats[f'{args.tasks[1]}_vid_acc'])})
            
            if float(val_stats[f'{args.tasks[0]}_acc']) > best0:
                best0 = float(val_stats[f'{args.tasks[0]}_acc'])
                best_epoch = epoch
                wandb.log({f'{args.tasks[0]}_Best_Val_acc': best0})
            
            if float(val_stats[f'{args.tasks[1]}_acc']) > best1:
                best1 = float(val_stats[f'{args.tasks[1]}_acc'])
                wandb.log({f'{args.tasks[1]}_Best_Val_acc': best1})

        else:
            val_stats = evaluate(model, dataloaders_val[args.tasks[0]], tokenizer, device, config, args.tasks[0])
            wandb.log({f'{args.tasks[0]}_Val_acc': float(val_stats[f'{args.tasks[0]}_acc'])})
            if f'{args.tasks[0]}_vid_acc' in val_stats:
                wandb.log({f'{args.tasks[0]}_Val_vid_acc': float(val_stats[f'{args.tasks[0]}_vid_acc'])})
            
            if float(val_stats[f'{args.tasks[0]}_acc']) > best0:
                best0 = float(val_stats[f'{args.tasks[0]}_acc'])
                best_epoch = epoch
                wandb.log({f'{args.tasks[0]}_Best_Val_acc': best0})
        
        if utils.is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                        }
                       
            # if float(val_stats['acc'])>best:
            #     save_obj = {
            #         'model': model_without_ddp.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'lr_scheduler': lr_scheduler.state_dict(),
            #         'config': config,
            #         'epoch': epoch,
            #     }
            #     torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
            #     best = float(val_stats['acc'])
            #     wandb.log({'Best Val Acc': best})
            #     best_epoch = epoch
            
            # log latest checkpoint
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_latest.pth'))
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        lr_scheduler.step(epoch+warmup_steps+1)
        if args.distributed:
            dist.barrier()
                
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
    parser.add_argument('--output_dir', default='/home/mila/b/benno.krojer/scratch/transfer-study-output', type=str)
    parser.add_argument('--evaluate', action='store_true')     
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--grad_accumulation', default=128, type=int)
    parser.add_argument('--lr', type=float, default=0.00003)
    parser.add_argument('--min_lr', type=float, default=0.000003)
    parser.add_argument('--warmup_lr', type=float, default=0.000002)
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--video_only', type=str, default='False')
    parser.add_argument('--static_only', type=str, default='False')
    parser.add_argument('--random_pair_sampling', type=str, default='False')
    parser.add_argument('--aug_prob', type=float, default=0.3)
    parser.add_argument('--distill', type=str, default='True')
    parser.add_argument('--inference_eval', action='store_true')

    parser.add_argument('--share_heads', type=str, default='False')
    parser.add_argument('--tasks', type=str, default='imagecode,spotdiff,clevr,img-edit,moment,naturalist,nlvr,svo')
    parser.add_argument('--sample_ratios', type=str, default='')
    parser.add_argument('--multitask', type=str, default='')
    parser.add_argument('--imagecode_head', type=str, default='big')
    parser.add_argument('--load_optimizer', action='store_true')
    parser.add_argument('--job_id', type=str)
    args = parser.parse_args()

    args.tasks = args.tasks.split(',')
    if args.sample_ratios == '':
        args.sample_ratios = [1] * len(args.tasks)
    else:
        args.sample_ratios = [float(x) for x in args.sample_ratios.split(',')]

    args.big_heads = ['nlvr', 'spotdiff', 'clevr', 'img-edit', 'naturalist']
    args.pretrained_heads = ['moment', 'svo']
    if args.imagecode_head == 'big':
        args.big_heads.append('imagecode')
    else:
        args.pretrained_heads.append('imagecode')

    if args.debug:
        wandb.init(project='Debug-can-be-deleted', settings=wandb.Settings(start_method='fork'), group=args.job_id)
    else:
        wandb.init(project='Multi-Image-Transfer-ALBEF', settings=wandb.Settings(start_method='fork'), group=args.job_id)

    args.video_only = args.video_only == 'True'
    args.random_pair_sampling = args.random_pair_sampling == 'True'
    args.distill = args.distill == 'True'
    args.share_heads = args.share_heads == 'True'
    args.multitask = args.multitask == 'True'
    args.static_only = args.static_only == 'True'

    wandb.config.update(args)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config['grad_accumulation'] = args.grad_accumulation
    config['optimizer']['lr'] = args.lr
    config['schedular']['lr'] = args.lr
    config['schedular']['min_lr'] = args.min_lr
    config['schedular']['warmup_lr'] = args.warmup_lr
    config['schedular']['epochs'] = args.max_epoch
    config['video_only'] = args.video_only
    config['random_pair_sampling'] = args.random_pair_sampling
    config['aug_prob'] = args.aug_prob
    config['distill'] = args.distill
    config['pretrained'] = args.checkpoint
    config['debug'] = args.debug
    config['tasks'] = args.tasks
    config['sample_ratios'] = args.sample_ratios
    config['task_heads'] = args.share_heads
    config['multitask'] = args.multitask
    config['static_only'] = args.static_only
    config['big_heads'] = args.big_heads
    config['pretrained_heads'] = args.pretrained_heads
    config['load_optimizer'] = args.load_optimizer

    transfer_type = 'multi' if args.multitask else 'seq'

    args.output_dir = os.path.join(args.output_dir,f'{transfer_type}_{"-".join(args.tasks)}_{"-".join([str(x) for x in args.sample_ratios])}_{args.job_id}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
