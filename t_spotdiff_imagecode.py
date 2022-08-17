import argparse
import os
import ruamel_yaml as yaml
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

from models.model_spotdiff import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer

from dataset_spotdiff import SpotdiffDataset

def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    total_loss = 0
    warmup_iterations = warmup_steps*step_size  
 
    for i,(image0, image1, text) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = torch.cat([image0, image1], dim=0)
        images = images.to(device)

        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").input_ids
        text_inputs = text_inputs.to(device)  
        
        if epoch>0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))        

        loss = model(images, text_inputs, train=True, alpha=alpha)    
        total_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
               
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        
        if epoch==0 and i%step_size==0 and i<=warmup_iterations: 
            scheduler.step(i//step_size)         

        if i != 0 and i%step_size==0:
            # wandb.log({'Loss': total_loss/step_size})
            print(f"Loss: {total_loss/step_size}")
            total_loss = 0
        
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

    total_loss = 0 
    for image0, image1, text in metric_logger.log_every(data_loader, print_freq, header):
        images = torch.cat([image0, image1], dim=0)
        images = images.to(device)
        
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").input_ids
        text_inputs = text_inputs.to(device)  

        loss = model(images, text_inputs, train=False)  
        total_loss += loss
        metric_logger.meters['loss'].update(loss.item(), n=image0.size(0))
                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
    
    
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
    print("Creating dataset")
    data_dir = '/Scratch/qing_meng/torch/ALBEF/spotdiff/'
    train_dataset = SpotdiffDataset(data_dir, 'train', config)
    val_dataset = SpotdiffDataset(data_dir, 'val', config)
    
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)         
    else:
        samplers = [None, None, None]

    train_loader, val_loader = create_loader([train_dataset, val_dataset],samplers,
                                                          batch_size=[args.batchsize]+[args.batchsize],
                                                          num_workers=[4,4],
                                                          is_trains=[True, False], 
                                                          collate_fns=[None,None])   

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model #### 
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, text_decoder=args.text_decoder, tokenizer=tokenizer)
    
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        if args.checkpoint == 'ALBEF.pth':
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
    # if args.checkpoint:    
    #     checkpoint = torch.load(args.checkpoint, map_location='cpu') 
    #     state_dict = checkpoint['model']
        
    #     # reshape positional embedding to accomodate for image resolution change
    #     pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'],model.visual_encoder)         
    #     state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        
    #     if config['distill']:
    #         model.copy_params()
            
    #     msg = model.load_state_dict(state_dict,strict=False)
    #     print('load checkpoint from %s'%args.checkpoint)
    #     print(msg)

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
    
    print("Start training")
    start_time = time.time()
    best = float('inf')
    best_epoch = 0

    for epoch in range(0, max_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)   
        val_stats = evaluate(model, val_loader, tokenizer, device, config)
        # test_stats = evaluate(model, test_loader, tokenizer, device, config)

        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                        }
                       
            if float(val_stats['loss']) < best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_spotdiff.pth')) 
                best = float(val_stats['loss'])
                best_epoch = epoch
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        lr_scheduler.step(epoch+warmup_steps+1)  
        # dist.barrier()   
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    if utils.is_main_process():   
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write("best epoch: %d"%best_epoch)               

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/NLVR.yaml')
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--output_dir', default='output/spotdiff')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--batchsize', type=int, default=4)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--decay', type=float)
    parser.add_argument('--grad_accumulation', type=int, default=1)
    parser.add_argument('--scheduler_always', type=str)
    parser.add_argument('--binary_cross_entropy', type=str)
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)
