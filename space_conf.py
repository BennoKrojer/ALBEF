space = {
    'grad_accumulation': [8,16,32,64,128,256],
    'lr': [3e-5, 6e-5, 9e-5, 3e-4, 6e-4, 9e-6, 6e-6,3e-6],
    'warmup_lr': [2e-5, 5e-5, 8e-5, 2e-4, 5e-4, 8e-6, 5e-6,2e-6],
    'min_lr': [3e-5, 9e-6, 6e-6,3e-6, 1e-7],
    # 'decay': [0.15 ,0.1, 0.05, 0.02],
    'max_epoch': [10,12,15,20],
    # 'video_only': [True, False],
    'random_pair_sampling': [True, False],
    'aug_prob': [0.3,0.4,0.5,0.6],
    'pretrained_cls_head': [True, False],
    'distill': [True, False]
}