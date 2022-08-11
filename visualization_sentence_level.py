import os
from pathlib import Path
model_path = '../ALBEF-old/refcoco.pth'
bert_config_path = 'configs/config_bert.json'
use_cuda = True
from dataset_imagecode import ImageCoDeDataset

from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel
from models.tokenization_bert import BertTokenizer

import torch
from torch import nn
from torchvision import transforms

import json

from PIL import Image

import cv2
import numpy as np

from skimage import transform as skimage_transform
from scipy.ndimage import filters
from matplotlib import pyplot as plt
import re


class VL_Transformer_ITM(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 config_bert = ''
                 ):
        super().__init__()
    
        bert_config = BertConfig.from_json_file(config_bert)

        self.visual_encoder = VisionTransformer(
            img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) 

        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)   
        
        self.itm_head = nn.Linear(768, 2) 

        
    def old_forward(self, image, text):
        image_embeds = self.visual_encoder(image) 

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        output = self.text_encoder(text.input_ids, 
                                attention_mask = text.attention_mask,
                                encoder_hidden_states = image_embeds,
                                encoder_attention_mask = image_atts,      
                                return_dict = True,
                               )     
           
        vl_embeddings = output.last_hidden_state[:,0,:]
        vl_output = self.itm_head(vl_embeddings)   
        return vl_output

    def forward(self, image, text):
        
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_embeds = text_output.last_hidden_state
        vl = self.text_encoder(encoder_embeds = text_embeds, 
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,      
                                        return_dict = True,
                                        mode = 'fusion',
                                       )            
        vl_embeddings = vl.last_hidden_state[:,0,:]
        vl_output = self.itm_head(vl_embeddings)            
        return vl_output

def pre_caption(caption,max_words=100):
    caption = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        caption.lower(),
    ).replace('-', ' ').replace('/', ' ')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    #truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])            
    return caption

def getAttMap(img, attMap, blur = True, overlap = True):
    attMap -= attMap.min()
    if attMap.max() > 0:
        attMap /= attMap.max()
    attMap = skimage_transform.resize(attMap, (img.shape[:2]), order = 3, mode = 'constant')
    if blur:
        attMap = filters.gaussian_filter(attMap, 0.02*max(img.shape[:2]))
        attMap -= attMap.min()
        attMap /= attMap.max()
    cmap = plt.get_cmap('jet')
    attMapV = cmap(attMap)
    attMapV = np.delete(attMapV, 3, 2)
    if overlap:
        attMap = 1*(1-attMap**0.7).reshape(attMap.shape + (1,))*img + (attMap**0.7).reshape(attMap.shape+(1,)) * attMapV
    return attMap


normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

transform = transforms.Compose([
    transforms.Resize((384,384),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])     

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model = VL_Transformer_ITM(text_encoder='bert-base-uncased', config_bert=bert_config_path)

checkpoint = torch.load(model_path, map_location='cpu')              
msg = model.load_state_dict(checkpoint,strict=False)
model.eval()

block_num = 11
TOKEN_MEAN = False

model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.save_attention = True

if use_cuda:
    model.cuda() 

data_dir = '../imagecode/data'
val_dataset = ImageCoDeDataset(data_dir, 'valid', {'image_res': 384})

for imgs, text, img_idx, is_video, img_dir in val_dataset:
    print(img_dir)
    img_files = list((Path(f'/network/scratch/b/benno.krojer/dataset/games/{img_dir}')).glob('*.jpg'))
    img_files = sorted(img_files, key=lambda x: int(str(x).split('/')[-1].split('.')[0][3:]))
    if not os.path.isdir(f'valid_visualization_{"mean" if TOKEN_MEAN else "cls"}_block{block_num}/{img_dir}_{img_idx}'):
        os.makedirs(f'valid_visualization_{"mean" if TOKEN_MEAN else "cls"}_block{block_num}/{img_dir}_{img_idx}')
    for j in range(10):
        image_path = img_files[j]
        img = imgs[j].unsqueeze(0)
        text = pre_caption(text)
        text_input = tokenizer(text, return_tensors="pt")

        if use_cuda:
            image = img.cuda()
            text_input = text_input.to(image.device)

        output = model(image, text_input)
        loss = output[:,1].sum()

        model.zero_grad()
        loss.backward()    

        with torch.no_grad():
            mask = text_input.attention_mask.view(text_input.attention_mask.size(0),1,-1,1,1)

            grads=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attn_gradients()
            cams=model.text_encoder.base_model.base_model.encoder.layer[block_num].crossattention.self.get_attention_map()

            cams = cams[:, :, :, 1:].reshape(image.size(0), 12, -1, 24, 24) * mask
            grads = grads[:, :, :, 1:].clamp(0).reshape(image.size(0), 12, -1, 24, 24) * mask

            gradcam = cams * grads
            if TOKEN_MEAN:
                gradcam = gradcam[0].mean(0).mean(0).cpu().detach()
            else:
                gradcam = gradcam[0].mean(0)[0].cpu().detach()

        # num_image = len(text_input.input_ids[0]) 
        fig, ax = plt.subplots(1, 1, figsize=(15,5*1))

        rgb_image = cv2.imread(str(image_path))[:, :, ::-1]
        rgb_image = np.float32(rgb_image) / 255

        ax.imshow(rgb_image)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlabel(text)
        gradcam_image = getAttMap(rgb_image, gradcam)
        ax.imshow(gradcam_image)
        plt.savefig(f'valid_visualization_{"mean" if TOKEN_MEAN else "cls"}_block{block_num}/{img_dir}_{img_idx}/{j}.png')