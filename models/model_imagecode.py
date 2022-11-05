from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel

import torch
from torch import nn
import torch.nn.functional as F

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 tokenizer = None,
                 config = None,     
                 ):
        super().__init__()
        
        self.tokenizer = tokenizer 
        self.distill = config['distill']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))    

        bert_config = BertConfig.from_json_file(config['bert_config'])
        bert_config.num_hidden_layers = 18
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)  
        self.pretrained_cls_head = config['pretrained_cls_head']
        self.it_head = nn.Linear(self.text_encoder.config.hidden_size, 3) #pretrained head
        self.cls_head = nn.Sequential(
                  nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                  nn.ReLU(),
                  nn.Linear(self.text_encoder.config.hidden_size, 2)
                ) # nvlr2 head

        self.share_cross_attention(self.text_encoder.encoder)

        if self.distill:
            self.visual_encoder_m = VisionTransformer(
                img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))                 
            self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False) 
            self.share_cross_attention(self.text_encoder_m.encoder)                
            self.it_head_m = nn.Linear(self.text_encoder.config.hidden_size, 3) #pretrained head 
            self.cls_head_m = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
                nn.ReLU(),
                nn.Linear(self.text_encoder.config.hidden_size, 2)
            )                # nvlr2 head

            self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                                [self.text_encoder,self.text_encoder_m],
                                [self.it_head,self.it_head_m] if self.pretrained_cls_head else [self.cls_head,self.cls_head_m],
                               ]
            self.copy_params()        
            self.momentum = 0.995
            
            
    def forward(self, image, text, targets, alpha=0, train=True, task=''):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        image0_embeds, image1_embeds = torch.split(image_embeds, image_embeds.shape[0]//2, dim=0)

        output = self.text_encoder(text.input_ids, 
                                   attention_mask = text.attention_mask, 
                                   encoder_hidden_states = [image0_embeds,image1_embeds],
                                   encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                             image_atts[image0_embeds.size(0):]],        
                                   return_dict = True,
                                  )
        hidden_state = output.last_hidden_state[:,0,:]   
        if not task == 'nlvr':
            prediction = self.it_head(hidden_state)[:,:2]
        else:
            prediction = self.cls_head(hidden_state)

        if train:
            if self.distill:                
                with torch.no_grad():
                    self._momentum_update()
                    image_embeds_m = self.visual_encoder_m(image)  
                    image0_embeds_m, image1_embeds_m = torch.split(image_embeds_m,targets.size(0))
                    output_m = self.text_encoder_m(text.input_ids, 
                                               attention_mask = text.attention_mask, 
                                               encoder_hidden_states = [image0_embeds_m,image1_embeds_m],
                                               encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
                                                                         image_atts[image0_embeds.size(0):]],        
                                               return_dict = True,
                                              )   
                    if not task == 'nlvr':
                        prediction_m = self.it_head_m(output_m.last_hidden_state[:,0,:])[:,:2]
                    else:
                        prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   

                loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
                    F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()                        
            else:        
                loss = F.cross_entropy(prediction, targets)     
            return loss  
        else:
            return prediction
 


    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                

    def share_cross_attention(self, model):
            
        for i in range(6):
            layer_num = 6+i*2
            modules_0 = model.layer[layer_num].crossattention.self._modules
            modules_1 = model.layer[layer_num+1].crossattention.self._modules

            for name in modules_0.keys():
                if 'key' in name or 'value' in name:
                    module_0 = modules_0[name]
                    module_1 = modules_1[name]
                    if hasattr(module_0, "weight"):
                        module_0.weight = module_1.weight
                        if hasattr(module_0, "bias"):
                            module_0.bias = module_1.bias    