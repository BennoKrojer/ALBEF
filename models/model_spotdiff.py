from functools import partial
from models.vit import VisionTransformer
from models.xbert import BertConfig, BertModel, BertLMHeadModel

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class ALBEF(nn.Module):
    def __init__(self,                 
                 text_encoder = None,
                 text_decoder = None,
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
        
        self.text_encoder = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False)      
        # self.cls_head = nn.Sequential(
        #           nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
        #           nn.ReLU(),
        #           nn.Linear(self.text_encoder.config.hidden_size, 2)
        #         )            
        # self.share_cross_attention(self.text_encoder.encoder)

        config_decoder = BertConfig.from_json_file(config['bert_config'])
        config_decoder.fusion_layer = 0
        config_decoder.num_hidden_layers = 6
        self.text_decoder = BertLMHeadModel.from_pretrained(text_decoder, config=config_decoder)    
        self.text_decoder.config.decoder_start_token_id = self.tokenizer.cls_token_id
        

        # if self.distill:
        #     self.visual_encoder_m = VisionTransformer(
        #         img_size=config['image_res'], patch_size=16, embed_dim=768, depth=12, num_heads=12, 
        #         mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))                 
        #     self.text_encoder_m = BertModel.from_pretrained(text_encoder, config=bert_config, add_pooling_layer=False) 
        #     self.share_cross_attention(self.text_encoder_m.encoder)                

        #     self.cls_head_m = nn.Sequential(
        #               nn.Linear(self.text_encoder.config.hidden_size, self.text_encoder.config.hidden_size),
        #               nn.ReLU(),
        #               nn.Linear(self.text_encoder.config.hidden_size, 2)
        #             )                

        #     self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
        #                         [self.text_encoder,self.text_encoder_m],
        #                         [self.cls_head,self.cls_head_m],
        #                        ]
        #     self.copy_params()        
        #     self.momentum = 0.995
            
            
    def forward(self, image, labels, alpha=0, train=True):
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        image0_embeds, image1_embeds = torch.split(image_embeds,2)                   

        decoder_input_ids = shift_tokens_right(
            labels, self.text_decoder.config.pad_token_id, self.text_decoder.config.decoder_start_token_id
        )
        decoder_outputs = self.text_decoder(input_ids=decoder_input_ids, encoder_hidden_states=[image0_embeds, image1_embeds])
        # output = self.text_encoder(text.input_ids, 
        #                            attention_mask = text.attention_mask, 
        #                            encoder_hidden_states = [image0_embeds,image1_embeds],
        #                            encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
        #                                                      image_atts[image0_embeds.size(0):]],        
        #                            return_dict = True,
        #                           )  
        # hidden_state = output.last_hidden_state[:,0,:]            
        # prediction = self.cls_head(hidden_state)
        # print(decoder_outputs)
        prediction  = decoder_outputs.logits

#         if train:
#             if self.distill:                
#                 with torch.no_grad():
#                     self._momentum_update()
#                     image_embeds_m = self.visual_encoder_m(image)  
#                     image0_embeds_m, image1_embeds_m = torch.split(image_embeds_m,targets.size(0))
#                     output_m = self.text_encoder_m(text.input_ids, 
#                                                attention_mask = text.attention_mask, 
#                                                encoder_hidden_states = [image0_embeds_m,image1_embeds_m],
#                                                encoder_attention_mask = [image_atts[:image0_embeds.size(0)],
#                                                                          image_atts[image0_embeds.size(0):]],        
#                                                return_dict = True,
#                                               )    
#                     prediction_m = self.cls_head_m(output_m.last_hidden_state[:,0,:])   

#                 loss = (1-alpha)*F.cross_entropy(prediction, targets) - alpha*torch.sum(
#                     F.log_softmax(prediction, dim=1)*F.softmax(prediction_m, dim=1),dim=1).mean()                        
#             else:        
#                 loss = F.cross_entropy(prediction, targets)     
#             return loss  
#         else:
#             return prediction
        if train or labels is not None:
            logits = decoder_outputs.logits
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.text_decoder.config.vocab_size), labels.view(-1))
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
