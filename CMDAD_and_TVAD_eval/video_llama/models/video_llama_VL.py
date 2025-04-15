import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.modeling_llama import LlamaForCausalLM
# from video_llama.models.Qformer import BertEncoder
from transformers import LlamaTokenizer,BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from video_llama.models.Qformer import BertConfig, BertLMHeadModel

@registry.register_model("video_llama_VL")
class VideoLLAMA_VL(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        encoder_config = BertConfig.from_pretrained("put bert-base-uncased path here")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        low_resource=False,  # use 8 bit and put vit in cpu
        frozen_video_Qformer=False,
        llama_proj_model='',
        max_frame_pos= 32,
        num_video_query_token = 32,
    ):
        super().__init__()
        self.low_resource = low_resource
        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, 768)

        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=768, num_hidden_layers =2)

        self.video_Qformer.cls = None
        self.video_Qformer.bert.embeddings.word_embeddings = None
        self.video_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.video_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            
            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')

        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            768, 4096
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            self.load_state_dict(llama_proj_weight['model'], strict=False)

        self.train_flag = 0

        ckpt_path = 'put VL_LLaMA_2_7B_Finetuned.pth path here'
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(ckpt['model'], strict=False)
        
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def forward(self, q_hidden_state):
        batch_size, time_length, _, _ = q_hidden_state.shape
        device = q_hidden_state.device
        position_ids = torch.arange(time_length, dtype=torch.long, device=q_hidden_state.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        frame_position_embeddings = self.video_frame_position_embedding(position_ids)

        frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
        # frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
        frame_hidden_state = q_hidden_state
        frame_hidden_state = frame_position_embeddings + frame_hidden_state

        # frame attention
        frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
        frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
        video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

        video_query_output = self.video_Qformer.bert(
            query_embeds=video_query_tokens,
            encoder_hidden_states=frame_hidden_state,
            encoder_attention_mask=frame_atts,
            return_dict=True,
            )
        video_hidden = video_query_output.last_hidden_state

        inputs_llama = self.llama_proj(video_hidden)
    
        return inputs_llama