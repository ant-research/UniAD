import torch
from torch import nn
import torch.nn.functional as F
import math

from typing import Dict, List, Optional, Tuple, Union

import transformers
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaPreTrainedModel, ACT2FN, LlamaRotaryEmbedding, LlamaLinearScalingRotaryEmbedding, LlamaDynamicNTKScalingRotaryEmbedding, apply_rotary_pos_emb, repeat_kv, LlamaMLP, LlamaModel
from collections import OrderedDict
from .model_tfm import PerceiverEncoder

# from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss, MSELoss

from dataclasses import dataclass

import loralib as lora

DEFAULT_BOS_TOKEN = '<BOS>'
DEFAULT_EOS_TOKEN = '<EOS>'
DEFAULT_AD_BOS_TOKEN = '<s>'
DEFAULT_AD_EOS_TOKEN = '</s>'
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMG_START_TOKEN = "[VIS]"
DEFAULT_IMG_END_TOKEN = "[/VIS]"
DEFAULT_IMG_TOKEN = "<v>"
DEFAULT_VIDEO_START_TOKEN = "[VID]"
DEFAULT_VIDEO_END_TOKEN = "[/VID]"
DEFAULT_VIDEO_TOKEN = "<video>"

def get_avg_embedding(model: transformers.PreTrainedModel):
    embeddings = model.get_input_embeddings().weight.data
    embeddings_avg = embeddings.mean(dim=0, keepdim=True)
    return embeddings_avg.float()

def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
        resize_output: bool = True,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))

    # if num_new_tokens > 0:
    input_embeddings = model.get_input_embeddings().weight.data
    input_embeddings_avg = input_embeddings.mean(dim=0, keepdim=True)
    #     input_embeddings[-num_new_tokens:] = input_embeddings_avg

    #     if resize_output:
    output_embeddings = model.get_output_embeddings().weight.data
    output_embeddings_avg = output_embeddings.mean(dim=0, keepdim=True)
    #         output_embeddings[-num_new_tokens:] = output_embeddings_avg
    return input_embeddings_avg, output_embeddings_avg

@dataclass
class RegressCausalLMOutputWithPast(CausalLMOutputWithPast):
    regression_loss: Optional[torch.FloatTensor] = None

class LlamaAttentionForReg(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, r=8, lora_alpha=16, lora_layers=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_layers = lora_layers

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = lora.Linear(self.hidden_size, self.num_heads * self.head_dim, r=16, lora_alpha=16, lora_dropout=0.05, bias=False)
        self.k_proj = lora.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, r=16, lora_alpha=16, lora_dropout=0.05, bias=False)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = lora.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, r=16, lora_alpha=16, lora_dropout=0.05, bias=False)
        self.o_proj = lora.Linear(self.num_heads * self.head_dim, self.hidden_size, r=16, lora_alpha=16, lora_dropout=0.05, bias=False)
        # self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self._init_rope()
    
    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, scaling_factor=scaling_factor
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

class LlamaDecoderLayerForReg(nn.Module):
    def __init__(self, config: LlamaConfig, r=8, lora_alpha=16, lora_layers=None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttentionForReg(config=config, r=r, lora_alpha=lora_alpha, lora_layers=lora_layers)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModelForReg(LlamaModel):
    def __init__(self, config: LlamaConfig, r=8, lora_alpha=16, lora_layers=None):
        super().__init__(config)
        # self.padding_idx = config.pad_token_id
        # self.vocab_size = config.vocab_size

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayerForReg(config, r=r, lora_alpha=lora_alpha, lora_layers=lora_layers) for _ in range(config.num_hidden_layers)])
        # self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

class LlamaForReg(transformers.LlamaForCausalLM):
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "lm_embedding_new": kwargs.get("lm_embedding_new"),
                "lm_heads_new": kwargs.get("lm_heads_new"),
                "video_length": kwargs.get("video_length"),
                "if_train": kwargs.get("if_train"),
                "automodel": kwargs.get("automodel"),
            }
        )
        # print('kwargs:', kwargs)
        # print('-------------------------------------------------------------------')
        # print('model_inputs:', model_inputs)
        # print('-------------------------------------------------------------------')
        return model_inputs
    
    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        regress_mask: torch.Tensor = None,
        regress_labels=None,
        reg_head=None,
        lm_embedding_new=None,
        lm_heads_new=None,
        video_length=None,
        if_train=True,
        vis_x=None,
        media_locations=None,
        automodel=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print('input_ids:', input_ids)
        # print('-------------------------------------------------------------------')
        # print('inputs_embeds:', inputs_embeds)
        # print('-------------------------------------------------------------------')
        # assert input_ids is None and inputs_embeds is not None
        if input_ids is not None:
            assert inputs_embeds is None
            B = input_ids.shape[0]
            sequence_input = input_ids

            mask_pad = sequence_input == automodel.pad_token_id
            sequence_input[mask_pad] = automodel.eos_ad_token_id

            tokens_text = sequence_input.clone()
            tokens_video = sequence_input.clone()
            tokens_char = sequence_input.clone()

            mask = sequence_input >= automodel.num_embeddings
            tokens_text[mask] = 0
            text_embeds = self.model.embed_tokens(tokens_text).float()

            if automodel.if_special_prompt == 1:
                mask_video = (tokens_video == automodel.video_start_token_id) + (tokens_video == automodel.video_end_token_id) + (tokens_video == automodel.video_token_id)
            else:
                mask_video = (tokens_video == automodel.video_token_id)
            tokens_video -= automodel.video_token_id
            tokens_video[~mask_video] = 0
            video_embeds_ = lm_embedding_new[0](tokens_video)

            if automodel.if_special_prompt == 1:
                mask_img = (tokens_char == automodel.img_start_token_id) + (tokens_char == automodel.img_end_token_id) + (tokens_char == automodel.image_token_id)
            else:
                mask_img = (tokens_char == automodel.image_token_id)
            tokens_char -= automodel.image_token_id
            tokens_char[~mask_img] = 0
            img_embeds = lm_embedding_new[1](tokens_char)

            text_embeds[mask_video] = video_embeds_[mask_video]
            text_embeds[mask_img] = img_embeds[mask_img]
            input_ids = None
            inputs_embeds = text_embeds

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # regression mask : image start token and image tokens  size: [B, video length + 1]
        # regression label: image tokens    size: [B, video length, C]
        regress_loss = None

        if regress_mask is not None and if_train:
            B = regress_mask.shape[0]
            visual_outputs = hidden_states[regress_mask].reshape(B, video_length, -1)
            visual_outputs = visual_outputs[..., :-1, :]   # [B, video length, C]
            visual_tokens = reg_head(visual_outputs)
            visual_labels = regress_labels.reshape(B, video_length, -1)[..., 1:, :]
            loss_fct = MSELoss()
            regress_loss = loss_fct(visual_tokens, visual_labels)

        lm_logits = self.lm_head(hidden_states)
        for head in lm_heads_new:
            lm_logits_new = head(hidden_states)
            lm_logits = torch.cat((lm_logits, lm_logits_new), dim=-1)

        loss = None
        if labels is not None and if_train:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return RegressCausalLMOutputWithPast(
            loss=loss,
            regression_loss=regress_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
class LlamaForReg_lora(transformers.LlamaForCausalLM):
    def __init__(self, config, r=8, lora_alpha=16, lora_layers=None):
        super().__init__(config)
        self.model = LlamaModelForReg(config, r=r, lora_alpha=lora_alpha, lora_layers=lora_layers)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "lm_embedding_new": kwargs.get("lm_embedding_new"),
                "lm_heads_new": kwargs.get("lm_heads_new"),
                "video_length": kwargs.get("video_length"),
                "if_train": kwargs.get("if_train"),
                "automodel": kwargs.get("automodel"),
            }
        )
        # print('kwargs:', kwargs)
        # print('-------------------------------------------------------------------')
        # print('model_inputs:', model_inputs)
        # print('-------------------------------------------------------------------')
        return model_inputs
    
    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        regress_mask: torch.Tensor = None,
        regress_labels=None,
        reg_head=None,
        lm_embedding_new=None,
        lm_heads_new=None,
        video_length=None,
        if_train=True,
        vis_x=None,
        media_locations=None,
        automodel=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # print('input_ids:', input_ids)
        # print('-------------------------------------------------------------------')
        # print('inputs_embeds:', inputs_embeds)
        # print('-------------------------------------------------------------------')
        # assert input_ids is None and inputs_embeds is not None
        if input_ids is not None:
            assert inputs_embeds is None
            B = input_ids.shape[0]
            sequence_input = input_ids

            mask_pad = sequence_input == automodel.pad_token_id
            sequence_input[mask_pad] = automodel.eos_ad_token_id

            tokens_text = sequence_input.clone()
            tokens_video = sequence_input.clone()
            tokens_char = sequence_input.clone()

            mask = sequence_input >= automodel.num_embeddings
            tokens_text[mask] = 0
            text_embeds = self.model.embed_tokens(tokens_text).float()

            if automodel.if_special_prompt == 1:
                mask_video = (tokens_video == automodel.video_start_token_id) + (tokens_video == automodel.video_end_token_id) + (tokens_video == automodel.video_token_id)
            else:
                mask_video = (tokens_video == automodel.video_token_id)
            tokens_video -= automodel.video_token_id
            tokens_video[~mask_video] = 0
            video_embeds_ = lm_embedding_new[0](tokens_video)

            if automodel.if_special_prompt == 1:
                mask_img = (tokens_char == automodel.img_start_token_id) + (tokens_char == automodel.img_end_token_id) + (tokens_char == automodel.image_token_id)
            else:
                mask_img = (tokens_char == automodel.image_token_id)
            tokens_char -= automodel.image_token_id
            tokens_char[~mask_img] = 0
            img_embeds = lm_embedding_new[1](tokens_char)

            text_embeds[mask_video] = video_embeds_[mask_video]
            text_embeds[mask_img] = img_embeds[mask_img]
            input_ids = None
            inputs_embeds = text_embeds

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        # regression mask : image start token and image tokens  size: [B, video length + 1]
        # regression label: image tokens    size: [B, video length, C]
        regress_loss = None

        if regress_mask is not None and if_train:
            B = regress_mask.shape[0]
            visual_outputs = hidden_states[regress_mask].reshape(B, video_length, -1)
            visual_outputs = visual_outputs[..., :-1, :]   # [B, video length, C]
            visual_tokens = reg_head(visual_outputs)
            visual_labels = regress_labels.reshape(B, video_length, -1)[..., 1:, :]
            loss_fct = MSELoss()
            regress_loss = loss_fct(visual_tokens, visual_labels)

        lm_logits = self.lm_head(hidden_states)
        for head in lm_heads_new:
            lm_logits_new = head(hidden_states)
            lm_logits = torch.cat((lm_logits, lm_logits_new), dim=-1)

        loss = None
        if labels is not None and if_train:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return RegressCausalLMOutputWithPast(
            loss=loss,
            regression_loss=regress_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class VideoCaptionModel(nn.Module):
    def __init__(self,
                 args,
                 prefix_size: int = 768,
                 video_length: int = 8,
                 new_video_token_num = 3,
                 new_image_token_num = 3,
                 **kwargs,
                 ):
        super().__init__()
        if len(kwargs):
            print(f'WARNING [VideoCaptionModel] kwargs not used: {kwargs}')
        LLM_path = args.LLM_path
        if_regression = args.Visual_Loss
        char_length=args.num_char
        num_layers=args.perceiver_depth
        if_img_only=args.if_img_only
        num_latents=args.num_latents
        if_special_prompt=args.if_special_prompt
        if_only_flamingo=args.if_only_flamingo
        char_prompt_type=args.char_prompt_type

        self.num_layers = num_layers
        if if_only_flamingo == 2:
            if args.if_lora == 0:
                self.gpt = LlamaForReg.from_pretrained(LLM_path)
            else:
                self.gpt = LlamaForReg_lora.from_pretrained(LLM_path)
        else:
            # self.gpt = GPTForReg_gated.from_pretrained(LLM_path)
            assert 0
        self.gpt_embedding_size = self.gpt.model.embed_tokens.weight.shape[1]
        self.tokenizer = transformers.LlamaTokenizer.from_pretrained(
            LLM_path,
            model_max_length=4096,
            padding_side="right",
            truncation=True,
            use_fast=False,
        )
        self.if_regression = if_regression
        if if_regression:
            self.reg_head = nn.Linear(self.gpt_embedding_size, self.gpt_embedding_size)
            nn.init.normal_(self.reg_head.weight, std=self.gpt_embedding_size ** -0.5)
            nn.init.zeros_(self.reg_head.bias)

        self.video_length = video_length
        if if_only_flamingo == 1:
            self.video_token_length = 1
            self.char_length = 1
        else:
            self.video_token_length = num_latents
            self.char_length = char_length
        self.num_embeddings = self.gpt.model.embed_tokens.weight.shape[0]
        self.if_special_prompt = if_special_prompt
        self.if_only_flamingo = if_only_flamingo
        self.char_prompt_type = char_prompt_type
        self.if_share_img_video = args.if_share_img_video

        if if_special_prompt == 1:    
            # self.bos_token_id = self.num_embeddings + 1
            # self.eos_token_id = self.num_embeddings - 1

            self.bos_ad_token_id = self.tokenizer.bos_token_id
            self.eos_ad_token_id = self.tokenizer.eos_token_id
            
            self.video_token_id = self.num_embeddings
            self.video_start_token_id = self.num_embeddings + 1
            self.video_end_token_id = self.num_embeddings + 2

            self.image_token_id = self.num_embeddings + 3
            self.img_start_token_id = self.num_embeddings + 4
            self.img_end_token_id = self.num_embeddings + 5

            self.pad_token_id = self.num_embeddings + 6

            input_init = get_avg_embedding(self.gpt)

            self.new_video_token_num = new_video_token_num
            self.video_input_token_embedding = nn.Embedding(self.new_video_token_num, embedding_dim=self.gpt_embedding_size)
            self.video_input_token_embedding.weight.data = input_init.repeat(self.new_video_token_num, 1)
            self.video_output_token_embedding = nn.Linear(self.gpt_embedding_size, self.new_video_token_num, bias=False)
            self.video_output_token_embedding.weight = self.video_input_token_embedding.weight

            self.new_image_token_num = new_image_token_num
            self.image_input_token_embedding = nn.Embedding(self.new_image_token_num, embedding_dim=self.gpt_embedding_size)
            self.image_input_token_embedding.weight.data = input_init.repeat(self.new_image_token_num, 1)
            self.image_output_token_embedding = nn.Linear(self.gpt_embedding_size, self.new_image_token_num, bias=False)
            self.image_output_token_embedding.weight = self.image_input_token_embedding.weight

        else:
            
            # self.bos_token_id = self.num_embeddings + 1
            # self.eos_token_id = self.num_embeddings + 2

            # self.bos_ad_token_id = self.tokenizer.bos_token_id
            self.eos_ad_token_id = self.tokenizer.eos_token_id
            
            self.video_token_id = self.num_embeddings
            # self.video_start_token_id = self.num_embeddings + 6
            # self.video_end_token_id = self.num_embeddings + 7

            self.image_token_id = self.num_embeddings + 1
            # self.img_start_token_id = self.num_embeddings + 9
            # self.img_end_token_id = self.num_embeddings + 10
            
            self.pad_token_id = self.num_embeddings + 2

            input_init = get_avg_embedding(self.gpt)

            self.new_video_token_num = 1
            self.video_input_token_embedding = nn.Embedding(self.new_video_token_num, embedding_dim=self.gpt_embedding_size)
            self.video_input_token_embedding.weight.data = input_init.repeat(self.new_video_token_num, 1)
            self.video_output_token_embedding = nn.Linear(self.gpt_embedding_size, self.new_video_token_num, bias=False)
            self.video_output_token_embedding.weight = self.video_input_token_embedding.weight

            self.new_image_token_num = 1
            self.image_input_token_embedding = nn.Embedding(self.new_image_token_num, embedding_dim=self.gpt_embedding_size)
            self.image_input_token_embedding.weight.data = input_init.repeat(self.new_image_token_num, 1)
            self.image_output_token_embedding = nn.Linear(self.gpt_embedding_size, self.new_image_token_num, bias=False)
            self.image_output_token_embedding.weight = self.image_input_token_embedding.weight
        
        ### visual ###
        if if_only_flamingo != 1:
            self.perceiver = PerceiverEncoder(
                num_latents=num_latents, 
                d_latents=self.gpt_embedding_size, 
                num_layers=num_layers, 
                nhead=self.gpt_embedding_size//128)
            self.project = nn.Linear(prefix_size, self.gpt_embedding_size, dtype=torch.float32)
            nn.init.normal_(self.project.weight, std=prefix_size ** -0.5)
            nn.init.zeros_(self.project.bias)

            if self.char_prompt_type < 2 and self.if_share_img_video == 0:
                self.perceiver_img = PerceiverEncoder(
                    num_latents=self.char_length, 
                    d_latents=self.gpt_embedding_size, 
                    num_layers=num_layers, 
                    nhead=self.gpt_embedding_size//128)
                
                self.project_img = nn.Linear(prefix_size, self.gpt_embedding_size, dtype=torch.float32)
                nn.init.normal_(self.project_img.weight, std=prefix_size ** -0.5)
                nn.init.zeros_(self.project_img.bias)

        if if_only_flamingo != 2:
            self.perceiver_all = PerceiverEncoder(
                num_latents=num_latents, 
                d_latents=self.gpt_embedding_size, 
                num_layers=num_layers, 
                nhead=self.gpt_embedding_size//128)
            
            self.project_all = nn.Linear(prefix_size, self.gpt_embedding_size, dtype=torch.float32)
            nn.init.normal_(self.project_all.weight, std=prefix_size ** -0.5)
            nn.init.zeros_(self.project_all.bias)

        self.if_img_only = if_img_only
        self.if_video_split = args.if_video_split

    def get_embed(self, text_input=None,):
        B = text_input.shape[0]
        sequence_input = text_input
        mask_pad = sequence_input == self.pad_token_id
        sequence_input[mask_pad] = self.eos_ad_token_id

        tokens_text = sequence_input.clone()
        tokens_video = sequence_input.clone()
        tokens_char = sequence_input.clone()

        mask = sequence_input >= self.num_embeddings
        tokens_text[mask] = 0
        text_embeds = self.gpt.model.embed_tokens(tokens_text).float()

        if self.if_special_prompt == 1:
            mask_video = (tokens_video == self.video_start_token_id) + (tokens_video == self.video_end_token_id) + (tokens_video == self.video_token_id)
        else:
            mask_video = (tokens_video == self.video_token_id)
        tokens_video -= self.video_token_id
        tokens_video[~mask_video] = 0
        video_embeds_ = self.video_input_token_embedding(tokens_video)

        if self.if_special_prompt == 1:
            mask_img = (tokens_char == self.img_start_token_id) + (tokens_char == self.img_end_token_id) + (tokens_char == self.image_token_id)
        else:
            mask_img = (tokens_char == self.image_token_id)
        tokens_char -= self.image_token_id
        tokens_char[~mask_img] = 0
        img_embeds = self.image_input_token_embedding(tokens_char)

        text_embeds[mask_video] = video_embeds_[mask_video]
        text_embeds[mask_img] = img_embeds[mask_img]

        return text_embeds

    def forward(self, charactor_images=None, video_embeds=None, text_input=None, text_mask=None, past_key_values=None, text_output=None, output_mask=None, if_train=False, idx=None, video_idx=None, ad_start_ids=None):
        """
        Process:
        1. image video text as input: 'Possible characters: Mike played by George Clooney <v>, Stephen played by Ryan Gosling <v>. Describe <video>*8: ad'
        inputs are prompt, video frames and ad <eos>, first generate prompt then add <bos>
        2. prepend [VIS], [VID] tokens to img, video features
        3. replace <v>, <video> with image and text features
        4. prepend <BOS> to sequence and append <EOS> to end of sequence
        5. ignore <pad>, tokens before the last <video>
        6. feed into forward and return two losses

        :param charactor_images: [B, num_charactors, C], after projected into Language shape
        :param video_embeds: [B, num_videoframes, C], after projected into Language shape
        :param text_input: [B, seq_len], ad
        :param text_mask: [B, seq_len], no mentor with -100 ignored by crossentropy
        :return:
        """

        B = text_input.shape[0]
        sequence_input = text_input

        targets = None

        mask_pad = sequence_input == self.pad_token_id
        sequence_input[mask_pad] = self.eos_ad_token_id

        tokens_text = sequence_input.clone()
        tokens_video = sequence_input.clone()
        tokens_char = sequence_input.clone()

        mask = sequence_input >= self.num_embeddings
        tokens_text[mask] = 0
        text_embeds = self.gpt.model.embed_tokens(tokens_text).float()

        if self.if_special_prompt == 1:
            mask_video = (tokens_video == self.video_start_token_id) + (tokens_video == self.video_end_token_id) + (tokens_video == self.video_token_id)
        else:
            mask_video = (tokens_video == self.video_token_id)
        tokens_video -= self.video_token_id
        tokens_video[~mask_video] = 0
        video_embeds_ = self.video_input_token_embedding(tokens_video)

        if self.if_special_prompt == 1:
            mask_img = (tokens_char == self.img_start_token_id) + (tokens_char == self.img_end_token_id) + (tokens_char == self.image_token_id)
        else:
            mask_img = (tokens_char == self.image_token_id)
        tokens_char -= self.image_token_id
        tokens_char[~mask_img] = 0
        img_embeds = self.image_input_token_embedding(tokens_char)

        text_embeds[mask_video] = video_embeds_[mask_video]
        text_embeds[mask_img] = img_embeds[mask_img]

        all_video_indices = (sequence_input == self.video_token_id).to(video_embeds.device)
        all_img_indices = (sequence_input == self.image_token_id).to(video_embeds.device)

        media_locations = None
        
        gated_embeds = None
        if self.if_only_flamingo != 2:
            for i in range(len(video_embeds)):
                current_video = video_embeds[i]
                if charactor_images is not None:
                    video_chars = video_idx[i]
                    all_chars = None
                    for char_idx in video_chars:
                        if all_chars is None:
                            all_chars = charactor_images[char_idx].unsqueeze(0)
                        else:
                            all_chars = torch.cat((all_chars, charactor_images[char_idx].unsqueeze(0)), dim=0)
                    if all_chars is not None:
                        current_video = torch.cat((current_video, all_chars), dim=0)
                else:
                    assert all_img_indices.sum() == 0
                    # assert self.char_prompt_type > 1
                current_video = current_video.unsqueeze(0)
                x = self.project_all(current_video)
                x = self.perceiver_all(x)
                if gated_embeds is None:
                    gated_embeds = x
                else:
                    gated_embeds = torch.cat((gated_embeds, x), dim=0)

            gated_embeds = gated_embeds.unsqueeze(1)

        if self.if_only_flamingo != 1:
            perceiver_embeds_ = self.project(video_embeds)
            if self.if_video_split == 0:
                perceiver_embeds = self.perceiver(perceiver_embeds_)
            else:
                perceiver_embeds = None
                for video_i in range(perceiver_embeds_.shape[1] // self.video_length):
                    temp_perceiver_embeds = self.perceiver(perceiver_embeds_[:, video_i * self.video_length:(video_i + 1) * self.video_length, :])
                    if perceiver_embeds is None:
                        perceiver_embeds = temp_perceiver_embeds
                    else:
                        perceiver_embeds = torch.cat((perceiver_embeds, temp_perceiver_embeds), dim=1)

            perceiver_embeds = perceiver_embeds.reshape(-1, perceiver_embeds.shape[-1]).float()
            text_embeds[all_video_indices] = perceiver_embeds

            if charactor_images is not None:
                assert len(charactor_images) == len(idx)
                if self.if_img_only == 0:
                    charactor_images = charactor_images.unsqueeze(1)
                    if self.if_share_img_video == 0:
                        char_img_embeds = self.project_img(charactor_images)
                        char_img_embeds = self.perceiver_img(char_img_embeds)
                    else:
                        char_img_embeds = self.project(charactor_images)
                        char_img_embeds = self.perceiver(char_img_embeds)
                elif self.if_img_only == 1:
                    char_img_embeds = None
                    for i in range(len(charactor_images)):
                        char_video = video_embeds[idx[i]]
                        char_input = torch.cat((char_video, charactor_images[i].unsqueeze(0)), dim=0).unsqueeze(0)
                        if self.if_share_img_video == 0:
                            c = self.project_img(char_input)
                            c = self.perceiver_img(c)
                        else:
                            c = self.project(char_input)
                            c = self.perceiver(c)
                        if char_img_embeds is None:
                            char_img_embeds = c
                        else:
                            char_img_embeds = torch.cat((char_img_embeds, c), dim=0)
                else:
                    assert 0

                char_img_embeds = char_img_embeds.reshape(-1, char_img_embeds.shape[-1]).float()
                text_embeds[all_img_indices] = char_img_embeds

            else:
                if self.if_share_img_video == 0:
                    assert all_img_indices.sum() == 0
                    nothing_ = self.project_img(torch.zeros_like(video_embeds).to(video_embeds.device))
                    if self.if_video_split == 0:
                        nothing = self.perceiver_img(nothing_)
                    else:
                        nothing = None
                        for video_i in range(nothing_.shape[1] // self.video_length):
                            temp_nothing = self.perceiver_img(nothing_[:, video_i * self.video_length:(video_i + 1) * self.video_length, :])
                            if nothing is None:
                                nothing = temp_nothing
                            else:
                                nothing = torch.cat((nothing, temp_nothing), dim=1)

                    # nothing = self.perceiver_img(nothing)
                    nothing = nothing.reshape(-1, nothing.shape[-1]).float()
                    text_embeds[all_video_indices] += 0 * nothing
                    assert (text_embeds[all_video_indices] == perceiver_embeds).all()
                # assert self.char_prompt_type > 1

        lm_heads_new=[self.video_output_token_embedding, self.image_output_token_embedding]
        outputs = self.gpt(
            inputs_embeds=text_embeds,
            attention_mask=text_mask,
            return_dict=True,
            labels=targets,
            lm_heads_new=lm_heads_new,
            video_length=self.video_token_length,
            past_key_values=past_key_values,
            if_train=if_train,
            vis_x=gated_embeds,
            media_locations=media_locations,
            automodel=None,
        )

        return outputs, text_embeds, gated_embeds