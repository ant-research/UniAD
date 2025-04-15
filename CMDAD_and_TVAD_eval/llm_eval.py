import os
import argparse
import ast
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import sys
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import json


@torch.no_grad()
def eval_each(model, tokenizer, text_gt, text_pred):
    sys_prompt = "[INST] <<SYS>>\nYou are an intelligent chatbot designed for evaluating the quality of generative outputs for movie audio descriptions. Your task is to compare the predicted audio descriptions with the correct audio descriptions and determine its level of match, considering mainly the visual elements like actions, objects and interactions. Here's how you can accomplish the task:------##INSTRUCTIONS: - Check if the predicted audio description covers the main visual events from the movie, especially focusing on the verbs and nouns.\n- Evaluate whether the predicted audio description includes specific details rather than just generic points. It should provide comprehensive information that is tied to specific elements of the video.\n- Consider synonyms or paraphrases as valid matches. Consider pronouns like 'he' or 'she' as valid matches with character names. Consider different character names as valid matches. \n- Provide a single evaluation score that reflects the level of match of the prediction, considering the visual elements like actions, objects and interactions. \n<</SYS>>\n\n{} [/INST] "

    prompt = (
        "Please evaluate the following movie audio description pair:\n\n"
        f"Correct Audio Description: {text_gt}\n"
        f"Predicted Audio Description: {text_pred}\n\n"
        "Provide your evaluation only as a matching score where the matching score is an integer value between 0 and 5, with 5 indicating the highest level of match. "
        "Please generate the response in the form of a Python dictionary string with keys 'score', where its value is the matching score in INTEGER, not STRING."
        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
        "For example, your response should look like this: {'score': }."
    )

    input_tokens = tokenizer(sys_prompt.format(prompt), return_tensors="pt", add_special_tokens=False).to('cuda').input_ids
    input_tokens = input_tokens.cuda()
    input_embedding = model.model.embed_tokens(input_tokens)

    outputs1 = model.generate(
        inputs_embeds=input_embedding,
        max_new_tokens=8,
        stopping_criteria=None,
        num_beams=1,
        do_sample=True,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
    )

    output_text = tokenizer.batch_decode(outputs1, add_special_tokens=False)
    return output_text


def main(args):
    # read llama2-7b model by customized config
    LLM_path = 'Put LLaMA2_7B_chat path here'
    tokenizer = LlamaTokenizer.from_pretrained(LLM_path)
    model = LlamaForCausalLM.from_pretrained(LLM_path, use_safetensors=False)
    model.eval()
    model.cuda()

    with open('Put AD gt json path here') as f:
        gt_ads = json.load(f)

    with open('Put AD prediction json path here') as f:
        generate_ads = json.load(f)

    # gt_ads = gt_ads['tvad']
    # generate_ads = generate_ads['tvad'] # for TVAD

    gt_ads = gt_ads['cmdad']
    generate_ads = generate_ads['cmdad'] # for CMDAD

    counter = 0
    all_output = []
    all_score = []

    for i, ad in enumerate(gt_ads):
        counter += 1
        if counter % 200 == 0:  # monitor progress
            print(np.mean(all_score), f'at {counter} sampeles')
        text_gt = ad
        text_pred = generate_ads[i]
        eval_out = eval_each(model=model, tokenizer=tokenizer, text_gt=text_gt, text_pred=text_pred)
        assert len(eval_out) == 1
        eval_out = eval_out[0]
        try:
            score_dict = ast.literal_eval(eval_out.replace('<s>','').replace('</s>','').strip())
        except:
            print('get error from:', eval_out)
            score_dict = {'score': 0}
        all_output.append(eval_out)
        all_score.append(score_dict['score'])

    print(f"final LLM-eval average score: {np.mean(all_score)}")

if __name__ == "__main__":
    main(None)