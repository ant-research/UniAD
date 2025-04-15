# Contextual AD Narration with Interleaved Multimodal Sequence
<a href="https://arxiv.org/abs/2403.12922"><img src="https://img.shields.io/badge/arXiv-2403.12922-b31b1b.svg"></a>
<a href="http://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>

[Hanlin Wang](https://scholar.google.com/citations?user=0uO4fzkAAAAJ&hl=zh-CN)<sup>1,3</sup>, [Zhan Tong](https://scholar.google.com/citations?user=6FsgWBMAAAAJ&hl=zh-CN)<sup>2</sup>, [Kecheng Zheng](https://zkcys001.github.io/)<sup>3</sup>, [Yujun Shen](https://shenyujun.github.io/)<sup>3</sup>, [Limin Wang](https://wanglimin.github.io/)<sup>1,4,†</sup><br>
<sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University<br> <sup>2</sup>ESAT, KU Leuven <sup>3</sup>Ant Group <sup>4</sup>Shanghai Artificial Intelligence Laboratory <sup> <br><sup>†</sup>corresponding author

- [Contextual AD Narration with Interleaved Multimodal Sequence](#contextual-ad-narration-with-interleaved-multimodal-sequence)
  - [Setup](#setup)
  - [Running](#running)
  - [Acknowledgement](#acknowledgement)

##  Setup
Follow the following guide to set up the environment.

1. Git clone repo

    ```
    git clone ()
    cd UniAD
    ```

2. Download and unzip checkpoints

   Download necessary files from [here]()

   Download 'CLIP_L14_frames_features_5fps.h5' from [MAD](https://github.com/Soldelli/MAD)

   Use method in [AutoAD-II](https://github.com/TengdaHan/AutoAD/tree/main/autoad_ii/character_recognition) to get 'MAD_examplers.pth.tar'
   
   Download 'LLAMA2-7B'

3. Create environment and install packages

    Create environment for MAD:

    ```
    conda create -n UniAD_MAD python=3.8 -y
    conda activate UniAD_MAD
    pip install -r requirements_MAD_clean.txt
    ```

    Create environment for CMDAD & TVAD:

    ```
    conda create -n UniAD_CMDAD python=3.8 -y
    conda activate UniAD_CMDAD
    pip install -r requirements_CMD_clean.txt
    pip install --no-deps torchvision==0.13.1
    ```

    Create environment for critic evaluation in CMDAD & TVAD:

    ```
    conda create -n UniAD_critic python=3.9 -y
    pip install -r requirements_critic.txt
    ```

##  Running
We train our model with 8 A100 GPUs and evaluate with a single A6000 GPU card.

### Evaluation
Conduct evaluation on MAD:
```
CUDA_VISIBLE_DEVICES=0 python main.py --LLM_path 'LLAMA2-7B path' --batch-size-val 3 --char_feature_path 'MAD_examplers.pth.tar path' --char_prompt_type 0 --resume 'MAD.pt path' --if_finutune_GPT 0 --if_img_only 0 --if_lora 1 --if_only_flamingo 2 --mylogs 'output file diectory path' --movie_feature_path 'CLIP_L14_frames_features_5fps.h5 path' --name MAD_LLAMA2 --previous_video_num 1 --val-data 'MAD_eval_char_refine_final.json path' --workers 4
```

Conduct evaluation on CMDAD & TVAD:
```
CUDA_VISIBLE_DEVICES=0 python main.py --if_finutune_GPT 0 --mylogs 'output file diectory path' --name CMDAD_LLAMA2 --precision fp32 --if_lora 1 --train-data "" --val-data 'cmdad_char_refine_eval.json path' --log-every-n-steps 1 --dataset-type json --batch-size 1 --batch-size-val 1 --workers 1 --Visual_Loss 0 --LLM_path 'LLAMA2-7B path' --if_only_flamingo 2 --if_special_prompt 0 --num_latents 32 --num_char 32 --if_img_only 0 --movie_feature_eval_path 'VideoLLaMa_CMD_eval_fp16.h5 path' --char_feature_path 'chars_all_videollama.pth.tar path' --previous_video_num 1 --lr 0.0001 --resume 'CMDAD_TVAD.pt path' --eval_data_name CMDAD

CUDA_VISIBLE_DEVICES=0 python main.py --if_finutune_GPT 0 --mylogs 'output file diectory path' --name TVAD_LLAMA2 --precision fp32 --if_lora 1 --train-data "" --val-data 'tvad_char_refine_eval.json path' --log-every-n-steps 1 --dataset-type json --batch-size 1 --batch-size-val 1 --workers 1 --Visual_Loss 0 --LLM_path 'LLAMA2-7B path' --if_only_flamingo 2 --if_special_prompt 0 --num_latents 32 --num_char 32 --if_img_only 0 --movie_feature_eval_path 'TV_eval_videollama.h5 path' --char_feature_path 'chars_all_videollama.pth.tar path' --previous_video_num 1 --lr 0.0001 --resume 'CMDAD_TVAD.pt path' --eval_data_name TVAD
```

### Note: 
During the preparation for open-sourcing, we conducted ablation experiments on CMDAD and TVAD using the latest experimental settings. We found that the effects of the context modeling and character refinement module were minimal after introducing the VideoLLaMA model and the prediction results from AutoAD-Zero. For more specific details, please refer to our updated arXiv paper.

### Train on MAD
Prepare training data of MAD from [MAD](https://github.com/Soldelli/MAD) and use character prediction results from [AutoAD](http://www.robots.ox.ac.uk/~htd/autoad/MAD_char_prob_dict_trainval_MV550_CBcharbank_cos_top10_cal_jul.pkl) to organize the data into the following format：

```
[
  {
    "start": "",
    "end": "",
    "ad": "",
    "char": [],
    "ad_chars": [],
    "ad_chars_in_chars": [],
    "context": [
      {
        "start": "",
        "end": "",
        "ad": "",
        "char": [],
        "ad_chars": [],
        "ad_chars_in_chars": []
      }
    ],
    "movie_id": ""
  },
  ...
]
```

Then run:
```
torchrun --nproc_per_nod 8 -m main --if_finutune_GPT 0 --accum-freq 4 --if_lora 1 --if_only_flamingo 2 --num_latents 30
    - --if_special_prompt 0
    - --if_img_only 0
    - --num_char 30
    - --previous_video_num 1
    - --AD_pretrained 1
    - --AD_pretrained_checkpoint 'LLaMA_AD_pretrain.pt path'
    - --movie_feature_path 'CLIP_L14_frames_features_5fps.h5 path'
    - --char_feature_path 'MAD_examplers.pth.tar path'
    - --dataset 'MAD'
    - --train-data 'train data path'
    - --val-data ''
    - --LLM_path 'LLAMA2-7B path'
    - --batch-size 3
    - --batch-size-val 3
    - --epochs 10
    - --LLM_name 'LLaMA'
    - --lr 0.00005
    - --warmup 6000
    - --save-frequency 1
    - --val-frequency 1
    - --precision 'fp32'
    - --mylogs 'output file diectory path'
    - --name 'output file name'
```
    

## Citation
Don't forget to cite this source if it proves useful in your research!
```bibtex
@article{wang2024levitor, 
	title={Contextual AD Narration with Interleaved Multimodal Sequence}, 
	author={Hanlin Wang and Zhan Tong and Kecheng Zheng and Yujun Shen and Limin Wang}, 
	year={2025}, 
	eprint={2403.12922}, 
	archivePrefix={arXiv}, 
	primaryClass={cs.CV}}
```

## Acknowledgement
Our implementation is based on 
- [openclip](https://github.com/mlfoundations/open_clip)
- [AutoAD](https://github.com/TengdaHan/AutoAD)

Thanks for their remarkable contribution and released code!

## Note
Note: This repo is governed by the license of Apache 2.0 We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc. 

(注：本仓库受Apache 2.0的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
