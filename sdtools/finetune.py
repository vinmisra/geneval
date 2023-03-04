'''
simple wrapper utility around the dreambooth_multi finetuning script.
Isolated as a separate process to avoid nasty Jupyter/CUDA memory leaks.
'''
import os, subprocess, sys
from enum import Enum
from typing import Optional, List, Union
import sdtools.cfg as cfg

os.environ['MKL_THREADING_LAYER'] = 'GNU'
DIR_DREAMBOOTH = __file__.rpartition('/')[0]

template = '''accelerate launch train_dreambooth_multi.py \
    --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4  \
    --train_text_encoder \
    --instance_data_dir={dir_finetune} \
    --class_data_dir={dir_class} \
    --output_dir={dir_output} \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --instance_prompt="{prompt_finetune}" \
    --class_prompt="{prompt_class}" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_checkpointing \
    --learning_rate={lr} \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images={n_class_img} \
    --snapshot_steps={snapshot_steps}'''

def train_model(
    lst_dir_finetune_img:List[str],
    dir_model:str,
    lst_prompt_finetune:List[str],
    lst_prompt_class:List[str],
    lst_n_classimg:List[int],
    lst_dir_class_img:List[str],
    lr:float,
    snapshot_steps:List[int],
    verbose:bool=False
):
    cmd = template.format(
        dir_finetune=':::'.join(lst_dir_finetune_img),
        dir_output=dir_model,
        prompt_finetune=':::'.join(lst_prompt_finetune),
        dir_class=':::'.join(lst_dir_class_img),
        prompt_class=':::'.join(lst_prompt_class),
        lr=lr,
        n_class_img=':::'.join([str(i) for i in lst_n_classimg]),
        snapshot_steps=':::'.join([str(i) for i in snapshot_steps])
    )
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)
    output = subprocess.run(cmd, shell=True, cwd=DIR_DREAMBOOTH)
    if verbose:
        print(cmd)