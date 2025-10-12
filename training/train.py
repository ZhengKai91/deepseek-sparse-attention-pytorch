import os
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Optional, Sequence
import json

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling
from torch.distributed import barrier
import wandb
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk

from models.model_loader import load_qwen3_dsa_from_pretrained


@dataclass
class ModelArguments:
    dataset_name_or_path: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="")
    use_wandb: bool = field(
        default=False,
    )
    wandb_name: str = field(
        default='test',
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )

def is_deepspeed_zero3(training_args: TrainingArguments) -> bool:
    if training_args.deepspeed is None:
        return False

    # 如果deepspeed是字符串，则从文件读取
    if isinstance(training_args.deepspeed, str):
        with open(training_args.deepspeed, 'r') as f:
            ds_config = json.load(f)
    else:
        ds_config = training_args.deepspeed

    zero_config = ds_config.get('zero_optimization', {})
    if zero_config.get('stage') == 3:
        return True
    return False

def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    base_model_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    #print('is_deepspeed_zero3:', is_deepspeed_zero3(training_args))
    model = load_qwen3_dsa_from_pretrained(
        base_model_name_or_path,
        is_ds_zero3=is_deepspeed_zero3(training_args),
        #use_flash_attention_2=training_args.use_flash_attn,
        torch_dtype=torch.bfloat16,
    )
     
    rank = int(os.environ.get('RANK', -1))
    if rank > 0:
        barrier()

    dataset_name_or_path = model_args.dataset_name_or_path
    if os.path.isfile(dataset_name_or_path):
        datasaet = load_dataset('json',data_files=model_args.dataset_name_or_path)
    else:
        dataset = laod_dataset(dataset_name_or_path)

    
    if rank == 0:
        barrier()
    
    if model_args.use_wandb:
        project_name = f'DSA-SFT-{model_args.wandb_name}'
        wandb.init(project=project_name, entity='', name=model_args.wandb_name, sync_tensorboard=False,
                job_type="CleanRepo", group=model_args.wandb_name)
        
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=None,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()