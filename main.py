import argparse
import importlib

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Trainer, Callback, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
import os
import torch
import json
from dotmap import DotMap

wandb_id = wandb.util.generate_id()

#############################
# Example usage
# $ python3 main.py --rewriter OracleReWriter --hits 100
#############################

parser = argparse.ArgumentParser(description='Fine Tune a query re-writing model for CAsT')
parser.add_argument('-d','--dataset', action='append', help='datasource to use for fine-tuning')
parser.add_argument('--skip_train', default=False, action='store_true')
parser.add_argument('--from_checkpoint', type=str, default='')
parser.add_argument('--num_eval_samples', type=int, default=-1)
parser.add_argument('--wandb_id', type=str, default=wandb.util.generate_id())
parser.add_argument('--name', type=str, default=None)

parser.add_argument('--gpu_id', type=str, default='0')
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batch_sz', type=int, default=8)
parser.add_argument('--num_return_sequences', type=int, default=4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--warmup_steps', type=int, default=200)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--inference_chunk_size', type=int, default=32)
parser.add_argument('--save_dir', type=str, default='checkpoints')

base_args = DotMap()


def main(args):
    seed_everything(args.seed)
    
    train_data = claim_verification_dataset.get_data('strategyQA', 'train')
    dev_data = claim_verification_dataset.get_data('strategyQA', 'dev')
    

    print("Done!")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)