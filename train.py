from __future__ import annotations

import argparse
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as Fyaml
import torch.optim as optim
import wandb
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import librosa

from mt.models.music_transcriber import get_model
from mt.utils.yaml import read_yaml
from mt.data.datasets.tokenizers import Tokenizer
from mt.data.datasets.maestro import Maestro
from mt.data.sampler import Sampler
from mt.data.collate import collate_fn
from mt.utils.torch import requires_grad, save_checkpoint, to_device
from mt.optim import get_optimizer_and_scheduler
from mt.optim.ema import update_ema
from mt.losses import lm_and_frame_loss


def train(args) -> None:

    # Arguments
    config_path = Path(args.config)
    wandb_log = not args.no_log
    filename = Path(__file__).stem
    device = "cuda"
    
    # Configs
    configs = read_yaml(config_path)
    batch_size = configs["train"]["batch_size_per_device"]
    
    # Checkpoints directory
    config_name = config_path.stem
    ckpts_dir = Path("./checkpoints") / filename / config_name
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer()

    # Dataset
    dataset = Maestro(
        root=configs["data"]["dataset"]["root"],
        split="train",
        segment_seconds=configs["data"]["dataset"]["crop_duration"],
        tokenizer=tokenizer,
        max_token_len=configs["data"]["dataset"]["max_tokens"],
    )

    sampler = Sampler(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True
    )

    # Model
    model = get_model(
        configs=configs["model"], 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

    # EMA
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs["train"]["optimizer"], 
        params=model.parameters()
    )
    
    # Validator TODO
    # validator = Validator(configs, conditioner, ema, device)

    # Logger
    if wandb_log:
        wandb.init(project="music_transcription", name=f"{filename}_{config_name}")
    
    for step, data in enumerate(tqdm(dataloader)):

        data = sim_multi_insts(data)
        data = to_device(data, device)

        # Attention masks
        x_mask = rearrange(data["token_mask"], 'b s l -> (b s) l')
        x_mask = x_mask[:, 0 : -1]
        self_attn_mask = (x_mask[:, :, None] & x_mask[:, None, :]).unsqueeze(1)  # (b*s, 1, l_q, l_q)

        # Input tokens
        tokens = rearrange(data["token"], 'b s l -> (b s) l')   # (b*s, l)
        onset_roll = rearrange(data["onset_roll"], 'b s t p -> (b s) t p')
        in_tokens = tokens[:, 0 : -1]
        tgt_tokens = tokens[:, 1:]

        in_dict = {
            "audio": data["audio"],  # (b, c, t)
            "token": in_tokens,  # (b*s, l)
            "self_attn_mask": self_attn_mask,  # (b*s, 1, l_q, l_q)
            "cross_attn_mask": None
        }

        tgt_dict = {
            "token": tgt_tokens,  # (b*s, l)
            "token_mask": x_mask,  # (b*s, l)
            "onset_roll": onset_roll  # (b*s, t, p)
        }

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        out_dict = model(in_dict)
        
        # Loss
        loss = lm_and_frame_loss(out_dict, tgt_dict)

        # 2.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()  # Update all parameters based on all parameter.grad
        if scheduler: 
            scheduler.step()
        update_ema(ema, model, decay=0.999)

        if step % 100 == 0:
            print(f"Loss: {loss:.4f}")
        
        # # ------ 3. Evaluation ------
        # # 3.1 Evaluate
        # if step % configs["train"]["test_every_n_steps"] == 0:
        #     for split in ["train", "test"]:
        #         out_dir = Path("./results") / filename / config_name / f"steps={step}_ema"
        #         validator(split=split, out_dir=out_dir)

        #     if wandb_log:
        #         wandb.log(
        #             data={
        #                 "train_loss": loss.item()
        #             },
        #             step=step
        #         )
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            ckpt_path = ckpts_dir / f"step={step}_ema.pth"
            save_checkpoint(ema, ckpt_path)
            print(f"Save model to {ckpt_path}")
        
        if step == configs["train"]["training_steps"]:
            break


def sim_multi_insts(data: dict) -> dict:
    n_insts = 13
    for key in ['frame_roll', 'onset_roll', 'offset_roll', 'velocity_roll']:
        data[key] = data[key].repeat(1, n_insts, 1, 1)

    for key in ['token', 'token_mask']:
        data[key] = data[key].repeat(1, n_insts, 1)

    for key in ['tokens_num']:
        data[key] = data[key].repeat(1, n_insts)

    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)