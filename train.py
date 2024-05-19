# train.py
import os
import time
import math
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
import numpy as np
import pickle
from contextlib import nullcontext
from models.gpt import GPT, GPTConfig
from utils.checkpointing import save_checkpoint_compressed, load_checkpoint_compressed
from utils.logging import initialize_wandb, log_metrics
from data.prepare_data import prepare_data

from config import config, model_config

def get_batch(data_dir, batch_size, block_size, device, split='train'):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def estimate_loss(model, data_dir, eval_iters, block_size, device, ctx):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data_dir, config['batch_size'], block_size, device, split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def train():
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=config['backend'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert config['gradient_accumulation_steps'] % ddp_world_size == 0
        gradient_accumulation_steps = config['gradient_accumulation_steps'] // ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
        gradient_accumulation_steps = config['gradient_accumulation_steps']

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * config['batch_size'] * config['block_size']
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    if master_process:
        os.makedirs(config['out_dir'], exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = 'cuda' if 'cuda' in config['device'] else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type != 'cpu' else nullcontext()

    data_dir = os.path.join('data', config['dataset'])
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        model_config['vocab_size'] = meta['vocab_size']
        print(f"found vocab_size = {meta['vocab_size']} (inside {meta_path})")

    if config['init_from'] == 'scratch':
        print("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_config)
        model = GPT(gptconf)
    elif config['init_from'] == 'resume':
        print(f"Resuming training from {config['out_dir']}")
        ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt.gz')
        gptconf = GPTConfig(**model_config)
        model = GPT(gptconf)
        optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)
        epoch = load_checkpoint_compressed(ckpt_path, model, optimizer)
    else:
        raise ValueError(f"Unsupported init_from value: {config['init_from']}")

    if model_config['block_size'] < model.config.block_size:
        model.crop_block_size(model_config['block_size'])

    model.to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], (config['beta1'], config['beta2']), device_type)

    if config['compile']:
        print("Compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    if ddp:
        model = ShardedDDP(model, optimizer)

    if config['wandb_log'] and master_process:
        initialize_wandb(config, config['wandb_project'], config['wandb_run_name'])

    def get_lr(it):
        if it < config['warmup_iters']:
            return config['learning_rate'] * it / config['warmup_iters']
        if it > config['lr_decay_iters']:
            return config['min_lr']
        decay_ratio = (it - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])

    iter_num = 0
    best_val_loss = 1e9
    X, Y = get_batch(data_dir, config['batch_size'], config['block_size'], device, 'train')
    t0 = time.time()
    local_iter_num = 0
    raw_model = model.module if ddp else model
    running_mfu = -1.0

    while True:
        lr = get_lr(iter_num) if config['decay_lr'] else config['learning_rate']
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if iter_num % config['eval_interval'] == 0 and master_process:
            losses = estimate_loss(model, data_dir, config['eval_iters'], config['block_size'], device, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if config['wandb_log']:
                log_metrics({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,
                })
            if losses['val'] < best_val_loss or config['always_save_checkpoint']:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_config,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt.gz')
                    save_checkpoint_compressed(model, optimizer, iter_num, ckpt_path)
                    print(f"Saving checkpoint to {config['out_dir']}")

        if iter_num == 0 and config['eval_only']:
            break

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch(data_dir, config['batch_size'], config['block_size'], device, 'train')
            scaler.scale(loss).backward()

        if config['grad_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config['log_interval'] == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(config['batch_size'] * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        if iter_num > config['max_iters']:
            break

    if ddp:
        destroy_process_group()

if __name__ == '__main__':
    prepare_data()
    train()
