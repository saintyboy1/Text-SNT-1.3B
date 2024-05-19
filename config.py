# config.py
import torch

# General training configuration
config = {
    'batch_size': 12,
    'gradient_accumulation_steps': 5 * 8,
    'block_size': 1024,
    'max_iters': 600000,
    'eval_interval': 2000,
    'log_interval': 1,
    'eval_iters': 200,
    'eval_only': False,
    'always_save_checkpoint': True,
    'learning_rate': 6e-4,
    'weight_decay': 1e-1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
    'decay_lr': True,
    'warmup_iters': 2000,
    'lr_decay_iters': 600000,
    'min_lr': 6e-5,
    'backend': 'nccl',
    'device': 'cuda',
    'dtype': 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16',
    'compile': True,
    'out_dir': 'out',
    'dataset': 'openwebtext',
    'init_from': 'scratch',
    'wandb_log': False,
    'wandb_project': 'owt',
    'wandb_run_name': 'text-snt-1.3b'
}

# Model configuration
model_config = {
    'n_layer': 24,
    'n_head': 32,
    'n_embd': 2048,
    'block_size': 1024,
    'bias': False,
    'dropout': 0.1,
    'vocab_size': 50304  # Default value, can be updated based on the dataset
}
