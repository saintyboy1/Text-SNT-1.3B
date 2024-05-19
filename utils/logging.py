# utils/logging.py
import wandb

def initialize_wandb(config, project, run_name):
    wandb.login() # login first
    
    wandb.init(project=project, name=run_name, config=config)
    
def log_metrics(metrics):
    wandb.log(metrics)
