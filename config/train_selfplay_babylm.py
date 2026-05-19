# self-play BPE model trained on BabyLM 10M
# identical architecture to train_babylm.py — only out_dir and wandb differ

out_dir = 'out-babylm-selfplay'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'babylm'
wandb_run_name = 'selfplay'

dataset = 'babylm'
gradient_accumulation_steps = 2
batch_size = 32
block_size = 256

n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 3e-5
beta2 = 0.95
weight_decay = 1e-1
warmup_iters = 200

# device = 'mps'
# compile = False
