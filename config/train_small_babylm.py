# Small BPE model trained on BabyLM 10M (~3.2M transformer params, ~16M total)
# Scaling ladder: Small → Medium → Large → XL
# head_dim = n_embd / n_head = 256 / 4 = 64 (held constant across ladder)

out_dir = 'outputs/scaling/small/baseline'
eval_interval = 250
eval_iters = 200
log_interval = 10
checkpoint_interval = 4000  # save named snapshots for gap-over-training analysis

always_save_checkpoint = True

wandb_log = False
wandb_project = 'babylm-scaling'
wandb_run_name = 'small-baseline'

dataset = 'babylm'
gradient_accumulation_steps = 2
batch_size = 16
block_size = 128

n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 3e-5
beta2 = 0.95
weight_decay = 1e-1
warmup_iters = 200
