# baseline BPE model trained on BabyLM 10M
#
# The lm_head (n_embd → vocab 50304) dominates step time on MPS, so we
# tune n_embd and batch rather than depth.  This gives ~24M parameters
# while retaining 6 transformer layers for meaningful representations.
#
# Measured ~5-8s/iter on M2 Air; 5000 iters ≈ 7-11 hours

out_dir = 'outputs/babylm/baseline'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = True  # save every eval so cluster runs can always resume

wandb_log = False
wandb_project = 'babylm'
wandb_run_name = 'baseline'

dataset = 'babylm'
gradient_accumulation_steps = 2
batch_size = 16       # effective batch = 16 * 2 = 32 sequences
block_size = 128      # 128 BPE tokens ≈ ~100 words of context

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
bias = False

learning_rate = 3e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 3e-5
beta2 = 0.95
weight_decay = 1e-1
warmup_iters = 200

# device = 'mps'
# compile = False
