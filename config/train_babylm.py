# baseline BPE model trained on BabyLM 10M
# ~50M parameters: 8 layers, 8 heads, 512 embedding, vocab 50304
# expect ~3-5s/iter on M2; 5000 iters ≈ 4-7 hours

out_dir = 'out-babylm'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'babylm'
wandb_run_name = 'baseline'

dataset = 'babylm'
gradient_accumulation_steps = 2
batch_size = 32        # effective batch = 32 * 2 = 64 sequences
block_size = 256       # 256 BPE tokens ≈ ~200 words of context

# model — larger than char-level to form meaningful BPE representations
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.1
bias = False

# optimiser — lower lr suits larger models
learning_rate = 3e-4
max_iters = 5000
lr_decay_iters = 5000
min_lr = 3e-5          # learning_rate / 10
beta2 = 0.95
weight_decay = 1e-1
warmup_iters = 200

# device = 'mps'
# compile = False
