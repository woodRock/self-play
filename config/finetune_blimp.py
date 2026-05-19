# Baseline SFT on BLiMP grammatical sentences.
# Loads the pretrained BabyLM checkpoint and fine-tunes with a standard LM objective.

out_dir      = 'out-blimp-sft'
pretrain_dir = 'out-babylm'
init_from    = 'finetune'

eval_interval = 100
eval_iters    = 50
log_interval  = 10

always_save_checkpoint = True

wandb_log      = False
wandb_project  = 'blimp-sft'
wandb_run_name = 'baseline-sft'

dataset     = 'blimp'
gradient_accumulation_steps = 1
batch_size  = 16
block_size  = 128

n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.1
bias     = False

learning_rate  = 1e-4
max_iters      = 1000
lr_decay_iters = 1000
min_lr         = 1e-5
warmup_iters   = 50
beta2          = 0.95
weight_decay   = 1e-1
