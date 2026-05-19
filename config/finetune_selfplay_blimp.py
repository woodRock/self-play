# Self-play SFT on BLiMP grammatical sentences.
# Same as finetune_blimp.py but uses train_selfplay_sft.py which weights
# token positions by how much harder the current model finds them vs its
# frozen opponent — focusing fine-tuning on the hardest syntactic phenomena.

out_dir      = 'out-blimp-selfplay-sft'
pretrain_dir = 'out-babylm'
init_from    = 'finetune'

eval_interval = 100
eval_iters    = 50
log_interval  = 10

always_save_checkpoint = True

wandb_log      = False
wandb_project  = 'blimp-sft'
wandb_run_name = 'selfplay-sft'

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

opponent_update_interval = 50
selfplay_lambda          = 0.5
