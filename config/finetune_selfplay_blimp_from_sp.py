# Self-play SFT on BLiMP, starting from the self-play pretrained model.
# Both pretraining and fine-tuning use self-play — tests whether the two
# effects compound.

out_dir      = 'out-blimp-selfplay-sft-from-sp'
pretrain_dir = 'out-babylm-selfplay'
init_from    = 'finetune'

eval_interval = 100
eval_iters    = 50
log_interval  = 10

always_save_checkpoint = True

wandb_log      = False
wandb_project  = 'blimp-sft'
wandb_run_name = 'selfplay-sft-from-sp'

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
