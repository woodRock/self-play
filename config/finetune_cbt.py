# Baseline SFT on CBT-NE (Children's Book Test, Named Entities).
# loss_mask = 'answer_only': only the blank-filling token(s) receive gradient.
# pretrain_dir and out_dir are overridden per-seed by the orchestration script.

out_dir      = 'outputs/cbt/bl-sft-bl'
pretrain_dir = 'outputs/babylm/baseline'
init_from    = 'finetune'

eval_interval = 100
eval_iters    = 50
log_interval  = 10

always_save_checkpoint = True

wandb_log      = False
wandb_project  = 'cbt-sft'
wandb_run_name = 'baseline-sft-cbt'

sft_dataset        = 'cbt_ne'
sft_data_dir       = 'data/cbt_ne'
sft_loss_mask_mode = 'answer_only'

gradient_accumulation_steps = 2
batch_size  = 16
block_size  = 128

n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.1
bias     = False

learning_rate  = 1e-5
max_iters      = 1000
lr_decay_iters = 1000
min_lr         = 1e-6
warmup_iters   = 50
beta2          = 0.95
weight_decay   = 1e-1
