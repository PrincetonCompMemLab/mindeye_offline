# verify_lr.py (Final "Fast-Forward" Version)

import torch
import torch.nn as nn

# --- 1. FAKE ARGUMENTS & MODEL ---
class Args:
    use_prior = True
    max_lr = 3e-4
    prior_lr = 3e-5
    lr_scheduler_type = 'cycle'
    num_epochs = 150
    num_iterations_per_epoch = 100
    # Add pct_start to match your main script's scheduler config
    pct_start = 2 / 150 

args = Args()

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ridge = nn.Linear(10, 10)
        self.backbone = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
        self.diffusion_prior = nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10))
model = FakeModel()

print(f"--- Running Dry Run with: max_lr={args.max_lr}, prior_lr={args.prior_lr} ---")

# --- 2. THE CORE LOGIC ---
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
effective_prior_lr = args.prior_lr if args.prior_lr is not None else args.max_lr
opt_grouped_parameters = [
    {'params': [p for n, p in model.ridge.named_parameters()]},
    {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)]},
    {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)]},
]
if args.use_prior:
    opt_grouped_parameters.extend([
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': effective_prior_lr},
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'lr': effective_prior_lr}
    ])
optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=args.max_lr)
lr_scheduler = None
if args.lr_scheduler_type == 'cycle':
    total_steps = int(args.num_epochs * args.num_iterations_per_epoch)
    max_lrs = [args.max_lr] * 3
    if args.use_prior:
        max_lrs.extend([effective_prior_lr] * 2)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=args.pct_start
    )

# --- 3. THE DEFINITIVE "FAST-FORWARD" VERIFICATION ---
print("\n--- Verifying Final LR Configuration by Simulating Training ---")

if args.lr_scheduler_type == 'cycle' and lr_scheduler is not None:
    # Calculate the step number where the LR should be at its peak
    peak_step = int(total_steps * args.pct_start)
    
    print(f"Fast-forwarding scheduler by {peak_step} steps to reach peak LR...")
    # Manually step the scheduler forward to the peak
    for _ in range(peak_step):
        lr_scheduler.step()
        
    print("\nOptimizer's Learning Rates AT THE PEAK:")
    # Now, inspect the LRs in the optimizer. They should match the max_lrs list.
    for i, group in enumerate(optimizer.param_groups):
         print(f"  Group {i}: {group['lr']:.2e}")
    print("(Note: Values may have tiny floating point inaccuracies but should match your max LRs)")
else:
    print("Scheduler not created.")

print("------------------------------------------------------------\n")