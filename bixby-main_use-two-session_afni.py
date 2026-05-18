#!/usr/bin/env python
# coding: utf-8

# In[11]:


import os
import sys
from tqdm import tqdm
import pickle
import argparse
import numpy as np
from collections import defaultdict

import torch
from scipy.stats import zscore
from torchvision import transforms

import utils

#sessions = ['01', '02', 'ses-04_study', 'ses-04_test', 'ses-04_snap']
           #'ses-05_study', 'ses-05_test', 'ses-05_snap']

sessions = ['01', '02', 'snap', 'study', 'test']


# In[12]:


### Set up

def is_interactive():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Running in Jupyter Notebook or JupyterLab
        elif shell == 'TerminalInteractiveShell':
            return False # Running in IPython terminal
        else:
            return False # Other interactive shells
    except NameError:
        return False  # Not running in an IPython environment


# In[13]:


if is_interactive():
    print("Code is running in a Jupyter Notebook. Using the following variables")
    
    
    sub_list = ['sub-01']
    sub = sub_list[0]
    
    suffix="_avgrepeats_unionmask_150epochs" 
    model_name = f"{sub}_2-session_task-mindeye_jupyter_{suffix}"

    batch_size = 8
    max_lr=3e-4
    mixup_pct=.33
    num_epochs=30
    use_prior=False
    prior_scale=None
    clip_scale=1.

    use_image_aug=False
    
    n_blocks=4
    hidden_dim=1024
    
    ckpt_interval = 99
    ckpt_saving = True
    
    wandb_log = False

    seed = 315
    
else:
    print("Code is running in a standard Python interpreter or IPython terminal.")
    
    parser = argparse.ArgumentParser(description="Parameters.")
    parser.add_argument(
        "-s",
        "--subj",
        action="store",
        #nargs="*",
        help=(
            "One or more subject identifiers (e.g., sub-01)."
            "If this is omitted, using a pre-defined subject_list."
        ),
    )
    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size can be increased by 10x if only training v2c and not diffusion diffuser",
    )
    parser.add_argument(
        "--max_lr",type=float,default=3e-4,
        )
    parser.add_argument(
        "--mixup_pct",type=float,default=.33,
        )
    parser.add_argument(
        "--num_epochs",type=int,default=120,
        help="number of epochs of training",
        )
    parser.add_argument(
        "--use_prior",action=argparse.BooleanOptionalAction,default=False,
        help="whether to train diffusion prior (True) or just rely on retrieval part of the pipeline (False)",
        )
    parser.add_argument(
        "--prior_scale",type=float,default=30,
        help="multiply diffusion prior loss by this",
    )
    parser.add_argument(
        "--clip_scale",type=float,default=1.,
        help="multiply contrastive loss by this number",
    )
    parser.add_argument(
        "--use_image_aug",action=argparse.BooleanOptionalAction,default=True,
        help="whether to use image augmentation",
    )
    parser.add_argument(
        "--n_blocks",type=int,default=2,
    )
    parser.add_argument(
        "--hidden_dim",type=int,default=1024,
    )
    parser.add_argument(
        "--ckpt_interval",type=int,default=5,
        help="save backup ckpt and reconstruct every x epochs",
    )
    parser.add_argument(
        "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
    )
    parser.add_argument(
        "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
        help="whether to log to wandb",
    )
    parser.add_argument(
        "--seed",type=int,default=42,
    )
    args = parser.parse_args()
    
    model_name = args.model_name
    sub = args.subj
    batch_size = args.batch_size
    max_lr = args.max_lr
    mixup_pct = args.mixup_pct
    num_epochs = args.num_epochs
    use_prior = args.use_prior
    prior_scale = args.prior_scale
    clip_scale = args.clip_scale
    use_image_aug = args.use_image_aug
    n_blocks = args.n_blocks
    hidden_dim = args.hidden_dim
    ckpt_interval = args.ckpt_interval
    ckpt_saving = args.ckpt_saving
    wandb_log = args.wandb_log
    seed = args.seed


# In[14]:


utils.seed_everything(seed)


# In[15]:


dic = {}

data_folder = '/scratch/gpfs/KNORMAN/wanjia/mindeye_testing/real_time_mindEye2/bixby_data/'

folder_path = os.path.join(data_folder, 'afni')

with open(f'{folder_path}/{sub}_roi_vox_all_sessions.pkl', 'rb') as file:
    dic[sub] = pickle.load(file)


# In[16]:


print(dic[sub].keys())
del dic[sub]['03']
del dic[sub]['union_mask']
print(dic[sub].keys())


# In[17]:


union_mask = dic[sub]['union_mask_2sess']
print('NSD mask size:', union_mask.shape)
print('union mask size:', sum(union_mask))
for ses in sessions:
    dic[sub][ses]['roi'] = dic[sub][ses]['roi'][:, union_mask]
    # z-score each session
    dic[sub][ses]['roi'] = np.nan_to_num(zscore(dic[sub][ses]['roi'], axis=0))
    s = dic[sub][ses]['roi'].shape
    print(f'{ses}: {s}')


# In[18]:


# 455 * 3 + 26 (13 pairs; 3 repeats) + 80 (2 repeats per session_
unique_images = list(set(dic[sub]['01']['trial'] + dic[sub]['02']['trial']))
print(len(unique_images))
test_unique_images = [f'A_{i}' for i in range(1,19)] + [f'B_{i}' for i in range(1,19)]


# In[19]:


import imageio.v2 as imageio
resize_transform = transforms.Resize((224, 224))

images = None

img_path = f'{folder_path}/loaded_mindeye_imgs.pkl'
idx_path = f'{folder_path}/loaded_mindeye_idxs.pkl'

# if os.path.exists(img_path):
#     with open(img_path, 'rb') as file:
#         images = pickle.load(file)
#         print('Loading image saved at: ', file)
#     with open(idx_path, 'rb') as file:
#         unique_images = pickle.load(file)
#         print('Loading image saved at: ', file)
# else:
for img in tqdm(unique_images):

    root_dir = os.path.join(data_folder, 'stimuli')
    if 'unchosen' in img:
        image_file = f'{root_dir}/unchosen_nsd_1000_images/{img}.png'
    elif 'special' in img and 'notspecial' not in img:
        image_file = f'{root_dir}/special515/{img}.jpg'
    elif 'notspecial' in img:
        image_file = f'{root_dir}/shared1000_notspecial/{img}.png'
    elif 'pair_' and '_w_' in img:
        image_file = f'{root_dir}/MST_pairs/{img}.jpg'
    else:
        print(img)

    if image_file and not os.path.exists(image_file):
        print('Cannot find the image at this path',image_file)
        break

    im = imageio.imread(image_file)
    im = torch.Tensor(im / 255).permute(2,0,1)
    im = resize_transform(im.unsqueeze(0))

    if images is None:
        images = im
    else:
        images = torch.vstack((images, im))
    
#     print(folder_path)
#     with open(img_path, 'wb') as file:
#         pickle.dump(images, file)
#         print('image saved at: ', file)
        
#     with open(idx_path, 'wb') as file:
#         pickle.dump(unique_images, file)
#         print('image idx saved at: ', file)
        
print("images", images.shape)


# In[20]:


test_img = None

img_path = f'{folder_path}/loaded_test_imgs.pkl'

# if os.path.exists(img_path):
#     with open(img_path, 'rb') as file:
#         test_img = pickle.load(file)
#         print('Loading image saved at: ', file)
# else:
for img in test_unique_images:

    root_dir = os.path.join(data_folder, 'stimuli', 'scenes')
    img_list = img.split('_')[0]
    img_id = int(img.split('_')[1])

    image_file = f'{root_dir}/list{img_list}/{img_id:02d}.png'

    if image_file and not os.path.exists(image_file):
        print('Cannot find the image at this path',image_file)
        break

    im = imageio.imread(image_file)
    im = torch.Tensor(im / 255).permute(2,0,1)
    im = resize_transform(im.unsqueeze(0))

    if test_img is None:
        test_img = im
    else:
        test_img = torch.vstack((test_img, im))


#     print(folder_path)
#     with open(img_path, 'wb') as file:
#         pickle.dump(test_img, file)
#         print('image saved at: ', file)
        
# print("testing images", test_img.shape)


# In[21]:


def find_repeated_strings(string_list):
    """
    Finds all repeated strings in a list and returns a dictionary 
    with the string as the key and a list of its indices as the value.
    Uses a set to track seen items efficiently.
    """
    # Set to quickly track which items have appeared once already
    seen_once = set()
    # Dictionary to store only the indices of items that repeat
    repeated_strings_dict = {}

    for index, string_val in enumerate(string_list):
        if string_val in repeated_strings_dict:
            # If already in the 'repeated_strings_dict', just append the new index
            repeated_strings_dict[string_val].append(index)
        elif string_val in seen_once:
            # First time seeing a repeat: move from 'seen_once' to 'repeated_strings_dict'
            repeated_strings_dict[string_val] = [string_list.index(string_val), index]
        else:
            # First time seeing the item overall
            seen_once.add(string_val)
            
    return repeated_strings_dict

def locate_repeat_index_per_run(sub_dict, unique_idx):
    
    vox = sub_dict['roi']
    runs = sub_dict['run']
    unique_runs = list(set(runs))
    trials = ["_".join(trial.split('_')[1:-1]) for trial in sub_dict['trial']]
    repeated_trial = find_repeated_strings(trials)

    # structure output idx dictionary
    unique_runs.sort()
    default_value = {}
    sorted_vox = dict.fromkeys(unique_runs, default_value)
    for k in sorted_vox.keys():
        sorted_vox[k] = defaultdict(list)
        
    for trial in test_unique_images:
        idx_list = repeated_trial[trial]
        for i in idx_list:
            curr_run = runs[i]
            sorted_vox[curr_run][trial].append(i)
    
    return repeated_trial, sorted_vox


# In[22]:


def average_repeats(vox, mindeye_trial, unique_images):
    
    repeated_trial = find_repeated_strings(mindeye_trial)
    
    sorted_vox = np.zeros((len(unique_images), vox.shape[1]))
    
    # Average repeated MST images
    for i, img in enumerate(unique_images):

        if img in repeated_trial.keys(): # deal with repeated images
            # average all repeats across sessions
            curr_trial_vox = np.mean(vox[repeated_trial[img]], axis=0)
            
        elif img in mindeye_trial: # deal with once images
            idx = mindeye_trial.index(img)
            curr_trial_vox = vox[idx, :]
            
        else: # error handeling
            print(f"{img} is not in the list")
            break
        
        sorted_vox[i, :] = curr_trial_vox
    
    return sorted_vox


def average_repeats_snap(vox, repeated_trial, unique_images):
        
    sorted_vox = np.zeros((len(unique_images), vox.shape[1]))
    assert len(repeated_trial.keys()) == len(unique_images)
    
    # Average repeated MST images
    for i, img in enumerate(unique_images):

        if img in repeated_trial.keys(): # deal with repeated images
            # average all repeats across sessions
            curr_trial_vox = np.mean(vox[repeated_trial[img]], axis=0)
            
        else: # error handeling
            print(f"{img} is not in the list")
            break
        
        sorted_vox[i, :] = curr_trial_vox
    
    return sorted_vox


# In[23]:


# Stacking multi-session data:
vox_data = {}

mindeye_vox = np.vstack((dic[sub]['01']['roi'],dic[sub]['02']['roi']))
mindeye_trial = dic[sub]['01']['trial']+dic[sub]['02']['trial']
mindeye_vox = average_repeats(mindeye_vox, mindeye_trial, unique_images)
vox_data[sub] = mindeye_vox


# In[24]:


# Loading Snap data
tasks = ['snap', 'study', 'test']
#tasks = ['ses-04_study', 'ses-04_test', 'ses-04_snap', 'ses-05_study', 'ses-05_test', 'ses-05_snap']

test_data = {}

test_data[sub] = {}

for task in tasks:

    sub_dict = dic[sub][task]

    repeat_idx, per_run_repeat_idx = locate_repeat_index_per_run(sub_dict, test_unique_images)

    test_data[sub][task] = average_repeats_snap(sub_dict['roi'], repeat_idx, test_unique_images)


# ### Testing single subject

# In[27]:


train_images = torch.Tensor(images)
train_vox = torch.Tensor(vox_data[sub])
assert len(train_images) == len(train_vox)


# In[28]:


print('train images shape:', train_images.shape)
print('train vox shape:', train_vox.shape)
#tasks = ['ses-04_study', 'ses-04_test', 'ses-04_snap', 'ses-05_study', 'ses-05_test', 'ses-05_snap']


# In[29]:


test_images = torch.Tensor(test_img)
test_vox = torch.Tensor(np.mean([test_data[sub]['study'], test_data[sub]['test'], test_data[sub]['snap']], axis=0))
test_vox_study = torch.Tensor(test_data[sub]['study'])
test_vox_test = torch.Tensor(test_data[sub]['test'])
test_vox_snap = torch.Tensor(test_data[sub]['snap'])
assert len(test_images) == len(test_vox)


# In[20]:


# test_images = torch.Tensor(test_img)
# test_vox = torch.Tensor(np.mean([test_data[sub]['ses-04_study'], test_data[sub]['ses-04_test'], test_data[sub]['ses-04_snap']], axis=0))
# test_vox_study = torch.Tensor(test_data[sub]['ses-04_study'])
# test_vox_test = torch.Tensor(test_data[sub]['ses-04_test'])
# test_vox_snap = torch.Tensor(test_data[sub]['ses-04_snap'])
# assert len(test_images) == len(test_vox)


# In[33]:


# test_vox_2 = torch.Tensor(np.mean([test_data[sub]['ses-05_study'], test_data[sub]['ses-05_test'], test_data[sub]['ses-05_snap']], axis=0))
# test_vox_study_2 = torch.Tensor(test_data[sub]['ses-05_study'])
# test_vox_test_2 = torch.Tensor(test_data[sub]['ses-05_test'])
# test_vox_snap_2 = torch.Tensor(test_data[sub]['ses-05_snap'])
# assert len(test_images) == len(test_vox_2)


# In[30]:


print('test images shape:', test_images.shape)
print('test vox shape ses04:', test_vox.shape)
#print('test vox shape ses05:', test_vox_2.shape)


# In[31]:


assert train_vox.shape[1] == test_vox.shape[1] #== test_vox_2.shape[1]


# ## Finished loading data. Setting up GPU

# In[32]:


### Multi-GPU config ###
from accelerate import Accelerator, DeepSpeedPlugin

local_rank = os.getenv('RANK')
if local_rank is None: 
    local_rank = 0
else:
    local_rank = int(local_rank)
print("LOCAL RANK ", local_rank)  

data_type = torch.float32 # change depending on your mixed_precision

accelerator = Accelerator(split_batches=False)


# In[33]:


print("PID of this process =",os.getpid())
device = accelerator.device
print("device:",device)
world_size = accelerator.state.num_processes
distributed = not accelerator.state.distributed_type == 'NO'
num_devices = torch.cuda.device_count()
global_batch_size = batch_size * num_devices
print("global_batch_size", global_batch_size)
if num_devices==0 or not distributed: num_devices = 1
num_workers = num_devices
print(accelerator.state)

# set data_type to match your mixed precision (automatically set based on deepspeed config)
if accelerator.mixed_precision == "bf16":
    data_type = torch.bfloat16
elif accelerator.mixed_precision == "fp16":
    data_type = torch.float16
else:
    data_type = torch.float32

print("distributed =",distributed, "num_devices =", num_devices, "local rank =", local_rank, "world size =", world_size, "data_type =", data_type)
print = accelerator.print # only print if local_rank=0


# In[34]:


## USING OpenCLIP ViT-bigG ###
sys.path.append('generative_models/')
import sgm
from generative_models.sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
# from generative_models.sgm.models.diffusion import DiffusionEngine
# from omegaconf import OmegaConf


# In[35]:


try:
    print(clip_img_embedder)
except:
    clip_img_embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    )
    clip_img_embedder.to(device)
clip_img_embedder.model.visual.set_grad_checkpointing(True)
clip_seq_dim = 256
clip_emb_dim = 1664


# In[36]:


num_voxels_list=[train_vox[0].shape[-1]]


# In[37]:


num_voxels_list


# In[38]:


from models import PriorNetwork, BrainDiffusionPrior


# In[39]:


model = utils.prepare_model_and_training(
    num_voxels_list=num_voxels_list,
    n_blocks=n_blocks,
    hidden_dim=hidden_dim,
    clip_emb_dim=clip_emb_dim,
    clip_seq_dim=clip_seq_dim,
    use_prior=use_prior,
    clip_scale=clip_scale
)


# In[40]:


# test on subject 1 with fake data
b = torch.randn((2,1,num_voxels_list[0]))
print(b.shape, model.ridge(b,0).shape)


# In[41]:


# test that the model works on some fake data
b = torch.randn((2,1,hidden_dim))
print("b.shape",b.shape)

backbone_, clip_, blur_ = model.backbone(b)
print(backbone_.shape, clip_.shape, blur_[0].shape, blur_[1].shape)


# ## Setup optimizer / lr / ckpt saving

# In[42]:


prior_lr=3e-4
lr_scheduler_type='cycle'
num_iterations_per_epoch=len(train_images)//batch_size

import time
ts = time.time()
outdir = os.path.join(data_folder, f'output_{sub}_{model_name}_{ts}')
if not os.path.exists(outdir) and ckpt_saving:
    os.makedirs(outdir,exist_ok=True)


# In[43]:


no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

opt_grouped_parameters = [
    {'params': [p for n, p in model.ridge.named_parameters()], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2},
    {'params': [p for n, p in model.backbone.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]
# model.backbone.requires_grad_(False)

if use_prior:
    effective_prior_lr = prior_lr if prior_lr is not None else max_lr
    print(f"--- Setting learning rate for diffusion_prior: {effective_prior_lr} ---")

    if prior_lr is not None:
        assert lr_scheduler_type == 'cycle'  # if prior_lr exists, ensure lr scheduler is cycle because we want to set custom lr for the prior. custom lr for prior is not implemented in the linear scheduler code.

    opt_grouped_parameters.extend([
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-2, 'lr': effective_prior_lr},
        {'params': [p for n, p in model.diffusion_prior.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': effective_prior_lr}
    ])

optimizer = torch.optim.AdamW(opt_grouped_parameters, lr=max_lr)

if lr_scheduler_type == 'linear':
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        total_iters=int(np.floor(num_epochs*num_iterations_per_epoch)),
        last_epoch=-1
    )
elif lr_scheduler_type == 'cycle':
    if num_iterations_per_epoch==0:
        num_iterations_per_epoch=1
    total_steps=int(np.floor(num_epochs*num_iterations_per_epoch))
    print("total_steps", total_steps)
    max_lrs = [max_lr] * 3  # for ridge and backbone
    if use_prior:
        max_lrs.extend([effective_prior_lr] * 2) # for prior

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lrs,
        total_steps=total_steps,
        final_div_factor=1000,
        last_epoch=-1, pct_start=2/num_epochs
    )
    
def save_ckpt(tag):
    ckpt_path = outdir+f'/{tag}.pth'
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'train_losses': losses,
            'test_losses': test_losses,
            'lrs': lrs,
            }, ckpt_path)
    print(f"\n---saved {outdir}/{tag} ckpt!---\n")
    
def load_ckpt(tag,load_lr=True,load_optimizer=True,load_epoch=True,strict=True,outdir=outdir,multisubj_loading=False): 
    print(f"\n---loading {outdir}/{tag}.pth ckpt---\n")
    checkpoint = torch.load(outdir+'/last.pth', map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    if multisubj_loading: # remove incompatible ridge layer that will otherwise error
        state_dict.pop('ridge.linears.0.weight',None)
    model.load_state_dict(state_dict, strict=strict)
    if load_epoch:
        globals()["epoch"] = checkpoint['epoch']
        print("Epoch",epoch)
    if load_optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if load_lr:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    del checkpoint

print("\nDone with model preparations!")
num_params = utils.count_params(model)


# In[44]:


epoch = 0
losses, test_losses, lrs = [], [], []
best_test_loss = 1e9
torch.cuda.empty_cache()
# tf32 data type is faster than standard float32
torch.backends.cuda.matmul.allow_tf32 = True


# In[45]:


# load multisubject stage1 ckpt if set
load_ckpt("last",outdir='/scratch/gpfs/KNORMAN/ri4541/MindEyeV2/src/mindeyev2/train_logs/multisubject_subj01_1024hid_nolow_300ep',load_lr=False,load_optimizer=False,load_epoch=False,strict=False,multisubj_loading=True)


# In[46]:


train_data = torch.utils.data.TensorDataset(torch.tensor(range(len(train_vox))))
train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True)

test_data = torch.utils.data.TensorDataset(torch.tensor(range(len(test_vox))))
test_dl = torch.utils.data.DataLoader(test_data, batch_size=36, shuffle=False, drop_last=True, pin_memory=True)


# In[47]:


model, optimizer, train_dl, lr_scheduler = accelerator.prepare(model, optimizer, train_dl, lr_scheduler)


# In[48]:


import torch.nn as nn


# In[49]:


for train_i, behav in enumerate(train_dl):  
    print(train_i)
    print(behav)
    print(behav[0])
    break


# In[50]:


for test_i, behav in enumerate(test_dl):  
    print(test_i)
    print(behav)
    break


# In[51]:


wandb_log = False


# In[52]:


clip_scale


# In[ ]:


print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
test_image, test_voxel = None, None
mse = nn.MSELoss()
l1 = nn.L1Loss()
soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
skip_train = True if epoch>=(num_epochs-1) else False # skip training if you are resuming from a fully trained model

for epoch in progress_bar:
    model.train()

    fwd_percent_correct = 0.
    bwd_percent_correct = 0.
    test_fwd_percent_correct1 = 0.
    test_bwd_percent_correct1 = 0.
    test_fwd_percent_correct2 = 0.
    test_bwd_percent_correct2 = 0.
    test_fwd_percent_correct3 = 0.
    test_bwd_percent_correct3 = 0.
    test_fwd_percent_correct4 = 0.
    test_bwd_percent_correct4 = 0.
    
    recon_cossim = 0.
    test_recon_cossim = 0.
    recon_mse = 0.
    test_recon_mse = 0.

    loss_clip_total = 0.
    loss_blurry_total = 0.
    loss_blurry_cont_total = 0.
    test_loss_clip_total = 0.
    
    loss_prior_total = 0.
    test_loss_prior_total = 0.

    blurry_pixcorr = 0.
    test_blurry_pixcorr = 0. 

    # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
    for train_i, behav in enumerate(train_dl):  
        with torch.cuda.amp.autocast(dtype=data_type):
            optimizer.zero_grad()
            loss = 0.
            
            behav = behav[0]

            image = train_images[behav.long().cpu()].to(device)
            voxel = train_vox[behav.long().cpu()]

            # voxel = (voxel - train_mean) / train_std
            voxel = torch.Tensor(voxel).unsqueeze(1).to(device)

            if use_image_aug: 
                image = img_augment(image)

            clip_target = clip_img_embedder(image)
            assert not torch.any(torch.isnan(clip_target))

            if epoch < int(mixup_pct * num_epochs):
                voxel, perm, betas, select = utils.mixco(voxel)

            voxel_ridge = model.ridge(voxel,0) #[model.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]
            # voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

            backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

            if clip_scale>0:
                clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

            if use_prior:
                loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
                loss_prior_total += loss_prior.item()
                loss_prior *= prior_scale
                loss += loss_prior

                recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
                recon_mse += mse(prior_out, clip_target).item()

            if clip_scale>0:
                if epoch < int(mixup_pct * num_epochs):                
                    loss_clip = utils.mixco_nce(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006,
                        perm=perm, betas=betas, select=select)
                else:
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                loss_clip_total += loss_clip.item()
                loss_clip *= clip_scale
                loss += loss_clip

            if clip_scale>0:
                # forward and backward top 1 accuracy        
                labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
                fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
                bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
            
            utils.check_loss(loss)
            accelerator.backward(loss)
            optimizer.step()

            losses.append(loss.item())
            lrs.append(optimizer.param_groups[0]['lr'])

            if lr_scheduler_type is not None:
                lr_scheduler.step()
                
            if train_i >= num_iterations_per_epoch-1:
                break
                
    model.eval()
    if local_rank==0:
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
            for test_i, behav in enumerate(test_dl):  
                behav = behav[0]

                loss=0.

                if behav.ndim>1:
                    image = test_images[behav[:,0].long().cpu()].to(device)
                    voxel = test_vox[behav.long().cpu()].mean(1)
                else:
                    image = test_images[behav.long().cpu()].to(device)
                    voxel1 = test_vox[behav.long().cpu()]
                    voxel2 = test_vox_study[behav.long().cpu()]
                    voxel3 = test_vox_test[behav.long().cpu()]
                    voxel4 = test_vox_snap[behav.long().cpu()]
                    
                voxel1 = torch.Tensor(voxel1).unsqueeze(1).to(device)
                voxel2 = torch.Tensor(voxel2).unsqueeze(1).to(device)
                voxel3 = torch.Tensor(voxel3).unsqueeze(1).to(device)
                voxel4 = torch.Tensor(voxel4).unsqueeze(1).to(device)


                clip_img_embedder = clip_img_embedder.to(device)
                clip_target = clip_img_embedder(image.float())
                
                voxel_ridge1 = model.ridge(voxel1,0)
                voxel_ridge2 = model.ridge(voxel2,0)
                voxel_ridge3 = model.ridge(voxel3,0)
                voxel_ridge4 = model.ridge(voxel4,0)
                
                backbone, clip_voxels1, blurry_image_enc_ = model.backbone(voxel_ridge1)                
                backbone, clip_voxels2, blurry_image_enc_ = model.backbone(voxel_ridge2)                
                backbone, clip_voxels3, blurry_image_enc_ = model.backbone(voxel_ridge3)               
                backbone, clip_voxels4, blurry_image_enc_ = model.backbone(voxel_ridge4)

                if clip_scale>0:
                    clip_voxels_norm1 = nn.functional.normalize(clip_voxels1.flatten(1), dim=-1)
                    clip_voxels_norm2 = nn.functional.normalize(clip_voxels2.flatten(1), dim=-1)
                    clip_voxels_norm3 = nn.functional.normalize(clip_voxels3.flatten(1), dim=-1)
                    clip_voxels_norm4 = nn.functional.normalize(clip_voxels4.flatten(1), dim=-1)
                    
                    clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
                # for some evals, only doing a subset of the samples per batch because of computational cost
                random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                
                if use_prior:
                    loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
                    test_loss_prior_total += loss_prior.item()
                    loss_prior *= prior_scale
                    loss += loss_prior
                        
                if clip_scale>0:
                    loss_clip = utils.soft_clip_loss(
                        clip_voxels_norm1,
                        clip_target_norm,
                        temp=.006)

                    test_loss_clip_total += loss_clip.item()
                    loss_clip = loss_clip * clip_scale
                    loss += loss_clip

                if clip_scale>0:
                    # forward and backward top 1 accuracy        
                    labels1 = torch.arange(len(clip_voxels_norm1)).to(clip_voxels_norm1.device) 
                    test_fwd_percent_correct1 += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm1, clip_target_norm), labels1, k=1).item()
                    test_bwd_percent_correct1 += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm1), labels1, k=1).item()
                    # forward and backward top 1 accuracy        
                    labels2 = torch.arange(len(clip_voxels_norm2)).to(clip_voxels_norm2.device) 
                    test_fwd_percent_correct2 += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm2, clip_target_norm), labels2, k=1).item()
                    test_bwd_percent_correct2 += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm2), labels2, k=1).item()
                    # forward and backward top 1 accuracy        
                    labels3 = torch.arange(len(clip_voxels_norm3)).to(clip_voxels_norm3.device) 
                    test_fwd_percent_correct3 += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm3, clip_target_norm), labels3, k=1).item()
                    test_bwd_percent_correct3 += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm3), labels3, k=1).item()
                    # forward and backward top 1 accuracy        
                    labels4 = torch.arange(len(clip_voxels_norm4)).to(clip_voxels_norm4.device) 
                    test_fwd_percent_correct4 += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm4, clip_target_norm), labels4, k=1).item()
                    test_bwd_percent_correct4 += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm4), labels4, k=1).item()
                
                utils.check_loss(loss)                
                test_losses.append(loss.item())

            # if utils.is_interactive(): clear_output(wait=True)
            if skip_train: break
            print("---")

            # assert (test_i+1) == 1
            logs = {"train/loss": np.mean(losses[-(train_i+1):]),
                "test/loss": np.mean(test_losses[-(test_i+1):]),
                "train/lr": lrs[-1],
                "train/num_steps": len(losses),
                "test/num_steps": len(test_losses),
                "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
                "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
                "test/test_fwd_pct_correct overall": test_fwd_percent_correct1 / (test_i + 1),
                "test/test_bwd_pct_correct overall": test_bwd_percent_correct1 / (test_i + 1),
                "test/test_fwd_pct_correct study": test_fwd_percent_correct2 / (test_i + 1),
                "test/test_bwd_pct_correct study": test_bwd_percent_correct2 / (test_i + 1),
                "test/test_fwd_pct_correct test": test_fwd_percent_correct3 / (test_i + 1),
                "test/test_bwd_pct_correct test": test_bwd_percent_correct3 / (test_i + 1),
                "test/test_fwd_pct_correct snap": test_fwd_percent_correct4 / (test_i + 1),
                "test/test_bwd_pct_correct snap": test_bwd_percent_correct4 / (test_i + 1),
                "train/loss_clip_total": loss_clip_total / (train_i + 1),
                #"train/loss_blurry_total": loss_blurry_total / (train_i + 1),
                #"train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
                "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
                #"train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
                #"test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
                # "train/recon_cossim": recon_cossim / (train_i + 1),
                # "test/recon_cossim": test_recon_cossim / (test_i + 1),
                # "train/recon_mse": recon_mse / (train_i + 1),
                # "test/recon_mse": test_recon_mse / (test_i + 1),
                "train/loss_prior": loss_prior_total / (train_i + 1),
                "test/loss_prior": test_loss_prior_total / (test_i + 1),
                }

            progress_bar.set_postfix(**logs)

            if wandb_log: wandb.log(logs)
            
    # Save model checkpoint and reconstruct
    if (ckpt_saving) and (epoch % ckpt_interval == 0):
        save_ckpt(f'last')

    # wait for other GPUs to catch up if needed
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()

print("\n===Finished!===\n")
if ckpt_saving:
    save_ckpt(f'last')


# In[ ]:





# In[39]:


# print(f"{model_name} starting with epoch {epoch} / {num_epochs}")
# progress_bar = tqdm(range(epoch,num_epochs), ncols=1200, disable=(local_rank!=0))
# test_image, test_voxel = None, None
# mse = nn.MSELoss()
# l1 = nn.L1Loss()
# soft_loss_temps = utils.cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
# skip_train = True if epoch>=(num_epochs-1) else False # skip training if you are resuming from a fully trained model

# for epoch in progress_bar:
#     model.train()

#     fwd_percent_correct = 0.
#     bwd_percent_correct = 0.
#     test_fwd_percent_correct = 0.
#     test_bwd_percent_correct = 0.
    
#     recon_cossim = 0.
#     test_recon_cossim = 0.
#     recon_mse = 0.
#     test_recon_mse = 0.

#     loss_clip_total = 0.
#     loss_blurry_total = 0.
#     loss_blurry_cont_total = 0.
#     test_loss_clip_total = 0.
    
#     loss_prior_total = 0.
#     test_loss_prior_total = 0.

#     blurry_pixcorr = 0.
#     test_blurry_pixcorr = 0. 

#     # you now have voxel_iters and image_iters with num_iterations_per_epoch batches each
#     for train_i, behav in enumerate(train_dl):  
#         with torch.cuda.amp.autocast(dtype=data_type):
#             optimizer.zero_grad()
#             loss = 0.
            
#             behav = behav[0]

#             image = train_images[behav.long().cpu()].to(device)
#             voxel = train_vox[behav.long().cpu()]

#             # voxel = (voxel - train_mean) / train_std
#             voxel = torch.Tensor(voxel).unsqueeze(1).to(device)

#             if use_image_aug: 
#                 image = img_augment(image)

#             clip_target = clip_img_embedder(image)
#             assert not torch.any(torch.isnan(clip_target))

#             if epoch < int(mixup_pct * num_epochs):
#                 voxel, perm, betas, select = utils.mixco(voxel)

#             voxel_ridge = model.ridge(voxel,0) #[model.ridge(voxel_list[si],si) for si,s in enumerate(subj_list)]
#             # voxel_ridge = torch.cat(voxel_ridge_list, dim=0)

#             backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

#             if clip_scale>0:
#                 clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
#                 clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)

#             if use_prior:
#                 loss_prior, prior_out = model.diffusion_prior(text_embed=backbone, image_embed=clip_target)
#                 loss_prior_total += loss_prior.item()
#                 loss_prior *= prior_scale
#                 loss += loss_prior

#                 recon_cossim += nn.functional.cosine_similarity(prior_out, clip_target).mean().item()
#                 recon_mse += mse(prior_out, clip_target).item()

#             if clip_scale>0:
#                 if epoch < int(mixup_pct * num_epochs):                
#                     loss_clip = utils.mixco_nce(
#                         clip_voxels_norm,
#                         clip_target_norm,
#                         temp=.006,
#                         perm=perm, betas=betas, select=select)
#                 else:
#                     epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
#                     loss_clip = utils.soft_clip_loss(
#                         clip_voxels_norm,
#                         clip_target_norm,
#                         temp=epoch_temp)

#                 loss_clip_total += loss_clip.item()
#                 loss_clip *= clip_scale
#                 loss += loss_clip

#             if clip_scale>0:
#                 # forward and backward top 1 accuracy        
#                 labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
#                 fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
#                 bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
            
#             utils.check_loss(loss)
#             accelerator.backward(loss)
#             optimizer.step()

#             losses.append(loss.item())
#             lrs.append(optimizer.param_groups[0]['lr'])

#             if lr_scheduler_type is not None:
#                 lr_scheduler.step()
                
#             if train_i >= num_iterations_per_epoch-1:
#                 break
                
#     model.eval()
#     if local_rank==0:
#         with torch.no_grad(), torch.cuda.amp.autocast(dtype=data_type): 
#             for test_i, behav in enumerate(test_dl):  
#                 behav = behav[0]

#                 loss=0.

#                 if behav.ndim>1:
#                     image = test_images[behav[:,0].long().cpu()].to(device)
#                     voxel = test_vox[behav.long().cpu()].mean(1)
#                 else:
#                     image = test_images[behav.long().cpu()].to(device)
#                     voxel = test_vox[behav.long().cpu()]
                    
#                 voxel = torch.Tensor(voxel).unsqueeze(1).to(device)

#                 clip_img_embedder = clip_img_embedder.to(device)
#                 clip_target = clip_img_embedder(image.float())
                
#                 voxel_ridge = model.ridge(voxel,0)

#                 backbone, clip_voxels, blurry_image_enc_ = model.backbone(voxel_ridge)

#                 if clip_scale>0:
#                     clip_voxels_norm = nn.functional.normalize(clip_voxels.flatten(1), dim=-1)
#                     clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1)
                
#                 # for some evals, only doing a subset of the samples per batch because of computational cost
#                 random_samps = np.random.choice(np.arange(len(image)), size=len(image)//5, replace=False)
                
#                 if use_prior:
#                     loss_prior, contaminated_prior_out = model.diffusion_prior(text_embed=backbone[random_samps], image_embed=clip_target[random_samps])
#                     test_loss_prior_total += loss_prior.item()
#                     loss_prior *= prior_scale
#                     loss += loss_prior
                        
#                 if clip_scale>0:
#                     loss_clip = utils.soft_clip_loss(
#                         clip_voxels_norm,
#                         clip_target_norm,
#                         temp=.006)

#                     test_loss_clip_total += loss_clip.item()
#                     loss_clip = loss_clip * clip_scale
#                     loss += loss_clip

#                 if clip_scale>0:
#                     # forward and backward top 1 accuracy        
#                     labels = torch.arange(len(clip_voxels_norm)).to(clip_voxels_norm.device) 
#                     test_fwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1).item()
#                     test_bwd_percent_correct += utils.topk(utils.batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1).item()
                
#                 utils.check_loss(loss)                
#                 test_losses.append(loss.item())

#             # if utils.is_interactive(): clear_output(wait=True)
#             if skip_train: break
#             print("---")

#             # assert (test_i+1) == 1
#             logs = {"train/loss": np.mean(losses[-(train_i+1):]),
#                 "test/loss": np.mean(test_losses[-(test_i+1):]),
#                 "train/lr": lrs[-1],
#                 "train/num_steps": len(losses),
#                 "test/num_steps": len(test_losses),
#                 "train/fwd_pct_correct": fwd_percent_correct / (train_i + 1),
#                 "train/bwd_pct_correct": bwd_percent_correct / (train_i + 1),
#                 "test/test_fwd_pct_correct": test_fwd_percent_correct / (test_i + 1),
#                 "test/test_bwd_pct_correct": test_bwd_percent_correct / (test_i + 1),
#                 "train/loss_clip_total": loss_clip_total / (train_i + 1),
#                 #"train/loss_blurry_total": loss_blurry_total / (train_i + 1),
#                 #"train/loss_blurry_cont_total": loss_blurry_cont_total / (train_i + 1),
#                 "test/loss_clip_total": test_loss_clip_total / (test_i + 1),
#                 #"train/blurry_pixcorr": blurry_pixcorr / (train_i + 1),
#                 #"test/blurry_pixcorr": test_blurry_pixcorr / (test_i + 1),
#                 "train/recon_cossim": recon_cossim / (train_i + 1),
#                 "test/recon_cossim": test_recon_cossim / (test_i + 1),
#                 "train/recon_mse": recon_mse / (train_i + 1),
#                 "test/recon_mse": test_recon_mse / (test_i + 1),
#                 "train/loss_prior": loss_prior_total / (train_i + 1),
#                 "test/loss_prior": test_loss_prior_total / (test_i + 1),
#                 }

#             progress_bar.set_postfix(**logs)

#             if wandb_log: wandb.log(logs)
            
#     # Save model checkpoint and reconstruct
#     if (ckpt_saving) and (epoch % ckpt_interval == 0):
#         save_ckpt(f'last')

#     # wait for other GPUs to catch up if needed
#     accelerator.wait_for_everyone()
#     torch.cuda.empty_cache()

# print("\n===Finished!===\n")
# if ckpt_saving:
#     save_ckpt(f'last')


# In[ ]:





# In[ ]:





# In[ ]:




