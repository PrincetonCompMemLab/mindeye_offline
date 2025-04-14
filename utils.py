import numpy as np
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import math
import webdataset as wds
import tempfile
from torchvision.utils import make_grid

import json
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import requests
import io
import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')

def seed_everything(seed=0, cudnn_deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
    else:
        ## needs to be False to use conv3D
        print('Note: not using cudnn.deterministic')

def np_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return PIL.Image.fromarray((x.transpose(1, 2, 0)*127.5+128).clip(0,255).astype('uint8'))

def torch_to_Image(x):
    if x.ndim==4:
        x=x[0]
    return transforms.ToPILImage()(x)

def Image_to_torch(x):
    try:
        x = (transforms.ToTensor()(x)[:3].unsqueeze(0)-.5)/.5
    except:
        x = (transforms.ToTensor()(x[0])[:3].unsqueeze(0)-.5)/.5
    return x

def torch_to_matplotlib(x,device=device):
    if torch.mean(x)>10:
        x = (x.permute(0, 2, 3, 1)).clamp(0, 255).to(torch.uint8)
    else:
        x = (x.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    if device=='cpu':
        return x[0]
    else:
        return x.cpu().numpy()[0]

def pairwise_cosine_similarity(A, B, dim=1, eps=1e-8):
    #https://stackoverflow.com/questions/67199317/pytorch-cosine-similarity-nxn-elements
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def batchwise_pearson_correlation(Z, B):
    # Calculate means
    Z_mean = torch.mean(Z, dim=1, keepdim=True)
    B_mean = torch.mean(B, dim=1, keepdim=True)

    # Subtract means
    Z_centered = Z - Z_mean
    B_centered = B - B_mean

    # Calculate Pearson correlation coefficient
    numerator = Z_centered @ B_centered.T
    Z_centered_norm = torch.linalg.norm(Z_centered, dim=1, keepdim=True)
    B_centered_norm = torch.linalg.norm(B_centered, dim=1, keepdim=True)
    denominator = Z_centered_norm @ B_centered_norm.T

    pearson_correlation = (numerator / denominator)
    return pearson_correlation

def batchwise_cosine_similarity(Z,B):
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def prenormed_batchwise_cosine_similarity(Z,B):
    return (Z @ B.T).T

def cosine_similarity(Z,B,l=0):
    Z = nn.functional.normalize(Z, p=2, dim=1)
    B = nn.functional.normalize(B, p=2, dim=1)
    # if l>0, use distribution normalization
    # https://twitter.com/YifeiZhou02/status/1716513495087472880
    Z = Z - l * torch.mean(Z,dim=0)
    B = B - l * torch.mean(B,dim=0)
    cosine_similarity = (Z @ B.T).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def get_non_diagonals(a):
    a = torch.triu(a,diagonal=1)+torch.tril(a,diagonal=-1)
    # make diagonals -1
    a=a.fill_diagonal_(-1)
    return a

def gather_features(image_features, voxel_features, accelerator):  
    all_image_features = accelerator.gather(image_features.contiguous())
    if voxel_features is not None:
        all_voxel_features = accelerator.gather(voxel_features.contiguous())
        return all_image_features, all_voxel_features
    return all_image_features

def soft_clip_loss(preds, targs, temp=0.125): #, distributed=False, accelerator=None):
    # if not distributed:
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    # else:
    #     all_targs = gather_features(targs, None, accelerator)
    #     clip_clip = (targs @ all_targs.T)/temp
    #     brain_clip = (preds @ all_targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def soft_siglip_loss(preds, targs, temp, bias):
    temp = torch.exp(temp)
    
    logits = (preds @ targs.T) * temp + bias
    # diagonals (aka paired samples) should be >0 and off-diagonals <0
    labels = (targs @ targs.T) - 1 + (torch.eye(len(targs)).to(targs.dtype).to(targs.device))

    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels[:len(preds)])) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels[:,:len(preds)])) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco_hard_siglip_loss(preds, targs, temp, bias, perm, betas):
    temp = torch.exp(temp)
    
    probs = torch.diag(betas)
    probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

    logits = (preds @ targs.T) * temp + bias
    labels = probs * 2 - 1
    #labels = torch.eye(len(targs)).to(targs.dtype).to(targs.device) * 2 - 1
    
    loss1 = -torch.sum(nn.functional.logsigmoid(logits * labels)) / len(preds)
    loss2 = -torch.sum(nn.functional.logsigmoid(logits.T * labels)) / len(preds)
    loss = (loss1 + loss2)/2
    return loss

def mixco(voxels, beta=0.15, s_thresh=0.5, perm=None, betas=None, select=None):
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1]*(len(voxels.shape)-1)
    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1
    return voxels, perm, betas, select

def mixco_clip_target(clip_target, perm, select, betas):
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = clip_target[select] * betas[select].reshape(-1, 1) + \
        clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    return clip_target

def mixco_nce(preds, targs, temp=0.1, perm=None, betas=None, select=None, distributed=False, 
              accelerator=None, local_rank=None, bidirectional=True):
    brain_clip = (preds @ targs.T)/temp
    
    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    else:
        loss =  F.cross_entropy(brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
        if bidirectional:
            loss2 = F.cross_entropy(brain_clip.T, torch.arange(brain_clip.shape[0]).to(brain_clip.device))
            loss = (loss + loss2)/2
        return loss
    
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('param counts:\n{:,} total\n{:,} trainable'.format(total, trainable))
    return trainable

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
    
def check_loss(loss):
    if loss.isnan().any():
        raise ValueError('NaN loss')

def cosine_anneal(start, end, steps):
    return end + (start - end)/2 * (1 + torch.cos(torch.pi*torch.arange(steps)/(steps-1)))

def resize(img, img_size=128):
    if img.ndim == 3: img = img[None]
    return nn.functional.interpolate(img, size=(img_size, img_size), mode='nearest')

import braceexpand
def get_dataloaders(
    batch_size,
    image_var='images',
    num_devices=None,
    num_workers=None,
    train_url=None,
    val_url=None,
    meta_url=None,
    num_train=None,
    num_val=None,
    cache_dir="/scratch/tmp/wds-cache",
    seed=0,
    voxels_key="nsdgeneral.npy",
    val_batch_size=None,
    to_tuple=["voxels", "images", "trial"],
    local_rank=0,
    world_size=1,
):
    print("Getting dataloaders...")
    assert image_var == 'images'
    
    def my_split_by_node(urls):
        return urls
    
    train_url = list(braceexpand.braceexpand(train_url))
    val_url = list(braceexpand.braceexpand(val_url))

    if num_devices is None:
        num_devices = torch.cuda.device_count()
    
    if num_workers is None:
        num_workers = num_devices
    
    if num_train is None:
        metadata = json.load(open(meta_url))
        num_train = metadata['totals']['train']
    if num_val is None:
        metadata = json.load(open(meta_url))
        num_val = metadata['totals']['val']

    if val_batch_size is None:
        val_batch_size = batch_size
        
    global_batch_size = batch_size * num_devices
    num_batches = math.floor(num_train / global_batch_size)
    num_worker_batches = math.floor(num_batches / num_workers)
    if num_worker_batches == 0: num_worker_batches = 1
    
    print("\nnum_train",num_train)
    print("global_batch_size",global_batch_size)
    print("batch_size",batch_size)
    print("num_workers",num_workers)
    print("num_batches",num_batches)
    print("num_worker_batches", num_worker_batches)
    
    # train_url = train_url[local_rank:world_size]
    train_data = wds.WebDataset(train_url, resampled=False, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .shuffle(500, initial=500, rng=random.Random(42))\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)#\
        # .batched(batch_size, partial=True)#\
        # .with_epoch(num_worker_batches)
    
    # BATCH SIZE SHOULD BE NONE!!! FOR TRAIN AND VAL | resampled=True for train | .batched(val_batch_size, partial=False)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=False)

    # Validation 
    print("val_batch_size",val_batch_size)
    val_data = wds.WebDataset(val_url, resampled=False, cache_dir=cache_dir, nodesplitter=my_split_by_node)\
        .shuffle(500, initial=500, rng=random.Random(42))\
        .decode("torch")\
        .rename(images="jpg;png", voxels=voxels_key, trial="trial.npy", coco="coco73k.npy", reps="num_uniques.npy")\
        .to_tuple(*to_tuple)#\
        # .batched(val_batch_size, partial=True)
    val_dl = torch.utils.data.DataLoader(val_data, batch_size=val_batch_size, num_workers=1, shuffle=False, drop_last=True)

    return train_dl, val_dl, num_train, num_val

pixcorr_preprocess = transforms.Compose([
    transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
])
def pixcorr(images,brains,nan=True):
    all_images_flattened = pixcorr_preprocess(images).reshape(len(images), -1)
    all_brain_recons_flattened = pixcorr_preprocess(brains).view(len(brains), -1)
    if nan:
        corrmean = torch.nanmean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    else:
        corrmean = torch.mean(torch.diag(batchwise_pearson_correlation(all_images_flattened, all_brain_recons_flattened)))
    return corrmean

def select_annotations(annots, random=True):
    """
    There are 5 annotations per image. Select one of them for each image.
    """
    for i, b in enumerate(annots):
        t = ''
        if random:
            # select random non-empty annotation
            while t == '':
                rand = torch.randint(5, (1,1))[0][0]
                t = b[rand]
        else:
            # select first non-empty annotation
            for j in range(5):
                if b[j] != '':
                    t = b[j]
                    break
        if i == 0:
            txt = np.array(t)
        else:
            txt = np.vstack((txt, t))
    txt = txt.flatten()
    return txt

def add_saturation(image, alpha=2):
    gray_image = 0.2989 * image[:, 0, :, :] + 0.5870 * image[:, 1, :, :] + 0.1140 * image[:, 2, :, :]
    gray_image = gray_image.unsqueeze(1).expand_as(image)
    saturated_image = alpha * image + (1 - alpha) * gray_image
    return torch.clamp(saturated_image, 0, 1)

def find_prompt_by_image_number(image_number, data):
    target_image_filename = f"img_t{image_number}.jpg"
    for entry in data:
        if 'target' in entry and entry['target'].endswith(target_image_filename):
            return entry['prompt']
    return -1

def compute_negative_l1_losses(preds, targets):
    batch_size = preds.size(0)
    
    # Expand dimensions for broadcasting
    expanded_preds = preds.unsqueeze(1)        # Shape: [batch_size, 1, 100]
    expanded_targets = targets.unsqueeze(0)    # Shape: [1, batch_size, 100]
    
    # Compute pairwise L1 differences
    l1_diffs = torch.abs(expanded_preds - expanded_targets)  # Shape: [batch_size, batch_size, 100]
    
    # Mask the diagonal to exclude positive pairs
    mask = torch.eye(batch_size).bool().to(l1_diffs.device)
    l1_diffs[mask] = 0
    
    # Sum L1 differences for each sample against all negatives
    negative_losses = l1_diffs.sum(dim=-1).mean()
    
    return negative_losses


def unclip_recon(x, diffusion_engine, vector_suffix,
                 num_samples=1, offset_noise_level=0.04):
    from generative_models.sgm.util import append_dims
    assert x.ndim==3
    if x.shape[0]==1:
        x = x[[0]]
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16), diffusion_engine.ema_scope():
        z = torch.randn(num_samples,4,96,96).to(device) # starting noise, can change to VAE outputs of initial image for img2img

        # clip_img_tokenized = clip_img_embedder(image) 
        # tokens = clip_img_tokenized
        token_shape = x.shape
        tokens = x
        c = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        tokens = torch.randn_like(x)
        uc = {"crossattn": tokens.repeat(num_samples,1,1), "vector": vector_suffix.repeat(num_samples,1)}

        for k in c:
            c[k], uc[k] = map(lambda y: y[k][:num_samples].to(device), (c, uc))

        noise = torch.randn_like(z)
        sigmas = diffusion_engine.sampler.discretization(diffusion_engine.sampler.num_steps)
        sigma = sigmas[0].to(z.device)

        if offset_noise_level > 0.0:
            noise = noise + offset_noise_level * append_dims(
                torch.randn(z.shape[0], device=z.device), z.ndim
            )
        noised_z = z + noise * append_dims(sigma, z.ndim)
        noised_z = noised_z / torch.sqrt(
            1.0 + sigmas[0] ** 2.0
        )  # Note: hardcoded to DDPM-like scaling. need to generalize later.

        def denoiser(x, sigma, c):
            return diffusion_engine.denoiser(diffusion_engine.model, x, sigma, c)

        samples_z = diffusion_engine.sampler(denoiser, noised_z, cond=c, uc=uc)
        samples_x = diffusion_engine.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x*.8+.2), min=0.0, max=1.0)
        # samples = torch.clamp((samples_x + .5) / 2.0, min=0.0, max=1.0)
        return samples
    
def soft_cont_loss(student_preds, teacher_preds, teacher_aug_preds, temp=0.125):
    teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T)/temp
    teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T)/temp
    student_teacher_aug = (student_preds @ teacher_aug_preds.T)/temp
    student_teacher_aug_t = (teacher_aug_preds @ student_preds.T)/temp

    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

def iterate_range(start, length, batchsize):
    batch_count = int(length // batchsize )
    residual = int(length % batchsize)
    for i in range(batch_count):
        yield range(start+i*batchsize, start+(i+1)*batchsize),batchsize
    if(residual>0):
        yield range(start+batch_count*batchsize,start+length),residual 
        
        
# Torch fwRF
def get_value(_x):
    return np.copy(_x.data.cpu().numpy())


#subject: nsd subject index between 1-8
#mode: vision, imagery
#stimtype: all, simple, complex, concepts
#average: whether to average across trials, will produce x that is (stimuli, 1, voxels)
#nest: whether to nest the data according to stimuli, will produce x that is (stimuli, trials, voxels)
import pickle
def condition_average(x, y, cond, nest=False):
    idx, idx_count = np.unique(cond, return_counts=True)
    idx_list = [np.array(cond)==i for i in np.sort(idx)]
    if nest:
        avg_x = torch.zeros((len(idx), idx_count.max(), x.shape[1]), dtype=torch.float32)
    else:
        avg_x = torch.zeros((len(idx), 1, x.shape[1]), dtype=torch.float32)
    for i, m in enumerate(idx_list):
        if nest:
            avg_x[i] = x[m]
        else:
            avg_x[i] = torch.mean(x[m], axis=0)
        
    return avg_x, y, len(idx_count)
def load_nsd_mental_imagery(subject, mode, stimtype="all", average=False, nest=False):
    # This file has a bunch of information about the stimuli and cue associations that will make loading it easier
    img_stim_file = "imagery/nsd_imagery/data/nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3"
    ex_file = open(img_stim_file, 'rb')
    imagery_dict = pickle.load(ex_file)
    ex_file.close()
    # Indicates what experiments trials belong to
    exps = imagery_dict['exps']
    # Indicates the cues for different stimuli
    cues = imagery_dict['cues']
    # Maps the cues to the stimulus image information
    image_map  = imagery_dict['image_map']
    # Organize the indices of the trials according to the modality and the type of stimuli
    cond_idx = {
    'visionsimple': np.arange(len(exps))[exps=='visA'],
    'visioncomplex': np.arange(len(exps))[exps=='visB'],
    'visionconcepts': np.arange(len(exps))[exps=='visC'],
    'visionall': np.arange(len(exps))[np.logical_or(np.logical_or(exps=='visA', exps=='visB'), exps=='visC')],
    'imagerysimple': np.arange(len(exps))[np.logical_or(exps=='imgA_1', exps=='imgA_2')],
    'imagerycomplex': np.arange(len(exps))[np.logical_or(exps=='imgB_1', exps=='imgB_2')],
    'imageryconcepts': np.arange(len(exps))[np.logical_or(exps=='imgC_1', exps=='imgC_2')],
    'imageryall': np.arange(len(exps))[np.logical_or(
                                        np.logical_or(
                                            np.logical_or(exps=='imgA_1', exps=='imgA_2'), 
                                            np.logical_or(exps=='imgB_1', exps=='imgB_2')), 
                                        np.logical_or(exps=='imgC_1', exps=='imgC_2'))]}
    # Load normalized betas
    x = torch.load("imagery/nsd_imagery/data/preprocessed_data/subject{}/nsd_imagery.pt".format(subject)).requires_grad_(False).to("cpu")
    # Find the trial indices conditioned on the type of trials we want to load
    cond_im_idx = {n: [image_map[c] for c in cues[idx]] for n,idx in cond_idx.items()}
    conditionals = cond_im_idx[mode+stimtype]
    # Stimuli file is of shape (18,3,425,425), these can be converted back into PIL images using transforms.ToPILImage()
    y = torch.load("imagery/nsd_imagery/data/nsddata_stimuli/stimuli/imagery_stimuli_18.pt").requires_grad_(False).to("cpu")
    # Prune the beta file down to specific experimental mode/stimuli type
    x = x[cond_idx[mode+stimtype]]
    # If stimtype is not all, then prune the image data down to the specific stimuli type
    if stimtype == "simple":
        y = y[:6]
    elif stimtype == "complex":
        y = y[6:12]
    elif stimtype == "concepts":
        y = y[12:]
    
    # Average or nest the betas across trials
    if average or nest:
        x, y, sample_count = condition_average(x, y, conditionals, nest=nest)
    else:
        x = x.reshape((x.shape[0], 1, x.shape[1]))
    
    # print(x.shape)
    return x, y
    
def bb_soft_clip_loss(preds, targs, temp=0.125):
    temp = np.exp(temp)
    clip_clip = (targs @ targs.T)/temp
    brain_brain = (preds @ preds.T)/temp
    
#     loss1 = -(brain_brain.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
#     loss2 = -(brain_brain.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
#     loss = (loss1 + loss2)/2
    
    loss = nn.functional.kl_div(brain_brain.log_softmax(-1), clip_clip.softmax(-1), reduction='batchmean')
    return loss #* 1e5

def bb_cossim_loss(preds, targs, temp=None):
    clip_clip = (targs @ targs.T)
    brain_brain = (preds @ preds.T)
    loss = 1 - nn.functional.cosine_similarity(brain_brain, clip_clip).mean()
    return loss 

def load_images_to_numpy(folder_path):
    file_names = [f for f in os.listdir(folder_path) if (f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'))]
    image_data = []
    image_names = []
    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)
        image_names.append(file_name)
        with Image.open(image_path) as img:
            img_array = np.array(img)
            if img_array.shape[1] != 224:
                img = img.resize((224,224))
                img_array = np.array(img)
            image_data.append(img_array)
    images_np = np.stack(image_data, axis=0)
    return images_np, image_names


import hashlib
def hash_image(image_tensor):
    # Convert tensor to bytes
    image_bytes = image_tensor.detach().cpu().numpy().tobytes()
    # Hash the bytes using SHA-256
    hash_object = hashlib.sha256(image_bytes)
    hex_dig = hash_object.hexdigest()
    return hex_dig


def find_paired_indices(x):
    unique_elements, counts = torch.unique(x, return_counts=True)
    repeated_elements = unique_elements[counts > 1]
    paired_indices = []
    
    for element in repeated_elements:
        indices = (x == element).nonzero(as_tuple=True)[0]
        # Instead of creating pairs, just collect the entire set of indices once
        paired_indices.append(indices[:len(indices)].tolist())
    
    return paired_indices


def zscore(data,train_mean=None,train_std=None):
    # assuming that first dim is num_samples and second dim is num_voxels
    if train_mean is None:
        train_mean = np.mean(data,axis=0)
    if train_std is None:
        train_std = np.std(data,axis=0)
    zscored_data = (data - train_mean) / (train_std + 1e-6)
    return zscored_data


def log_io(func):  # the first argument must be input; output must be a kwarg for this to work properly
    def wrapper(*args, **kwargs):
        inp = args[0]
        output = kwargs['output']
        print(f'\n*** Loading data from {inp} ***\n')
        result = func(*args, **kwargs)
        print(f'\n*** Saved resampled data to {output} ***\n')
        return result
    return wrapper

@log_io  
def resample(inp, ref, target_size, omat, output=None):  
    os.system(f"flirt -in {inp} \
                    -ref {ref} \
                    -applyisoxfm {target_size} -nosearch \
                    -omat {omat} \
                    -out {output}")

@log_io
def applyxfm(inp, ref, init, interp, output=None):
    os.system(f"flirt -in {inp} \
                -ref {ref} \
                -out {output} \
                -applyxfm -init {init} \
                -interp {interp}")

@log_io
def apply_thresh(inp, thresh, output=None):
    os.system(f"fslmaths {inp} -thr {thresh} -bin {output}")

def resample_betas(orig_glmsingle_path, sub, session, task_name, vox, glmsingle_path, glm_save_path_resampled, ref_name, omat):
    # convert vox to nifti object and save
    orig_mask = nib.load(f"{orig_glmsingle_path}/{sub}_{session}{task_name}_brain.nii.gz")

    # apply mask and save original betas
    print("original:", vox.shape)
    vox_nii = unmask(vox, orig_mask)
    glm_save_path = f"{glmsingle_path}/vox.nii.gz"
    nib.save(vox_nii, glm_save_path)
    print(f"saved original glmsingle betas to {glm_save_path}")

    # resample and save betas
    applyxfm(glm_save_path, ref_name, omat, resample_method, output=glm_save_path_resampled)
    vox = nib.load(glm_save_path_resampled)
    print("vox after resampling", vox.shape)

    return vox


def load_preprocess_betas(glmsingle_path, session, ses_list,
                              remove_close_to_MST, image_names, 
                              remove_random_n, vox_idx):
    glmsingle = np.load(f"{glmsingle_path}/TYPED_FITHRF_GLMDENOISE_RR.npz", allow_pickle=True)
    vox = glmsingle['betasmd'].T

    print("vox", vox.shape)

    # Preprocess betas
    if vox.ndim == 4:
        vox = vox[:, 0, 0]
        print("vox", vox.shape)

    if remove_close_to_MST:
        x = [x for x in image_names if x != 'blank.jpg' and str(x) != 'nan']
        close_to_MST_idx = [y for y, z in enumerate(x) if 'closest_pairs' in z]
        close_to_MST_mask = np.ones(len(vox), dtype=bool)
        close_to_MST_mask[close_to_MST_idx] = False
        vox = vox[close_to_MST_mask]
        print("vox after removing close_to_MST", vox.shape)

    elif remove_random_n:
        random_n_mask = np.ones(len(vox), dtype=bool)
        random_n_mask[vox_idx] = False
        vox = vox[random_n_mask]
        print(f"vox after removing {n_to_remove}", vox.shape)

    return vox


def prepare_model_and_training(
    num_voxels_list, 
    n_blocks,
    hidden_dim, 
    clip_emb_dim, 
    clip_seq_dim, 
    clip_scale,
    use_prior=False, 
):
    """
    Prepare MindEye model, optimizer, and learning rate scheduler.
    
    Args:
        num_voxels_list (list): List of number of voxels for each subject
        hidden_dim (int): Hidden dimension for model layers
        clip_emb_dim (int): CLIP embedding dimension
        clip_seq_dim (int): CLIP sequence dimension
        use_prior (bool): Whether to include diffusion prior network
    
    Returns:
        model
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from models import VersatileDiffusionPriorNetwork, BrainDiffusionPrior
    from MindEye2 import MindEyeModule, RidgeRegression, BrainNetwork
    import utils

    model = MindEyeModule()
    print(model)

    model.ridge = RidgeRegression(num_voxels_list, out_features=hidden_dim)
    utils.count_params(model.ridge)
    utils.count_params(model)

    model.backbone = BrainNetwork(h=hidden_dim, in_dim=hidden_dim, out_dim=clip_emb_dim*clip_seq_dim, seq_len=1, n_blocks=n_blocks,
                              clip_size=clip_emb_dim)
    utils.count_params(model.backbone)
    utils.count_params(model)

    if use_prior:
        # setup diffusion prior network
        out_dim = clip_emb_dim
        depth = 6
        dim_head = 52
        heads = clip_emb_dim//52 # heads * dim_head = clip_emb_dim
        timesteps = 100
        prior_network = VersatileDiffusionPriorNetwork(
                dim=out_dim,
                depth=depth,
                dim_head=dim_head,
                heads=heads,
                causal=False,
                num_tokens = clip_seq_dim,
                learned_query_mode="pos_emb"
            )
        model.diffusion_prior = BrainDiffusionPrior(
            net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            cond_drop_prob=0.2,
            image_embed_scale=None,
        )

        utils.count_params(model.diffusion_prior)
        utils.count_params(model)

    return model


def get_slurm_seed(default=0):
    """Returns SLURM array seed or a default seed if not running in SLURM."""
    try:
        seed = int(os.environ["SLURM_ARRAY_TASK_ID"])
        print(f"Using SLURM job array seed: {seed}")
    except KeyError:
        print(f"SLURM seed not found, using default: {default}")
        seed = default
    return seed


def get_slurm_job():
    """Returns ID of current SLURM job"""
    return int(os.environ["SLURM_ARRAY_JOB_ID"])


def filter_and_average_mst(vox, vox_image_dict):
    """
    Filters and averages repeated MST images while retaining unique images.
    
    Args:
        vox (np.ndarray): Original array of shape (num_images, num_features).
        vox_image_dict (dict): Maps image indices to file paths.
    Returns:
        tuple: Filtered array and corresponding kept indices.
    """
    from copy import deepcopy
    
    # Identify repeated MST paths
    repeats = {}
    for idx, path in vox_image_dict.items():
        if "MST_pairs" in path:
            repeats.setdefault(path, []).append(idx)
    
    # Create mask to track kept entries
    keep_mask = np.ones(vox.shape[0], dtype=bool)
    output_vox = deepcopy(vox).astype(np.float32)
    
    # Average repeated MST images
    for indices in repeats.values():
        if len(indices) > 1:
            avg_values = np.mean(vox[indices], axis=0)
            output_vox[indices[0]] = avg_values
            keep_mask[indices[1:]] = False
    
    return output_vox[keep_mask], np.where(keep_mask)[0]


def verify_image_patterns(image_to_indices):
    failures = []
    for image_name, sessions in image_to_indices.items():
        session1, session2 = sessions
        total_count = len(session1) + len(session2)

        if "special515" in image_name:
            if not (
                (len(session1) == 3 and len(session2) == 0) or
                (len(session1) == 0 and len(session2) == 3) or
                (len(session1) == 1 and len(session2) == 0) or
                (len(session1) == 0 and len(session2) == 1)
            ):
                failures.append(f"{image_name} does not appear 3x in only 1 session.")
        elif "MST_pairs" in image_name:
            if not (len(session1) == 2 and len(session2) == 2):
                failures.append(f"{image_name} does not appear 2x in both sessions.")
        else:
            if not (
                (total_count == 1) and
                (len(session1) == 1 and len(session2) == 0 or len(session1) == 0 and len(session2) == 1)
            ):
                failures.append(f"{image_name} does not appear 1x in only 1 session.")

    return failures

def compute_avg_repeat_corrs(vox_repeats: np.ndarray) -> np.ndarray:
    """
    Given an array of shape (n_repeats, n_voxels), compute the average correlation
    across all unique repeat combinations for each voxel.
    Returns:
        rels: (n_voxels,) array of averaged correlations
    """
    import itertools
    n_repeats, n_vox = vox_repeats.shape
    combos = list(itertools.combinations(range(n_repeats), 2))
    
    rels = np.full(n_vox, np.nan)
    
    # For each voxel
    for v in range(n_vox):
        corrs = []
        # Calculate correlation for each pair of repeats
        for i, j in combos:
            r = np.corrcoef(vox_repeats[i, v], vox_repeats[j, v])[0, 1]
            corrs.append(r)
        # Average across all pairwise correlations
        rels[v] = np.mean(corrs)
    
    return rels


def get_pairs(data, repeat_indices=(0, 1)):
    """
    Extract pairs based on specified repeat indices, falling back to available repeats.
    
    Parameters:
    - data: List of items, where each item may have different number of repeats
    - repeat_indices: Tuple of indices (i, j) to extract if available
    
    Returns:
    - Array of pairs
    """
    result = []
    
    for item in data:
        # Determine what repeats are actually available
        num_repeats = len(item)
        
        # Handle the requested indices
        i, j = repeat_indices
        
        # Adjust indices if they're out of bounds
        if i >= num_repeats:
            i = min(num_repeats - 1, 0)
        if j >= num_repeats:
            j = min(num_repeats - 1, 1 if num_repeats > 1 else 0)
            
        # Create the pair
        result.append([item[i], item[j]])
    
    return np.array(result)


def compute_vox_rels(vox, pairs, sub, session, rdm=False, repeat_indices=(0,1)):
    from tqdm import tqdm
    pairs = get_pairs(pairs, repeat_indices=repeat_indices)
    # print(pairs)
    # _tmp = [(i[0],i[-1]) for i in pairs]
    # breakpoint()
    # vox_pairs = zscore(vox[_tmp])  # zscoring based on first and last repeat only
    # rels = compute_avg_repeat_corrs(vox_pairs)

    # _tmp = [(i[0],i[1]) for i in pairs]
    # vox_pairs = zscore(vox[_tmp])
    
    vox_pairs = zscore(vox[pairs])
    rels = np.full(vox.shape[-1], np.nan)
    for v in tqdm(range(vox.shape[-1])):
        rels[v] = np.corrcoef(vox_pairs[:, 0, v], vox_pairs[:, 1, v])[1, 0]
    
    print("rels", rels.shape)
    assert np.sum(np.all(np.isnan(rels))) == 0
    
    if rdm:  # generate a Representational Dissimilarity Matrix to visualize how similar the voxel patterns are across images
        # average voxel patterns across repeats
        vox0 = np.zeros((len(pairs), vox.shape[-1], 2))
        for ipair, pair in enumerate(tqdm(pairs)):
            i, j = pair[:2]  # Using the first two repeats
            vox0[ipair, :, :] = vox[pair].T
        vox_avg = vox0.mean(-1)

        # plot the RDM at various thresholds
        r_thresholds = np.array([.0, .1, .2, .3])
        rdm = np.zeros((len(r_thresholds), len(pairs), len(pairs))) 

        for ir_thresh, r_thresh in enumerate(r_thresholds):
            print(f"reliability threshold = {r_thresh}")
            for i in tqdm(range(len(pairs))):
                for j in range(len(pairs)):
                    rdm[ir_thresh, i, j] = np.corrcoef(vox_avg[i, rels > r_thresh], 
                                                       vox_avg[j, rels > r_thresh])[0, 1]
        n_thresh = len(r_thresholds)
        fig, axs = plt.subplots(1, n_thresh, figsize=(4 * n_thresh, 4), squeeze=False)

        for i, r_thresh in enumerate(r_thresholds):
            ax = axs[0, i]
            im = ax.imshow(rdm[i], clim=(-1, 1))
            ax.set_title(f"r > {r_thresh:.1f}")
            ax.set_xlabel("Image")
            ax.set_ylabel("Image")
            fig.colorbar(im, ax=ax, shrink=0.8)

        # Optional: add a supertitle with subject/session/repeat info
        fig.suptitle(f"{sub}_{session}\nrepeat combo {r}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Leave space for suptitle
        plt.show()

            # thresh = .2
            # plt.figure(figsize=(4, 4))
            # plt.imshow(rdm[np.where(r_thresholds == thresh)[0].item()], clim=(-1, 1))
            # plt.colorbar(shrink=0.8)
            # plt.title(f"{sub}_{session}\nreliability threshold={thresh}; repeats {r}")
            # plt.show()

        for thresh in range(rdm.shape[0]):
            for img in range(rdm.shape[1]):
                assert np.isclose(rdm[thresh, img, img], 1)
    
    return rels


def load_masks(img_list):
    from nilearn.masking import intersect_masks
    import nilearn

    masks = [nilearn.image.load_img(mask) for mask in img_list]
    assert all(np.allclose(masks[0].affine, m.affine) for m in masks)
    return masks, intersect_masks(masks, threshold=0.5, connected=True)


def get_mask(ses_list, sub, func_task_name):
    assert isinstance(ses_list, list), "ses_list is not a list"
    mask_imgs = []
    nsd_imgs = []
    for ses in ses_list:
        prefix = f"/scratch/gpfs/ri4541/MindEyeV2/src/mindeyev2/glmsingle_{sub}_{ses}_task-{func_task_name}/{sub}_{ses}_task-{func_task_name}"
        mask_path = prefix + "_brain.nii.gz"
        nsd_path = prefix + "_nsdgeneral.nii.gz"
        print(mask_path)
        print(nsd_path)
        assert os.path.exists(mask_path)
        assert os.path.exists(nsd_path)
        mask_imgs.append(mask_path)
        nsd_imgs.append(nsd_path)

    func_masks, avg_mask = load_masks(mask_imgs)
    print(f'intersected brain masks from {ses_list}')
    
    nsd_masks, roi = load_masks(nsd_imgs)
    print(f'intersected nsdgeneral roi masks from {ses_list}')

    return func_masks, avg_mask, nsd_masks, roi



def process_images(image_names, unique_images, remove_close_to_MST=False, remove_random_n=False, imgs_to_remove=None, sub=None, session=None):
    image_idx = np.array([])
    vox_image_names = np.array([])
    all_MST_images = {}
    
    for i, im in enumerate(image_names):
        if im == "blank.jpg" or str(im) == "nan":
            continue
                
        if remove_close_to_MST and "closest_pairs" in im:
            continue
        
        if remove_random_n and im in imgs_to_remove:
            continue
            
        vox_image_names = np.append(vox_image_names, im)
        image_idx_ = np.where(im == unique_images)[0].item()
        image_idx = np.append(image_idx, image_idx_)
        
        if sub == 'ses-01' and session in ('ses-01', 'ses-04'):
            if ('w_' in im or 'paired_image_' in im or re.match(r'all_stimuli/rtmindeye_stimuli/\d{1,2}_\d{1,3}\.png$', im) 
                or re.match(r'images/\d{1,2}_\d{1,3}\.png$', im)):
                all_MST_images[i] = im
        elif 'MST' in im:
            all_MST_images[i] = im
    
    image_idx = torch.Tensor(image_idx).long()
    unique_MST_images = np.unique(list(all_MST_images.values()))
    
    MST_ID = np.array([], dtype=int)
    if remove_close_to_MST:
        close_to_MST_idx = np.array([], dtype=int)
    if remove_random_n:
        random_n_idx = np.array([], dtype=int)
    
    vox_idx = np.array([], dtype=int)
    j = 0  # Counter for indexing vox based on removed images
    
    for i, im in enumerate(image_names):
        if im == "blank.jpg" or str(im) == "nan":
            continue
        
        if remove_close_to_MST and "closest_pairs" in im:
            close_to_MST_idx = np.append(close_to_MST_idx, i)
            continue
        
        if remove_random_n and im in imgs_to_remove:
            vox_idx = np.append(vox_idx, j)
            j += 1
            continue
        
        j += 1
        curr = np.where(im == unique_MST_images)
        
        if curr[0].size == 0:
            MST_ID = np.append(MST_ID, len(unique_MST_images))  # Out of range index for filtering later
        else:
            MST_ID = np.append(MST_ID, curr)
    
    assert len(MST_ID) == len(image_idx)
    
    pairs = find_paired_indices(image_idx)
    pairs = sorted(pairs, key=lambda x: x[0])
    
    return image_idx, vox_image_names, pairs
