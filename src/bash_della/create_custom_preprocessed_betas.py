from nilearn.image import get_data, index_img, concat_imgs, new_img_like
import nibabel as nib
import numpy as np
import os
import nibabel as nib
from subprocess import call
from datetime import datetime, date
from nilearn.glm.first_level import *
from nilearn import image, plotting
import nibabel as nib
import pandas as pd
from subprocess import call
from pathlib import Path
from datetime import datetime, date
from nilearn.image import get_data, index_img, concat_imgs, new_img_like
from nilearn.signal import clean
range_runs = (1,13)


# DAY 1 #

# load in the bolds imgs time series and events tsv from nsd rawdata
ndscore_bold_imgs = [nib.load(f'/scratch/gpfs/rk1593/rt_mindEye/raw/subj01/raw/0/sub-01_ses-nsd01_task-nsdcore_run-{run:02d}_bold.nii.gz') for run in range(range_runs[0], range_runs[1])]
ndscore_events = [pd.read_csv(f'/scratch/gpfs/rk1593/rt_mindEye/raw/subj01/raw/0/sub-01_ses-nsd01_task-nsdcore_run-{run:02d}_events.tsv', sep = "\t", header = 0) for run in range(range_runs[0], range_runs[1])]
# create a new list of events_df's which will have the trial_type modified to be unique identifiers
new_ndscore_events = []
mask_img = nib.load('/scratch/gpfs/rk1593/rt_mindEye/masking_operation/thresholded_masks/day1_subj1/mask_transformed_0.4444440000_day1_subj1.nii.gz')
mask_on = True
betas_custom = np.zeros(shape=(30000, len(mask_img.get_fdata().flatten().tolist())))

if not mask_on:f
    mask_img = None
lsa_glm = FirstLevelModel(t_r=1.6,slice_time_ref=.5,hrf_model='glover',
                        drift_model=None,high_pass=None,mask_img=mask_img,
                        signal_scaling=False,smoothing_fwhm=None,noise_model='ar1',
                        n_jobs=-1,verbose=-1,memory_level=1,minimize_memory=True)
from datetime import date

today = date.today()

# Month abbreviation, day and year	
d4 = today.strftime("%b-%d-%Y")
print("d4 =", d4)

def motion_correct(in_=None, reffile=None, out=None):
    # Motion correct to this run's functional reference
    # takes short 10 volume bold run and temporal avg of that as single 3d reference volume
    # create pseudo
    
    command = f"mcflirt -in {in_} -reffile {reffile} -plots -out {out}"
    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
    pad_print(f"Motion correction time: {B-A:.4f}")
!module load fsl/6.0.6.2 

# this for loop is going to give us beta maps with their trial names
# we have a beta map for each stimulus trial given in all 12 runs of day 1 here
beta_maps_list = []
all_trial_names_list = []
starting_run = 11
if starting_run > 0:
    if mask_on:
        beta_maps_list = np.load("/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day1/masked_betas_subj1_day1_up_to_run" + \
                                 str(starting_run - 1) + "_" + d4 + ".npy").tolist()
        all_trial_names_list = np.load("/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day1/masked_trial_ids_up_to_run" + \
                                       str(starting_run - 1) + "_" + d4 + ".npy").tolist()

    else:
        beta_maps_list = np.load("/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day1/betas_subj1_day1_up_to_run" + \
                                 str(starting_run - 1) + "_" + d4 + ".npy").tolist()
        all_trial_names_list = np.load("/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day1/trial_ids_up_to_run" + \
                                       str(starting_run - 1) + "_" + d4 + ".npy").tolist()


for run_num,events_df in enumerate(ndscore_events):
    print(run_num)
    if run_num < starting_run:
        continue
    new_events_df = events_df
    for i_trial, trial in new_events_df.iterrows():
        new_events_df.loc[i_trial, "trial_type"] = "COCOimage_" + str(trial["73k_id"]) + "_trialNum_" + str(trial["trial_number"]) + "_runNum" + str(run_num)
    mc_params = []
    imgs = []
    raw = ndscore_bold_imgs[run_num]
    ntrs = raw.shape[-1]
    for tr in range(ntrs):
        print(tr)
        tmp = '/scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/day1_subj1/tmp.nii'
        nib.save(index_img(raw,tr),tmp)

        # motion correct and load in the parameters
        mc = '/scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/day1_subj1/tmp_mc'
        os.system(f"mcflirt -in {tmp} -reffile /scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates//day1_subj1/refvol.nii -plots -out {mc}")
        mc_params.append(np.loadtxt(f'{mc}.par'))

        #spatial smoothing
        sm = '/scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/day1_subj1/tmp_mc_sm'
        os.system(f'fslmaths {mc} -kernel gauss {5/2.3548} -fmean {sm}')
        imgs.append(get_data(sm + ".nii.gz"))
    
    img = np.rollaxis(np.array(imgs),0,4)
    img = new_img_like(raw,img,copy_header=True)
#     out = f'/scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/sub-01_ses-nsd01_task-nsdcore_run-{run_num:02d}_bold_mc_sm.nii.gz'
#     nib.save(img,out)
    mc_params = np.array(mc_params)
    print(mc_params.shape)
    lsa_glm.fit(run_imgs=img,events=new_events_df, confounds = pd.DataFrame(mc_params))
    fig, ax = plt.subplots(figsize=(10, 10))
    plotting.plot_design_matrix(lsa_glm.design_matrices_[0], ax=ax)
    fig.show()
    trialwise_conditions = new_events_df["trial_type"].unique()
    for condition in trialwise_conditions:
        beta_map = lsa_glm.compute_contrast(str(condition), output_type="effect_size")
        beta_maps_list.append(beta_map.get_fdata().flatten().tolist())
        all_trial_names_list.append(condition)
    pdb.set_trace()
    print("saving...")
    if mask_on:
        np.save(file = "/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day1/masked_betas_subj1_day1_up_to_run" + str(run_num) + "_" + d4, arr = np.vstack(beta_maps_list))
        np.save(file = "/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day1/masked_trial_ids_up_to_run" + str(run_num) + "_" + d4, arr = np.array(all_trial_names_list))

    else:
        np.save(file = "/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day1/betas_subj1_day1_up_to_run" + str(run_num) + "_" + d4, arr = np.vstack(beta_maps_list))
        np.save(file = "/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day1/trial_ids_up_to_run" + str(run_num) + "_" + d4, arr = np.array(all_trial_names_list))

# DAY 2 #
import os
import nibabel as nib
from subprocess import call
from datetime import datetime, date
from nilearn.glm.first_level import *
from nilearn import image, plotting
import nibabel as nib

from subprocess import call
from pathlib import Path
from datetime import datetime, date
from nilearn.image import get_data, index_img, concat_imgs, new_img_like
from nilearn.signal import clean
range_runs = (1,13)
# load in the bolds imgs time series and events tsv from nsd rawdata
ndscore_bold_imgs = [nib.load(f'/scratch/gpfs/rk1593/rt_mindEye/raw/subj01/raw/1/sub-01_ses-nsd02_task-nsdcore_run-{run:02d}_bold.nii.gz') for run in range(range_runs[0], range_runs[1])]
ndscore_events = [pd.read_csv(f'/scratch/gpfs/rk1593/rt_mindEye/raw/subj01/raw/1/sub-01_ses-nsd02_task-nsdcore_run-{run:02d}_events.tsv', sep = "\t", header = 0) for run in range(range_runs[0], range_runs[1])]
# create a new list of events_df's which will have the trial_type modified to be unique identifiers
new_ndscore_events = []
mask_img = nib.load("/scratch/gpfs/rk1593/rt_mindEye/masking_operation/" + \
                    "thresholded_masks/day2_subj1/mask_transformed_0.4444440000_day2_subj1.nii.gz")
mask_on = True
betas_custom = np.zeros(shape=(30000, np.unique(mask_img.get_fdata(), return_counts = True)[1][1]))
import pdb
pdb.set_trace()
if not mask_on:
    mask_img = None
lsa_glm = FirstLevelModel(t_r=1.6,slice_time_ref=.5,hrf_model='glover',
                        drift_model=None,high_pass=None,mask_img=mask_img,
                        signal_scaling=False,smoothing_fwhm=None,noise_model='ar1',
                        n_jobs=-1,verbose=-1,memory_level=1,minimize_memory=True)
from datetime import date

today = date.today()

# Month abbreviation, day and year	
d4 = today.strftime("%b-%d-%Y")
print("d4 =", d4)

def motion_correct(in_=None, reffile=None, out=None):
    # Motion correct to this run's functional reference
    # takes short 10 volume bold run and temporal avg of that as single 3d reference volume
    # create pseudo
    
    command = f"mcflirt -in {in_} -reffile {reffile} -plots -out {out}"
    A = datetime.now().timestamp(); call(command,shell=True); B = datetime.now().timestamp()
    pad_print(f"Motion correction time: {B-A:.4f}")
!module load fsl/6.0.6.2 

# this for loop is going to give us beta maps with their trial names
# we have a beta map for each stimulus trial given in all 12 runs of day 1 here
beta_maps_list = []
all_trial_names_list = []
starting_run = 0
if starting_run > 0:
    if mask_on:
        beta_maps_list = np.load("/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day2/masked_betas_subj1_day2_up_to_run" + \
                                 str(starting_run - 1) + "_" + d4 + ".npy").tolist()
        all_trial_names_list = np.load("/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day2/masked_trial_ids_day2_up_to_run" + \
                                       str(starting_run - 1) + "_" + d4 + ".npy").tolist()

    else:
        beta_maps_list = np.load("/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day2/betas_subj1_day2_up_to_run" + \
                                 str(starting_run - 1) + "_" + d4 + ".npy").tolist()
        all_trial_names_list = np.load("/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day2/trial_ids_day2_up_to_run" + \
                                       str(starting_run - 1) + "_" + d4 + ".npy").tolist()


for run_num,events_df in enumerate(ndscore_events):
    print(run_num)
    if run_num < starting_run:
        continue
    new_events_df = events_df
    for i_trial, trial in new_events_df.iterrows():
        new_events_df.loc[i_trial, "trial_type"] = "COCOimage_" + str(trial["73k_id"]) + "_trialNum_" + str(trial["trial_number"]) + "_runNum" + str(run_num)
    mc_params = []
    imgs = []
    raw = ndscore_bold_imgs[run_num]
    ntrs = raw.shape[-1]
    for tr in range(ntrs):
        print(tr)
        tmp = '/scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/day2_subj1/tmp.nii'
        nib.save(index_img(raw,tr),tmp)

        # motion correct and load in the parameters
        mc = '/scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/day2_subj1/tmp_mc'
        os.system(f"mcflirt -in {tmp} -reffile /scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/day2_subj1/refvol.nii -plots -out {mc}")
        mc_params.append(np.loadtxt(f'{mc}.par'))

        #spatial smoothing
        sm = '/scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/day2_subj1/tmp_mc_sm'
        os.system(f'fslmaths {mc} -kernel gauss {5/2.3548} -fmean {sm}')
        imgs.append(get_data(sm + ".nii.gz"))
    
    img = np.rollaxis(np.array(imgs),0,4)
    img = new_img_like(raw,img,copy_header=True)
#     out = f'/scratch/gpfs/rk1593/rt_mindEye/preprocessing_intermediates/sub-01_ses-nsd01_task-nsdcore_run-{run_num:02d}_bold_mc_sm.nii.gz'
#     nib.save(img,out)
    mc_params = np.array(mc_params)
    print(mc_params.shape)
    lsa_glm.fit(run_imgs=img,events=new_events_df, confounds = pd.DataFrame(mc_params))
    fig, ax = plt.subplots(figsize=(10, 10))
    plotting.plot_design_matrix(lsa_glm.design_matrices_[0], ax=ax)
    fig.show()
    trialwise_conditions = new_events_df["trial_type"].unique()
    for condition in trialwise_conditions:
        beta_map = lsa_glm.compute_contrast(str(condition), output_type="effect_size")
        beta_maps_list.append(beta_map.get_fdata().flatten().tolist())
        all_trial_names_list.append(condition)
    print("saving...")
    if mask_on:
        np.save(file = "/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day2/masked_betas_subj1_day2_up_to_run" + str(run_num) + "_" + d4, arr = np.vstack(beta_maps_list))
        np.save(file = "/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day2/masked_trial_ids_day2_up_to_run" + str(run_num) + "_" + d4, arr = np.array(all_trial_names_list))

    else:
        np.save(file = "/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day2/betas_subj1_day2_up_to_run" + str(run_num) + "_" + d4, arr = np.vstack(beta_maps_list))
        np.save(file = "/scratch/gpfs/rk1593/rt_mindEye/betas/subj1/day2/trial_ids_day2_up_to_run" + str(run_num) + "_" + d4, arr = np.array(all_trial_names_list))
