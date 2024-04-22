import numpy as np
import pandas as pd
import pdb
import os
import pickle

run_to_blanks = {"0": [9, 19, 29, 39, 49, 59, 68, 69, 70, 71], 
"1": [9, 19, 31, 42, 56, 68, 69, 70, 71], 
"2": [9, 19, 29, 39, 49, 59, 68, 69, 70, 71], 
"3": [9, 19, 31, 45, 56, 68, 69, 70, 71], 
"4": [9, 19, 29, 39, 49, 59, 68, 69, 70, 71], 
"5": [9, 20, 30, 44, 56, 68, 69, 70, 71], 
"6": [9, 19, 29, 39, 49, 59, 68, 69, 70, 71], 
"7": [13, 24, 36, 46, 56, 68, 69, 70, 71], 
"8": [9, 19, 29, 39, 49, 59, 68, 69, 70, 71], 
"9": [9, 19, 33, 44, 56, 68, 69, 70, 71], 
"10": [9, 19, 29, 39, 49, 59, 68, 69, 70, 71], 
"11": [9, 19, 33, 44, 56, 68, 69, 70, 71], 
"12": [9, 19, 29, 39, 49, 59, 68, 69, 70, 71], 
"13": [9, 23, 33, 44, 56, 68, 69, 70, 71], 
"14": [9, 19, 29, 39, 49, 59, 68, 69, 70, 71], 
"15": [9, 21, 31, 45, 56, 68, 69, 70, 71]}

# get the unique stims sorted here for each participant's design matrix
images_list = sorted(os.listdir("../images/"))
run_length = 288
num_runs = 16
tr_length = 1
num_trs = 288
num_unique_stim_trials_per_session = 850
assert(len(images_list) == num_unique_stim_trials_per_session)
participant_id = 1
conditions_file = pd.read_csv(f"../conditions_files/participant{participant_id}_.csv")
data_file = pd.read_csv("../data/1_replicate_nsd_2024-04-16_10h54.13.489.csv")
design = []

# add in columns for each unique stim
for run_num in range(num_runs):
    this_run_df = data_file[data_file["run_num"] == run_num]
    this_run_conditions_file = conditions_file[conditions_file["run_num"] == run_num]
    this_run_design = pd.DataFrame(np.zeros(shape = (num_trs,
                                   num_unique_stim_trials_per_session)),  
                                   columns = images_list)
    run_time0 = this_run_df.iloc[0,:]["trial.started"]
    num_stims = 0
    for index, row in this_run_df.iterrows():
        if not row["is_blank_trial"]:
            image_start_time = round(row["image.started"] - run_time0)
            onset_tr_num = image_start_time
            if num_stims < 2:
                print(onset_tr_num) 
            image_id = row["current_image"].split("/")[1]
            this_run_design.loc[onset_tr_num,image_id] = 1
            num_stims += 1
        else:
            pass
    design.append(this_run_design)
file = "design_matrices/design_participant" + str(participant_id)
file_pi = open(file, 'wb') 
pickle.dump(design, file_pi)
file_pi.close()
