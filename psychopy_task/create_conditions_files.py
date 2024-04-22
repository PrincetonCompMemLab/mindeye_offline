import random
import copy
import pandas as pd
import pdb
import os
import numpy as np

n_participants = 1000
images_paths = os.listdir("images/") # 750 + 100 pairs
assert(len(images_paths) == 850)
import pickle
# open a file, where you stored the pickled data
file = open('images150', 'rb')
# dump information to that file
repeats150 = pickle.load(file)
print(repeats150)
# close the file
file.close()
# add the 150 to the 750
pdb.set_trace()
images_paths += repeats150
print(images_paths)
print(len(images_paths))
assert(len(images_paths) == 1000)
num_runs = 16
trials_per_run = 72
# set blank trials as fixed across participants but varied across trials
import json
f = open('blank_trials.json')
run_to_blanks = json.load(f)
f.close()
print(run_to_blanks)

def no_adjacent_duplicates(images_list):
    for previous, current, next_image in zip([""] + images_list[:-1], 
                                       images_list, 
                                       images_list[1:] + [""]):
        if previous == current or current == next_image:
            print("ADJACENT DUP: ", images_list)
            return False
    print("CLEAR: ",images_list)
    return True

all_images_lists = []

for p_id in range(n_participants):
    random.seed(p_id)
    # make dirs
    participant_path = "conditions_files/participant" + str(p_id) + "_"
    random.shuffle(images_paths)
    while not no_adjacent_duplicates(images_paths) or images_paths in all_images_lists:
        random.shuffle(images_paths)
    all_images_lists.append(images_paths)
    current_image_list = []
    is_repeat_list = []
    run_num_list = []
    is_new_run_list = []
    is_blank_trial_list = []
    trial_index_list = []
    image_index = 0
    all_blanks_list_list = []
    for run_num in range(num_runs):
        blank_trial_indices = run_to_blanks[str(run_num)]
        for trial_index in range(trials_per_run):
            run_num_list.append(run_num)
            all_blanks_list_list.append(blank_trial_indices)
            trial_index_list.append(trial_index)
            if trial_index == (trials_per_run - 1) and run_num != (num_runs - 1):
                is_new_run_list.append(1)
            else:
                is_new_run_list.append(0)

            if trial_index in blank_trial_indices:
                current_image_list.append("blank.jpg")
                is_blank_trial_list.append(1)
                is_repeat_list.append(0)
            else:
                # print("len(images_paths): ",len(images_paths))
                # print("image_index: ",image_index)
                image_path = images_paths[image_index]
                if "images/" + image_path in current_image_list:
                    is_repeat_list.append(1)
                else:
                    is_repeat_list.append(0)
                current_image_list.append("images/" + image_path)
                is_blank_trial_list.append(0)
                image_index += 1
    # output study and test
    output_dict = {"current_image": current_image_list,
                   "is_repeat": is_repeat_list,
                   "trial_index": trial_index_list,
                   "is_blank_trial":is_blank_trial_list,
                   "is_new_run": is_new_run_list,
                   "run_num": run_num_list,
                   "all_blanks_list": all_blanks_list_list}
    output_df = pd.DataFrame(output_dict)
    study_test_file_path = participant_path + ".csv"
    output_df.to_csv(study_test_file_path, index = False)
