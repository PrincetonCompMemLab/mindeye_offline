import random
import copy
import pandas as pd
import pdb
import os
import numpy as np

n_participants = 1000
images_paths = os.listdir("images/")[:750]
random.shuffle(images_paths)
images_paths_500 = images_paths[:500]
images_paths = (images_paths_500 + images_paths_500)
num_runs = 16
trials_per_run = 72
# set blank trials as fixed across participants but varied across trials
import json
f = open('blank_trials.json')
run_to_blanks = json.load(f)
f.close()
print(run_to_blanks)

# def between9and14continuous(all_indices):
#     all_indices = sorted(all_indices)
#     print("all_indices: ", all_indices)
#     for i, index in enumerate(all_indices):
#         if i == 0:
#             # print("all_indices[1]: ", all_indices[1])
#             # print("index: ", index)
#             if not ((all_indices[1] - index) >= 9 \
#                     and (all_indices[1] - index) <= 14):
#                 return False
#         elif i == len(all_indices) - 1:
#             # print("all_indices[len(all_indices) - 2]: ", \
#             #        all_indices[len(all_indices) - 2])
#             # print("index: ", index)
#             if not (abs(all_indices[len(all_indices) - 2] - index) >= 9 \
#                     and abs(all_indices[len(all_indices) - 2] - index) <= 14):
#                 return False
#         else:
#             # print("all_indices[i + 1]: ", all_indices[i + 1])
#             # print("all_indices[i - 1]: ", all_indices[i - 1])
#             # print("index: ", index)
#             if not ((all_indices[i + 1] - index) >= 9 \
#                     and (all_indices[i + 1] - index) <= 14):
#                 return False
#             if not (abs(all_indices[i - 1] - index) >= 9 \
#                     and abs(all_indices[i - 1] - index) <= 14):
#                 return False
#     return True
# for run_num in range(num_runs):
#     print("run_num: ", run_num)
#     blank_trial_indices = [68,69,70,71]
#     # if even, sample index only 9 or 10 indices above the last draw to not 
#     # run into a situation where there is no slot for a 5th blank trial
#     if run_num % 2 == 0:
#         index_increments = [10,10,10,10,10,10]
#     else:
#         index_increments = [10,10,14,12,11]
#     random.shuffle(index_increments)
#     previous_blank_index = -1
#     for ii in index_increments:
#         new_blank_index = previous_blank_index + ii
#         previous_blank_index = new_blank_index
#         blank_trial_indices.append(new_blank_index)
#     blank_trial_indices = sorted(blank_trial_indices)
#     assert(between9and14continuous(blank_trial_indices[:-3]))
#     print(between9and14continuous(blank_trial_indices[:-3]))
#     run_to_blanks[run_num] = blank_trial_indices
# print(run_to_blanks)

for p_id in range(n_participants):
    random.seed(p_id)
    # make dirs
    participant_path = "conditions_files/participant" + str(p_id) + "_"
    random.shuffle(images_paths)
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
                current_image_list.append("images/" + "blank.jpg")
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
