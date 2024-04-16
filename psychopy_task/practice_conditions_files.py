import random
import copy
import pandas as pd
import pdb
import os
import numpy as np

n_participants = 1000
images_paths = os.listdir("practice_images/")
assert(len(images_paths) == 10)
images_paths = images_paths + images_paths
practice_trials_num = 20
# set blank trials as fixed across participants but varied across trials
for p_id in range(n_participants):
    random.seed(p_id)
    # make dirs
    participant_path = "practice_conditions_files/practice_participant" + str(p_id) + "_"
    random.shuffle(images_paths)
    current_image_list = []
    is_blank_trial_list = []
    trial_index_list = []
    image_index = 0
    for trial_index in range(practice_trials_num):
        trial_index_list.append(trial_index)

        if trial_index == 10:
            current_image_list.append("blank.jpg")
            is_blank_trial_list.append(1)
        else:
            # print("len(images_paths): ",len(images_paths))
            # print("image_index: ",image_index)
            image_path = images_paths[image_index]
            current_image_list.append("practice_images/" + image_path)
            is_blank_trial_list.append(0)
            image_index += 1
    # output study and test
    output_dict = {
        "current_image": current_image_list,
        "trial_index": trial_index_list,
        "is_blank_trial":is_blank_trial_list,
          }
    output_df = pd.DataFrame(output_dict)
    practice_file_path = participant_path + ".csv"
    output_df.to_csv(practice_file_path, index = False)
