from collections import defaultdict
import numpy as np


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

def locate_repeat_index_per_run(sub_dict, test_images):
    
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
        
    for trial in test_images:
        idx_list = repeated_trial[trial]
        for i in idx_list:
            curr_run = runs[i]
            sorted_vox[curr_run][trial].append(i)
    
    return repeated_trial, sorted_vox

def average_repeats_across_run(vox, repeated_trial, unique_images):
        
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


def average_repeats_for_each_run(vox, per_run_repeat_idx, unique_images):
    
    sorted_dict = {}
        
    for run in per_run_repeat_idx.keys():
        
        sorted_dict[run] = np.zeros((len(unique_images), vox.shape[1]))
        idx_dict = per_run_repeat_idx[run]
    
        # Average repeated MST images
        for i, img in enumerate(unique_images):

            if img in idx_dict.keys(): # deal with repeated images
                # average all repeats across sessions
                curr_trial_vox = np.mean(vox[idx_dict[img]], axis=0)

            else: # error handeling
                print(f"{img} is not in the list")
                break

            sorted_dict[run][i, :] = curr_trial_vox
            
        assert len(per_run_repeat_idx[run].keys()) == len(unique_images)

    
    return sorted_dict