import pandas as pd
import numpy as np
import pdb
input_events_file = ""
tr_length = 1.6
df = pd.read_csv('../raw/sub-01_ses-nsd01_task-nsdcore_run-01_events.tsv', sep = "\t", header = 0)
total_TRs = 188
run_num = 1
times_with_stim = []
stim_name_present = []
stim_duration = 3
threshold = tr_length / 2
onsets = df["onset"].tolist()
for i_trial, trial in df.iterrows():
    df.loc[i_trial, "trial_type"] = "COCOimage_" + \
               str(trial["73k_id"]) + "_trialNum_" +\
                  str(trial["trial_number"]) + "_runNum" + str(run_num)
stim_names = df["trial_type"].tolist()

for index, row in df.iterrows():
    for i in range(int(row["duration"]) + 1):
        times_with_stim.append(int(row["onset"]) + i)
        stim_name_present.append("COCOimage_" + str(row["73k_id"]) + \
                                  "_trialNum_" + str(row["trial_number"]) + \
                                    "_runNum" + str(run_num))

tr_list = []
tr_range_list = []
tr_trial_name = [] # "blank" or "COCOimage_" + str(row["73k_id"]) + \
                                #   "_trialNum_" + str(row["trial_number"]) + \
                                #     "_runNum" + str(run_num)
def overlap_amount(range1, range2):
    # see if range1 value 1 or 2 is in range 2
    range1_value1, range1_value2 = range1[0], range1[1]
    range2_value1, range2_value2 = range2[0], range2[1]
    # if the first value of range 1 is within, then ask how much is remaining
    if range1_value1 >= range2_value1 and range1_value1 <= range2_value2: 
        return min(1.6, range2_value2 - range1_value1)
    # if the first value of range 1 is not within, but the second value
    # of range 1 is within, then ask how much it is in it
    if range1_value2 >= range2_value1 and range1_value2 <= range2_value2: 
        return range1_value2 - range2_value1
    return 0


for tr_index in range(1,total_TRs + 1):
    tr_start = (tr_index - 1) * tr_length
    tr_end = tr_index * tr_length
    # go through each onset and see if this tr is within it
    overlapping_events_amount = []
    overlapping_events_name = []
    for onset_index, onset in enumerate(onsets):
        overlap = overlap_amount(range1=(tr_start, tr_end), range2 = (onset, onset + stim_duration))
        if overlap > threshold:
            overlapping_events_amount.append(overlap)
            overlapping_events_name.append(stim_names[onset_index])
    
    if len(overlapping_events_amount) == 0:
        tr_trial_name.append("blank")
    else:
        name_str = ""
        for event_name in overlapping_events_name:
            name_str += event_name
            name_str += "_"
        tr_trial_name.append(name_str)
    tr_list.append(tr_index)
    tr_range_list.append((tr_start,tr_end))
print("tr_trial_name: ", tr_trial_name)
print("tr_list: ", tr_list)
tr_label_hrf = ["", "", ""] + tr_trial_name[:-3]
print("tr_label_hrf: ", tr_label_hrf)
output_dict = {"tr_list": tr_list,
               "tr_trial_name": tr_trial_name,
               "tr_label_hrf": tr_label_hrf}
output_df = pd.DataFrame(output_dict)
output_df.to_csv("tr_labels.csv")