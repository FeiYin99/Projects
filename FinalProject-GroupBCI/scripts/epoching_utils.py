import csv
import json
import numpy as np
from scipy.signal import butter, sosfiltfilt  



def process_annotations(annotations_csv_path):
    ## Process annotations csv file
    
    doorways = {}
    
    with open(annotations_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        next(reader, None)
        
        
        i = 1
        
        for row in reader:
            if(row[2] != 'N/A' and row[4] == 'N/A'):
                #Patient doorway entries
                event_dict = {'walk': int(row[0]),
                              'door': int(row[1]),
                              't_focus': convert_seconds(row[2]),
                              't_enter': convert_seconds(row[3]),
                              't_5_before': convert_seconds(row[3]) - 5,
                              't_5_after': convert_seconds(row[3]) + 5,
                              'type': 'P'}
                doorways[i] = event_dict
            elif (row[3] == 'N/A' and row[5] != 'N/A'):
                #Other doorway entries
                event_dict = {'walk': int(row[0]),
                              'door': int(row[1]),
                              't_focus': convert_seconds(row[4]),
                              't_enter': convert_seconds(row[5]),
                              't_5_before': convert_seconds(row[5]) - 5,
                              't_5_after': convert_seconds(row[5]) + 5,
                              'type': 'S'}
                doorways[i] = event_dict
            elif (row[3] != 'N/A' and row[5] != 'N/A'):
                event_dict = {'walk': int(row[0]),
                              'door': int(row[1]),
                              't_focus': convert_seconds(row[2]),
                              't_enter': convert_seconds(row[3]),
                              't_5_before': convert_seconds(row[3]) - 5,
                              't_5_after': convert_seconds(row[3]) + 5,
                              'type': 'P'}
                doorways[i] = event_dict
                event_dict = {'walk': int(row[0]),
                              'door': int(row[1]),
                              't_focus': convert_seconds(row[4]), 
                              't_enter': convert_seconds(row[5]),
                              't_5_before': convert_seconds(row[5]) - 5,
                              't_5_after': convert_seconds(row[5]) + 5,
                              'type': 'S'}
                doorways[i+1] = event_dict
                i = i+1
            i = i+1
        
    return doorways


def search_events(event_dict, walk='any', door='any', event_type='any'):
    
    if (walk != 'any' or door != 'any' or event_type != 'any'):
        new_dict = event_dict
        if (walk != 'any'):
            temp_dict = {}
            for event_num in new_dict:
                event = new_dict[event_num]
                if (event.get('walk') == walk):
                    temp_dict[event_num] = event
            new_dict = temp_dict
                
        if (door != 'any'):
            temp_dict = {}
            for event_num in new_dict:
                event = new_dict[event_num]
                if (event['door'] == door):
                    temp_dict[event_num] = event
            new_dict = temp_dict
                    
        if (event_type != 'any'):
            temp_dict = {}
            for event_num in new_dict:
                event = new_dict[event_num]
                if (event['type'] == event_type):
                    temp_dict[event_num] = event
            new_dict = temp_dict
            
        if (len(new_dict) == 0):
            print("No events found for those parameters.")
    else:
        print("No event parameters were set.")
        
    return new_dict


def print_events(my_dict):
    print(json.dumps(my_dict, indent=4))
    
    
def convert_seconds(timestring):
    time_arr = timestring.split(':')
    total_seconds = float(time_arr[0])*60 + float(time_arr[1]) + float(time_arr[2])/1000
    return total_seconds
    
    
def find_corr_time_n_index(time, time_list):
    ## Given a time stamp and a sorted list of time stamps,
    ##   find the closes time stamp from the list and the corresponding index
    
    if time < time_list[0] or time > time_list[-1]:
        print('[WARNING] The given time stamp is out of bound of the sorted time list')
    
    diff_abs = np.abs(np.array(time_list) - time)
    corr_index = np.argmin(diff_abs)
    corr_time = time_list[corr_index]
    return corr_time, corr_index


def get_epochs(ieeg_data, ieeg_time, walk_events, pre_entrance_index, post_entrance_index):
    
    epoch_list = []
    indices = []
    
    for event_num in walk_events:
        t_enter = walk_events.get(int(event_num)).get('t_enter')
        _, idx = find_corr_time_n_index(t_enter, ieeg_time)
        epoch = ieeg_data[:, idx - pre_entrance_index : idx + post_entrance_index]
        epoch_list.append(epoch)
        indices.append(idx)
        
    return np.array(epoch_list), np.array(indices)


def get_non_entrance_epochs(ieeg_data, pre_entrance_idx, post_entrance_idx, pt_entrance_indices, ot_entrance_indices=None):
    
    ## Create a list of ranges from the entrance indices that non entrance epochs cannot overlap with
    entrance_ranges = []
    
    for idx in pt_entrance_indices:
        entrance_ranges.append([idx - pre_entrance_idx, idx + post_entrance_idx])
    
    if ot_entrance_indices is not None:
        for idx in ot_entrance_indices:
            entrance_ranges.append([idx - pre_entrance_idx, idx + post_entrance_idx])
    
    entrance_ranges = np.array(entrance_ranges)
        
    ## Randomly sample non entrance indices from all possible entrance indices,
    ##   resample if start or end overlaps with entrance ranges, 
    ##   
    
    ## Randomly sample non entrance indices
    num_non_entrance_needed = 10 * entrance_ranges.shape[0]
    random_lower_limit = pre_entrance_idx
    random_upper_limit = ieeg_data.shape[1] - post_entrance_idx
    non_entrance_indices = []
    
    ## Continue until we have 10 times the total number of patient and others entrance events
    while len(non_entrance_indices) < num_non_entrance_needed:
        
        ## Randomly select a non entrance index in the valid range
        non_entrance_idx = np.random.randint(random_lower_limit, random_upper_limit)
        
        ## Get the start and stop of the non entrance range corresponding to the selected index
        non_entrance_range_start = non_entrance_idx - pre_entrance_idx
        non_entrance_range_end = non_entrance_idx + post_entrance_idx
        
        ## Check if the start and stop of non entrance range overlaps with any entrance ranges.
        ## This can be done by subtracting the potential index from the entrance ranges.
        ## If an entrance range's start and stop have different signs (product is negative), 
        ##   that means the potential index is in that entrance range, thus overlaping occurs.
        entrance_ranges_diff_start = entrance_ranges - non_entrance_range_start
        entrance_ranges_diff_start_prod = entrance_ranges_diff_start[:, 0] * entrance_ranges_diff_start[:, 1]
        
        entrance_ranges_diff_end = entrance_ranges - non_entrance_range_end
        entrance_ranges_diff_end_prod = entrance_ranges_diff_end[:, 0] * entrance_ranges_diff_end[:, 1]
        
        if np.all(entrance_ranges_diff_start_prod > 0) and np.all(entrance_ranges_diff_end_prod > 0):
            non_entrance_indices.append(non_entrance_idx)
        
    ## Make Epochs from the non entrance indices found
    non_entrance_epochs = []
    
    for idx in non_entrance_indices:
        non_entrance_epochs.append(ieeg_data[:, idx - pre_entrance_idx : idx + post_entrance_idx])
    
    return np.array(non_entrance_epochs)


def band_pass_filter(ieeg_data, lower_passband, upper_passband, butterworth_order=6, sampling_frequency=250):
    
    ## Create filter
    sos = butter(butterworth_order, 
                 [lower_passband, upper_passband], 
                 analog = False, 
                 btype = 'band', 
                 output = 'sos', 
                 fs = sampling_frequency)

    filtered_data = sosfiltfilt(sos, ieeg_data, axis=1)
    return filtered_data
