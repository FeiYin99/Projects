import numpy as np
import scipy.signal



def merge_ieeg_mark_indices(ieeg_mark_indices, mark_artifact_span):

    ## Merge mark indices into ranges that are within a certain distance between each other
    ##   That distance is defined by mark_artifact_span
    merged_ieeg_mark_ranges = []
    merged_mark_range_start_idx = -1
    merged_mark_range_end_idx = -1
    merged_open = False
    
    for i in range(len(ieeg_mark_indices) - 1):
    
        cur_mark_idx = ieeg_mark_indices[i]
        nxt_mark_idx = ieeg_mark_indices[i + 1]
    
        cur_mark_range_end_idx = cur_mark_idx + mark_artifact_span
    
        if merged_open:
            if cur_mark_range_end_idx < nxt_mark_idx:
                merged_open = False
                merged_mark_range_end_idx = cur_mark_range_end_idx
                merged_ieeg_mark_ranges.append([merged_mark_range_start_idx, merged_mark_range_end_idx])
    
        else:
            if cur_mark_range_end_idx < nxt_mark_idx:
                merged_ieeg_mark_ranges.append([cur_mark_idx, cur_mark_range_end_idx])
    
            else:
                merged_open = True
                merged_mark_range_start_idx = cur_mark_idx
                
    if merged_open:
        merged_mark_range_end_idx = ieeg_mark_indices[-1] + mark_artifact_span
        merged_ieeg_mark_ranges.append([merged_mark_range_start_idx, merged_mark_range_end_idx])
    
    print('Merged mark ranges: ', merged_ieeg_mark_ranges)
    print('Number of merged mark ranges: ', len(merged_ieeg_mark_ranges))
    
    return np.array(merged_ieeg_mark_ranges)


def replace_ieeg_mark_ranges(ieeg_data_channel, ieeg_mark_ranges):

    ## Replace each mark range with data around it
    ##   Use the data to the left and right of the mark range with the same span

    num_ranges = ieeg_mark_ranges.shape[0]
    for i in range(num_ranges):
    
        mark_range_start_idx = ieeg_mark_ranges[i, 0]
        mark_range_end_idx = ieeg_mark_ranges[i, 1]
        mark_range_span = mark_range_end_idx - mark_range_start_idx
        
        ## Define left replacement range
        left_replace_range_start_idx = mark_range_start_idx - mark_range_span
        left_replace_range_end_idx = mark_range_start_idx
    
        ## Define right replacement range
        right_replace_range_start_idx = mark_range_end_idx
        right_replace_range_end_idx = right_replace_range_start_idx + mark_range_span
    
        ## If the left or right replacement range overlaps with other mark ranges, 
        ##   limit replacement ranges such that no overlap occurs
        left_limited = False
        right_limited = False
    
        if i > 0 and left_replace_range_start_idx <= ieeg_mark_ranges[i - 1, 1]:
            left_replace_range_start_idx = ieeg_mark_ranges[i - 1, 1] + 1
            left_limited = True
    
        if i < num_ranges - 1 and right_replace_range_end_idx >= ieeg_mark_ranges[i + 1, 0]:
            right_replace_range_end_idx = ieeg_mark_ranges[i + 1, 0] - 1
            right_limited = True
    
        ## Replace iEEG data in mark ranges with the some combination of left and right replacement ranges
        left_replacement_data = ieeg_data_channel[left_replace_range_start_idx : left_replace_range_end_idx]
        right_replacement_data = ieeg_data_channel[right_replace_range_start_idx : right_replace_range_end_idx]
    
        print('Mark range index: ', i,
            ' Mark range start and end indices: ', [mark_range_start_idx, mark_range_end_idx], 
            ' Mark range span: ', mark_range_span,
            ' Available left and right replacement range span: ', [left_replacement_data.shape[0], right_replacement_data.shape[0]])
    
        ## If either left or right replacement ranges are limited, upsample or repeat the data in that range
        if left_limited and right_limited:
            concat_replacement_data = np.concatenate((left_replacement_data, right_replacement_data))
    
            ## Upsample
            ieeg_data_channel[mark_range_start_idx : mark_range_end_idx] = scipy.signal.resample(concat_replacement_data, mark_range_span)
    
            ## Repeat
            #ieeg_data_channel[mark_range_start_idx : mark_range_end_idx] = np.resize(concat_replacement_data, mark_range_span)
    
        ## If left replacement range alone is limited, use right replacement range
        elif left_limited:
            ieeg_data_channel[mark_range_start_idx : mark_range_end_idx] = right_replacement_data
    
        ## If right replacement range alone is limited, use left replacement range
        elif right_limited:
            ieeg_data_channel[mark_range_start_idx : mark_range_end_idx] = left_replacement_data
    
        ## If both replacement ranges are available, use their average
        else:
            ieeg_data_channel[mark_range_start_idx : mark_range_end_idx] = 0.5 * (left_replacement_data + right_replacement_data)
