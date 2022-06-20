import os
import pandas as pd
import numpy as np

# Load Files
def filelist(root):
    '''
    Return a fully-qualified list of filenames under root directory; 
    sort names alphabetically.
    '''
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            allfiles.append(os.path.join(path, name))
    return sorted(allfiles)

# Process Files
def longest_true_seq(bool_curve):
    '''
    Given an array of booleans,
    return indices of longest streak
    of Trues
    '''
    longest_streak = 0
    longest_streak_idx = []

    idx = [] 
    streak = 0
    for i in range(len(bool_curve)):
        if bool_curve[i] == True:
            streak += 1
            idx.append(i)
        else:
            if streak > longest_streak:
                longest_streak = streak
                longest_streak_idx = idx     
            streak = 0
            idx = []
    if streak > longest_streak and longest_streak < 150: 
        return [0]
    return longest_streak_idx

def add_zeros(curve, bh_start_idx, bh_end_idx, beforeBH_len):
    ''' 
    Return trace with zeros appended to beginning 
    so length of input trace is desired length
    '''
    front_of_curve = curve[:bh_start_idx]
    num_zeros = beforeBH_len - len(front_of_curve)
    zeros = np.zeros([1, num_zeros])[0]
    return np.concatenate((zeros, curve[:bh_end_idx]))

def process_curve(curve, beforeBH_len, afterBH_len):
    '''
    Extract input trace from entire trace
    '''
    deriv = np.diff(curve)
    breath_hold_idx = longest_true_seq(abs(deriv)<=0.001) 
    bh_start_idx = breath_hold_idx[0]
    
    if beforeBH_len==0:
        if len(breath_hold_idx) < 100:
             return [], 0, 0
        bh_end_idx = len(curve)
        beforeBH_len = bh_start_idx
    else:
        if len(breath_hold_idx) < afterBH_len:
             return [], 0, 0
        bh_end_idx = breath_hold_idx[afterBH_len-1] + 1
        if bh_start_idx < beforeBH_len:
            return add_zeros(curve, bh_start_idx, bh_end_idx, beforeBH_len), len(breath_hold_idx)*.01, breath_hold_idx

    curve_start_idx = bh_start_idx - beforeBH_len
    return curve[curve_start_idx:bh_end_idx], len(breath_hold_idx)*.01, breath_hold_idx

# Create Data Frame Format
def get_breath_df(file_root, beforeBH_len=1400, afterBH_len=100, full_trace=False):
    '''
    Given a root with files, get a dataframe of shape (5680, 2)
    with input data traces (arrays) and 
    output data breath_holds (floats) 
    '''
    filenames = filelist(file_root)
    
    orig_curves = []
    csv_breath_holds = []
    traces = []
    data_breath_holds = []
    bh_idxs = []
    bh_start_end = []
    
    for file in filenames:
        if file.endswith('.CSV'):     
            df = pd.read_csv(file, header=None)
            curve = np.array(df.iloc[3:, 0], dtype='float32')
            
            if full_trace:
                input_trace, bh_len, bh_idx = process_curve(curve, 0, 0)
            else:
                input_trace, bh_len, bh_idx = process_curve(curve, beforeBH_len, afterBH_len)
                
            if len(input_trace) != 0:
                orig_curves.append(curve)
                traces.append(input_trace)
                csv_breath_holds.append(float(df.iloc[1,1]))
                data_breath_holds.append(bh_len)
                bh_idxs.append(bh_idx)
                bh_start_end.append((bh_idx[0], bh_idx[-1]))
    
    data = {'Trace': traces,'Csv_breath_holds': csv_breath_holds, 'Data_breath_holds': data_breath_holds, 
            'Full_trace': orig_curves, "breathhold_idx": bh_idxs, 'bh_start_end':bh_start_end}
    return pd.DataFrame(data)


def equalize_len_trace(df, trim_len):
    equal_traces = []
    for i in range(df.shape[0]):
        curve = df.iloc[:,0][i]
        if len(curve) > trim_len:
            start_idx = len(curve)-trim_len
            equal_traces.append(curve[start_idx:])
        elif len(curve) < trim_len:
            num_zeros = trim_len - len(curve)
            
            random_nums = (np.random.random_sample(num_zeros)-0.5)
            box_pts = 100
            box = np.ones(box_pts)/box_pts
            random_nums = np.convolve(random_nums, box, mode = 'same')
            random_nums = random_nums[:num_zeros]
            #zeros = np.zeros([1, num_zeros])[0]
            equal_traces.append(np.concatenate((random_nums, curve)))
        else:
            equal_traces.append(curve)
        
        
            
    df["Trace"] = equal_traces