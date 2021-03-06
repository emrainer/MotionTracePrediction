import pandas as pd
import numpy as np

# Function to reduce dataframe memory usage

def reduce_memory_usage(df, verbose=True):
    '''
    Function to decrease memory usage of data frame.
    Adapted from: https://www.kaggle.com/code/kyakovlev/m5-simple-fe/notebook
    
    
    Parameters:
            df (DataFrame): A pandas dataframe

    Returns:
            df (DataFrame): A pandas dataframe with reduced memory usage
            
    '''
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        
        if df[col].dtypes == 'O': # List
            rows = []
            for row in df[col]:
                row = np.array(row).astype('float16')
                rows.append(row)
            df[col] = rows

        if col_type in numerics: 
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb from {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, start_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


            
    
    
    