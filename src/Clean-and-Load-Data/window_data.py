########################## Load Libraries  ###################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt     
import torch
##############################################################################

def split_trace(sequence, n_steps, n_output):
    '''
    Splits a list or array into input sequences of len(n_steps) 
    and target sequences of len(n_output). 
    
    ** Slides window by n_steps + n_ouput
    
    Ex. n_steps = 2, n_output = 1
    [1,2,3,4,5,6] ->>> [1,2] -> [3]
                       [4,5] -> [6]
                           
    Parameters:
            sequence (list or array): A sequence of values
            n_steps (int): Length of input sequences (X)
            n_output (int): Length of target sequences (Y)

    Returns:
            X (np.array): A numpy array of input sequences, each n_steps long
            y (np.array): A numpy array of target sequences, each n_output long
    '''
    
    X, y = list(), list()
    
    i = 0
    while i < len(sequence)-n_output:
        end_ix = i + n_steps
        
        # check if we are beyond the sequence
        if end_ix > len(sequence)-n_output:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_output]
        X.append(seq_x)
        y.append(seq_y)
        
        i += n_output+ n_steps
        
    X = np.array(X)
    y = np.array(y)
    return X, y

def create_windowed_data(n_steps, n_output, trace_column):
    '''
    This function is intended to split each trace in a column of respiratory traces
    and return a data frame of the windowed data.
                           
    Parameters:
            n_steps (int): Length of input sequences (X)
            n_output (int): Length of target sequences (Y)
            trace_column (Pandas series or list of lists): A collection of respitory traces
    Returns:
            df (DataFrame): A pandas dataframe with columns: Trace, next_points, Trace_Id
                            Trace_Id is an Id column that identifies the trace that 
                            each input/target sequence pair originated from
                            
    '''
    
    traces = []
    next_pt = []
    trace_num = []
    i = 0
    
    for row in trace_column:
        Xs, ys = split_trace(row, n_steps, n_output)
        traces.append(Xs)
        next_pt.append(ys)
        trace_num.append(np.ones(len(Xs), dtype='int64')*i)
        i += 1
    
    # Flatten lists 
    traces2 = [np.array(x, dtype='float32') for sublist in traces for x in sublist]
    next_pt2 = [x for sublist in next_pt for x in sublist]
    trace_num2 = [i for row in trace_num for i in row]
    
    # Return in dataframe form
    df = pd.DataFrame({"Trace":traces2, "Next_pts": next_pt2, "Trace_Id":trace_num2})
    
    return df

def sliding_split_trace(sequence, n_steps, n_output):
    '''
    Splits a list or array into input sequences of len(n_steps) 
    and target sequences of len(n_output).
    
    ** Slides window by n_output
    
    Ex. n_steps = 2, n_output = 1
    [1,2,3,4,5,6] ->>> [1,2] -> [3]
                       [2,3] -> [4]
                       [3,4] -> [5]
                       [4,5] -> [6]
                       
    Parameters:
            sequence (list or array): A sequence of values
            n_steps (int): Length of input sequences (X)
            n_output (int): Length of target sequences (Y)

    Returns:
            X (np.array): A numpy array of input sequences, each n_steps long
            y (np.array): A numpy array of target sequences, each n_output long
    
    '''
    
    X, y = list(), list()
    
    i = 0
    while i < len(sequence)-n_output:
        end_ix = i + n_steps
        
        # check if we are beyond the sequence
        if end_ix > len(sequence)-n_output:
            break
            
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:end_ix+n_output]
        X.append(seq_x)
        y.append(seq_y)
        
        i += n_steps
        
    return np.array(X), np.array(y)

def create_sliding_window_data(n_steps, num_outputs, trace_column):
    '''
    This function is intended to split each trace in a column of respiratory traces
    and return a data frame of the windowed data.
    
    ** Slides window by n_output
                           
    Parameters:
            n_steps (int): Length of input sequences (X)
            n_output (int): Length of target sequences (Y)
            trace_column (Pandas series or list of lists): A collection of respitory traces
    Returns:
            df (DataFrame): A pandas dataframe with columns: Trace, next_points, Trace_Id
                            Trace_Id is an Id column that identifies the trace that 
                            each input/target sequence pair originated from
                            
    '''
    
    traces = []
    next_pt = []
    trace_num = []
    i = 0
    for row in trace_column:
        Xs, ys = sliding_split_trace(row, n_steps, num_outputs)
        traces.append(Xs)
        next_pt.append(ys)
        trace_num.append(np.ones(len(Xs), dtype='int64')*i)
        i += 1
    
    traces2 = [np.array(x, dtype='float32') for sublist in traces for x in sublist]
    next_pt2 = [x for sublist in next_pt for x in sublist]
    trace_num2 = [i for row in trace_num for i in row]
    
    df = pd.DataFrame({"Trace":traces2, "Next_pts": next_pt2, "Trace_num":trace_num2})
    
    return df

def reshape_trace(trace):
    '''
    Reshapes trace in form: (len_traces, n, 1)
    '''
    trace = np.transpose(trace)
    return trace.reshape(trace.shape[0],trace.shape[1], 1)

def plot_train_test_results(lstm_model, Xtrain, Ytrain, Xtest, Ytest, unscaled_xtrain, unscaled_xtest, num_rows = 4):
    '''
    Creates plots for sample predictions on training and test sets
    '''

    # input window size
    iw = Xtrain.shape[0]
    ow = Ytest.shape[0]

    # figure setup 
    num_cols = 2
    num_plots = num_rows * num_cols

    fig, ax = plt.subplots(num_rows, num_cols, figsize = (10, 13))

    # plot training/test predictionsabs
    for ii in range(num_rows):
        # train set
        xt = reshape_trace(list(unscaled_xtrain))
        
        jj = np.random.randint(0, len(Xtrain))
        X_train_plt = Xtrain[:, jj, :]
        Y_train_pred = lstm_model.predict(torch.from_numpy(X_train_plt).type(torch.Tensor), target_len = ow)

        ax[ii, 0].plot(np.arange(0, iw), xt[:, jj, 0], 'k', linewidth = 2, label = 'Input')
        ax[ii, 0].plot(np.arange(iw - 1, iw + ow), np.concatenate([[xt[-1, jj, 0]], Ytrain[:, jj, 0]]),
                     color = (0.2, 0.42, 0.72), linewidth = 2, label = 'Target')
        ax[ii, 0].plot(np.arange(iw - 1, iw + ow),  np.concatenate([[xt[-1, jj, 0]], Y_train_pred[:, 0]]),
                     color = (0.76, 0.01, 0.01), linewidth = 2, label = 'Prediction')
        ax[ii, 0].set_xlim([0, iw + ow - 1])
        ax[ii, 0].set_xlabel('$t$')
        ax[ii, 0].set_ylabel('$y$')


        # test set
        xt = reshape_trace(list(unscaled_xtest))
        X_test_plt = Xtest[:, jj, :]
        Y_test_pred = lstm_model.predict(torch.from_numpy(X_test_plt).type(torch.Tensor), target_len = ow)
        
        ax[ii, 1].plot(np.arange(0, iw), xt[:, jj, 0], 'k', linewidth = 2, label = 'Input')
        ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[xt[-1, jj, 0]], Ytest[:, jj, 0]]),
                     color = (0.2, 0.42, 0.72), linewidth = 2, label = 'Target')
        
        ax[ii, 1].plot(np.arange(iw - 1, iw + ow), np.concatenate([[xt[-1, jj, 0]], Y_test_pred[:, 0]]),
                      color = (0.76, 0.01, 0.01), linewidth = 2, label = 'Prediction')
        ax[ii, 1].set_xlim([0, iw + ow - 1])
        ax[ii, 1].set_xlabel('$t$')
        ax[ii, 1].set_ylabel('$y$')
        
        # Labels
        if ii == 0:
            ax[ii, 0].set_title('Train')

            ax[ii, 1].legend(bbox_to_anchor=(1, 1))
            ax[ii, 1].set_title('Test')

        plt.suptitle('LSTM Encoder-Decoder Prediction Examples', x = 0.445, y = 1.)
        plt.tight_layout()
        plt.subplots_adjust(top = 0.95)
        
        plt.savefig('prediction_examples4.png')


    return 


def gather_preds(mymodel, dataset, num_outputs):
    '''
    Gather predictions for a dataset
    '''
    preds = []
    for i in range(dataset.x.shape[1]):
        x_t = dataset.x[:, i, :]
        Y_pred = mymodel.predict(x_t, target_len = num_outputs)
        preds.append(np.array([item for sublist in Y_pred for item in sublist]))
    return preds


def plot_preds(trace):
    '''
    Plot predictions
    '''
    trace = np.concatenate(trace).ravel().tolist()
    plt.plot(trace)
    plt.show()
    
def reconstruct_trace(full_traces, pred_trace):
    '''
    Plot original trace and predictions on same plot to compare
    '''
    fig, axs = plt.subplots(3,3,figsize=(15,15))
    for i in range(3):
        for j in range(3):
            idx = np.random.choice(full_traces.index)
            
            test_trace = np.concatenate(full_traces[idx]).ravel().tolist()
            axs[i,j].plot(test_trace, '#3c434f')
            
            test_pred = np.concatenate(pred_trace[idx]).ravel().tolist()
            axs[i,j].plot(test_pred, 'hotpink',linestyle='-')
 
    plt.show()