"""
Functions for data preprocessing
"""

import numpy as np
import pandas as pd


#Function for getting a dataframe from a sklearn dataset object
def skdata_to_table(data):
    X = data["data"]
    y = data["target"]
    names = data["feature_names"]
    df = pd.DataFrame(X, columns = names)
    df["target"] = y
    
    return df


def split_dataframe(df, label_col = "target", weight_col = "target", shuffle = True, seed = 42, normalize = True, ):
    """
    Function which takes a dataframe and splits the data into 80/10/10 train/val/test splits
    
    params:
    df: The dataframe containing the data
    label_col: the column to extract and use as labels
    weight_col: the column to extract and use as weights
    shuffle: whether to shuffle the data
    seed: if shuffling, what seed to use
    normalize: if true, normalize the X features (does not notmalize weights or labels)
    
    returns:
    a list of numpy arrays: X_train, y_train, X_val, y_val, X_test, y_test
    """
    np.random.seed(seed)
    n = df.shape[0]
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    
    train_idxs = idxs[:int(0.8 * n)]
    val_idxs = idxs[int(0.8 * n) : int(0.9 * n)]
    test_idxs = idxs[int(0.9 * n):]
    
    X_train = df.iloc[train_idxs].drop(columns = label_col).values
    X_val   = df.iloc[val_idxs  ].drop(columns = label_col).values
    X_test  = df.iloc[test_idxs ].drop(columns = label_col).values
    
    y_train = df.iloc[train_idxs][label_col].values
    y_val   = df.iloc[val_idxs  ][label_col].values
    y_test  = df.iloc[test_idxs ][label_col].values

    wt_train = df.iloc[train_idxs][weight_col].values
    wt_val   = df.iloc[val_idxs  ][weight_col].values
    wt_test  = df.iloc[test_idxs ][weight_col].values
    
    if normalize:
        mu = np.mean(X_train, axis = 0)
        std = np.std(X_train, axis = 0)
        
        X_train = (X_train - mu)/std
        X_val   = (X_val   - mu)/std
        X_test  = (X_test  - mu)/std
    
    return [X_train, y_train, wt_train, X_val, y_val, wt_val, X_test, y_test, wt_test]
