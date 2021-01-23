#!/usr/bin/env python

"""split_data.py: Splits a saved dataset into 80/10/10 train/test/val numpy arrays, with X, y, and weight traits seperated"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
sys.path.append("../utils")

from preprocessing import split_dataframe

def get_args():
    
    parser = argparse.ArgumentParser(description='Splits a saved dataset.')
    parser.add_argument('path', type=str,
                        help='The path to the saved dataset.')
    parser.add_argument('--label_col', type = str, default="target",
                        help='The column name of the target variable in the dataset,')
    parser.add_argument('--weight_col', type = str, default="target",
                        help='The name of the weight trait column in the dataset,')
    parser.add_argument('--shuffle', type = bool, default=True,
                        help='Whether to shuffle the data before splitting.')
    parser.add_argument('--seed', type = int, default=42,
                        help='The random seed to use if shuffling.')
    parser.add_argument('--normalize', type = bool, default=True,
                        help='Whether to normalize the dataset columns.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    print("Spliting dataset at", args.path)
    df = pd.read_csv(args.path)
    arrays = split_dataframe(df, 
                             label_col = args.label_col, 
                             weight_col = args.weight_col,
                             shuffle = args.shuffle, 
                             seed = args.seed, 
                             normalize = args.normalize)
    
    out_path = os.path.dirname(args.path)
    for arr, name in zip(arrays, ["X_train", "y_train", "traits_train", "X_val", "y_val", "traits_val", "X_test", "y_test", "traits_test"]):
        np.save(os.path.join(out_path, name + ".npy"), arr)

    print("Arrays saved to", out_path)