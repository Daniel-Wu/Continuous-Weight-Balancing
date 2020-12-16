#!/usr/bin/env python

"""download_cali_housing.py: Downloads, extracts, and saves the california housing dataset."""

import sys
import os
sys.path.append("../utils")

from preprocessing import skdata_to_table
from sklearn.datasets import fetch_california_housing

if __name__ == "__main__":
    print("Downloading california housing dataset.")
    cali_dataset = fetch_california_housing()
    df = skdata_to_table(cali_dataset)
    
    #Save data
    os.makedirs("../data/cali_housing", exist_ok = True)
    path = "../data/cali_housing/data.csv"
    df.to_csv(path, index = False)
    print("Dataset saved at", path)