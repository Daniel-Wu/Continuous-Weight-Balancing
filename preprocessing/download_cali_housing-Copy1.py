#!/usr/bin/env python

"""Foobar.py: Downloads, extracts, and saves the california housing dataset."""

import sys
import os
sys.path.append("../utils")

from pathlib import Path
from preprocessing import skdata_to_table
from sklearn.datasets import fetch_california_housing

data_dir = Path("../data/hd_uci)
kaggle_path = 'https://www.kaggle.com/ronitf/heart-disease-uci?select=heart.csv'
                
if __name__ == "__main__":
    print("Downloading Heart Disease dataset.")
    
    df = 
    path = data_dir / 'data.csv'
                
    #Make directory
    os.makedirs(path, exist_ok=True)
    
    
    df.to_csv(path)
    
    print("Dataset saved at", path)