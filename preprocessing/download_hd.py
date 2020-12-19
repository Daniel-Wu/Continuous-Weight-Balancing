#!/usr/bin/env python

"""download_cali_housing.py: Downloads, extracts, and saves the california housing dataset."""

import sys
import os
sys.path.append("../utils")

from preprocessing import skdata_to_table
from sklearn.datasets import fetch_california_housing

if __name__ == "__main__":
    print("Downloading heart disease dataset.")

    ## Public GDrive link to Kaggle data
    gdrive_url = 'https://drive.google.com/uc?id=1fF4ZGnGgfKCsuFSnvyuBE_a5oZXPe85B'
 	local_dir = "../data/hd"

    os.makedirs(local_dir, exist_ok = True)

    local_path = os.path.join(local_dir, 'data.csv')

    gdown.download(gdrive_url, local_path, quiet=False)

    print("Dataset saved at", local_path)