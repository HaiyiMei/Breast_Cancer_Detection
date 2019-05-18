import os
import sys
import glob
import random
import numpy as np
import pandas as pd
from shutil import copyfile

# Base Directory where data is stored
base_data_dir = './data'

in_loc = base_data_dir
out_loc = os.path.join(base_data_dir, 'data_processed')
all_files = glob.glob(in_loc+'/**/*.png', recursive=True)

for file in all_files:
    name = file.rsplit('/', 1)[1]
    fold = file.rsplit('fold/', 1)[1].split('/', 1)[0]
    ttv_dir = file.rsplit('fold/', 1)[1].split('/', 1)[1].split('/', 1)[0]
    zoom = file.rsplit('/', 1)[1].rsplit('-', 1)[0].rsplit('-', 1)[1]
    tumor_class = file.rsplit('/', 1)[1].split('_', 1)[1].split('-', 1)[0]
    new_folder = os.path.join(out_loc, zoom, ttv_dir, tumor_class[0])
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    copyfile(file, os.path.join(new_folder, name))
