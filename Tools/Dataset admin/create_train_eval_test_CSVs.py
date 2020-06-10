import csv
import os
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np

# Define folders
folder = "E:\\bird img\\"

dB_location = "E:\\ML Training Data\\"
dB_filename = "dB of all bird images from all datasets 2.csv"

# Load data into data frames
image_db = pd.read_csv(dB_location + dB_filename, index_col=None, header=0)

# birds = image_db['Common name'].unique()
# with open('E:\\ML Training Data\\bird list.csv', 'w') as f:
#     csv.writer(f).writerow(birds)

image_db = image_db.sample(frac=1).reset_index(drop=True)

num_images = len(image_db)
train_eval_test_proportions = [0.7, 0.2, 0.1]
filenames = ['train', 'eval', 'test']
high_idx = 0
for i, fn in enumerate(filenames):

    low_idx = high_idx
    high_idx = low_idx + int(train_eval_test_proportions[i] * num_images)

    df = image_db.iloc[low_idx : high_idx]
    csv_filename = filenames[i] + '.csv'
    df.to_csv(r'E:\ML Training Data\\' + csv_filename, index=False)

print("Done!")

