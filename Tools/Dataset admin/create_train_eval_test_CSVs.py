import csv
import os
import pandas as pd
from glob import glob
from PIL import Image
import numpy as np
import imghdr

# Define folders
# folder = "E:\\bird img\\"

dB_location = "E:\\ML Training Data\\"
dB_filename = "dB of all bird images from all datasets 3.csv"

# Load data into data frames
image_db = pd.read_csv(dB_location + dB_filename, index_col=None, header=0)
ML_training_map = pd.read_csv(dB_location + 'ML training map.csv', index_col=None, header=0)

fileNum = '13'

doExtract = True

if doExtract:

    try:
        # Generate IDs
        full_bird_list_df = pd.DataFrame(columns=['Label', 'ID'])
        ID = -1
        for i in range(len(ML_training_map)):

            if not ML_training_map['Include'][i]:
                continue

            common_name = ML_training_map['Common name'][i]

            if ML_training_map['Unisex'][i]:

                label = common_name
                ID += 1

                # Add to dataframe
                full_bird_list_df = full_bird_list_df.append({'Label': label, 'ID': ID}, ignore_index=True)

            if ML_training_map['Male'][i]:

                label = common_name + ' Male'
                ID += 1

                # Add to dataframe
                full_bird_list_df = full_bird_list_df.append({'Label': label, 'ID': ID}, ignore_index=True)

            if ML_training_map['Female'][i]:

                label = common_name + ' Female'
                ID += 1

                # Add to dataframe
                full_bird_list_df = full_bird_list_df.append({'Label': label, 'ID': ID}, ignore_index=True)

        full_bird_list_df.to_csv(r'E:\ML Training Data\\' + 'Class label to ID map.csv', index=False)

        birds = full_bird_list_df['Label'].unique()
        # with open('E:\\ML Training Data\\bird list.csv', 'w') as f:
        #     csv.writer(f).writerow(birds)

        image_db = image_db[image_db['Common name'] != 'Yellow-rumped Warbler']
        image_db = image_db[image_db['Common name'] != 'Yellow-rumped Warbler (Myrtle)']
        image_db = image_db.reset_index(drop=True)

        partial_load = False
        if partial_load:
            df_train_formatted = pd.read_csv(dB_location + 'files extracted for training 12.csv', index_col=None, header=0)

            where_were_we = len(df_train_formatted)
            last_image_path_and_filename = df_train_formatted['path_and_filename'].iloc[-1]
            while last_image_path_and_filename != image_db['Full Path'].iloc[where_were_we]:
                where_were_we += 1
            else:
                where_were_we += 1

        else:
            where_were_we = 0
            df_train_formatted = pd.DataFrame(columns=['Label', 'path_and_filename', 'ID'])

        # Process each DB entry and make a new CSV
        for i in range(where_were_we, len(image_db)):

            path_filename = image_db['Full Path'][i]

            test_read_image = True
            if test_read_image:
                try:
                    # Check if PIL can read it without error
                    image = Image.open(path_filename)
                    if image.format not in ['JPEG', 'PNG', 'GIF']:
                        print('PIL test: wrong filetype, skipping ' + path_filename + ' (' + image.format + ')')
                        continue

                    # Check for filetype
                    filetype = imghdr.what(path_filename)
                    if filetype not in ['jpeg', 'png', 'gif']:
                        print('IMGHDR test: wrong filetype, skipping ' + path_filename + ' (' + filetype + ')')
                        continue

                except:
                    print("Can't read " + path_filename + ". Skipping")
                    continue

            sex = image_db['Sex'][i]

            common_name = image_db['Common name'][i]

            # Get map for current species
            map = ML_training_map[ML_training_map['Common name'] == common_name]

            if map['Unisex'].values:

                label = common_name

            elif map['Male'].values == 1 and sex == 'Male':

                label = common_name + ' Male'

            elif map['Female'].values == 1 and sex == 'Female':

                label = common_name + ' Female'

            else:
                # print(str(i) + ": skipping " + common_name + " | " + path_filename)

                continue
                # raise Exception("Not able to categorize entry. Sumthin's bad wrong!")

            tmp_ID = full_bird_list_df[full_bird_list_df['Label'] == label]['ID'].values
            if tmp_ID.size == 0:
                continue
            else:
                ID = int(tmp_ID)

            data_dict = {'Label': label, 'path_and_filename': path_filename, 'ID': ID}
            df_train_formatted = df_train_formatted.append(data_dict, ignore_index=True)

            if (i % 1000) == 0:
                print(str(i) + ": " + label)

                df_train_formatted.to_csv(r'E:\ML Training Data\\' + 'files extracted for training ' + fileNum + '.csv',
                                          index=False)

    except:
        print("Mofo exeption")
        print("Saving what can be saved ... ")
        df_train_formatted.to_csv(r'E:\ML Training Data\\' + 'files extracted for training ' + fileNum + '.csv',
                                  index=False)
        print('Bye')
        raise

else:
    df_train_formatted = pd.read_csv(dB_location + 'files extracted for training ' + fileNum + '.csv',
                                     index_col=None, header=0)

# df_train_formatted['ID'] = pd.to_numeric(df_train_formatted['ID'])


# Create train, eval, and test CSVs

num_images = len(df_train_formatted)
train_eval_test_proportions = [0.7, 0.2, 0.1]
filenames = ['train', 'eval', 'test']
high_idx = 0

birds = df_train_formatted['Label'].unique()
print("")
print("*********************")
print("List of the " + str(len(birds)) + " classes:")
for b in birds:
    print(b)
print("*********************")
print("")

# Randomly shuffle dataset
df_train_formatted = df_train_formatted.sample(frac=1).reset_index(drop=True)

for i, fn in enumerate(filenames):

    low_idx = high_idx
    high_idx = low_idx + int(train_eval_test_proportions[i] * num_images)

    df = df_train_formatted.iloc[low_idx : high_idx]
    csv_filename = filenames[i] + '.csv'
    df.to_csv(r'E:\ML Training Data\\' + csv_filename, index=False)

print("Done!")

