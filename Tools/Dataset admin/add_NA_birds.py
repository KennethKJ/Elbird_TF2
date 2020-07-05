import os
import pandas as pd
from glob import glob
from PIL import Image

# Define folders
folder = "E:\\bird img\\"

dB_location = "E:\\ML Training Data\\"
dB_filename = "dB of all bird images from all datasets.csv"

image_folder = "E:\\ML Training Data\\(2) Bird Photo Booth Users Group_files\\"

df_filename = 'dB of all bird images from all datasets 3' + '.csv'

# Load data into data frames
image_db = pd.read_csv(dB_location + dB_filename, index_col=None, header=0)
df_map = pd.read_csv(dB_location + "Folder2commonNameMap.csv", index_col=None, header=0)

for b in range(1, len(df_map['Common name'])):

    print(str(b) + ": Working on " + df_map['Common name'][b])

    current_folder = image_folder + df_map['Birds_dB name'][b] + "\\"

    image_filenames = glob(current_folder + '*.jpg')

    common_name = df_map['Common name'][b]

    if common_name == "Background":
        scientific_name = 'NA'

    elif common_name == "European Starling":
        scientific_name = 'Sturnus vulgaris'

    elif common_name == "Squirrel":
        scientific_name = 'Sciurus carolinensis'

    else:
        # Make sure common name is identical
        df_tmp = image_db[image_db['Common name'] == common_name]
        # ... and grab scientific name
        scientific_name = df_tmp['Scientific Name'].index[0]


    age = df_map['Age'][b]
    sex = df_map['Sex'][b]
    if df_map['Breeding Plumage'][b] == 1:
        breeding = 'Yes'
    else:
        breeding = 'Unknown'

    for img_filename in image_filenames:

        imf = img_filename

        try:

            # Try to read image to ensure it exists and can be read
            # image = Image.open(imf)

            path, fn = os.path.split(imf)
            # print(fn)

            data_dict = {'Common name': common_name,
                         'Scientific Name': scientific_name,
                         'Age': age,
                         'Breeding': breeding,
                         'Sex': sex,
                         'Dataset': 'NA Birds & BPB fused',
                         'Folder': df_map['Birds_dB name'][b],
                         'Image Filename': fn,
                         'Full Path': imf,
                         'Makaulay CSV': 'NA'}

            # Append data to dataframe
            image_db = image_db.append(data_dict, ignore_index=True)

        except FileNotFoundError:
            print(imf + " not found, skipping")

        except OSError:
            print(imf + " is causing an OS Error, skipping")

        except:
            print("Something's bad wrong!")
            raise

image_db.to_csv(r'E:\ML Training Data\\' + df_filename, index=False)
