import os
import pandas as pd
from glob import glob
from PIL import Image

filenames = glob('C:\\Users\\alert\\Google Drive\\ML lite\\PictureDownloader\\csv files\\*.csv')
folder = "E:\\bird img\\"

image_db = pd.DataFrame(columns=['Common name',
                                 'Scientific Name',
                                 'Age',
                                 'Breeding',
                                 'Sex',
                                 'Dataset',
                                 'Folder',
                                 'Image Filename',
                                 'Full Path',
                                 'Makaulay CSV'])

for filename in filenames:

    class_name = filename[64:filename.find("ML", 64)-1]
    print("Working on " + class_name)
    df = pd.read_csv(filename)
    current_folder = folder + class_name + "\\"
    image_filenames = glob('E:\\ML Training Data\\Makaulay Library Bird Images\\Images\\' + class_name + '\\*.jpg')

    common_name = df['Common Name'][1]
    scientific_name = df['Scientific Name'][1]

    for i in range(len(df)):

        imf = 'E:\\ML Training Data\\Makaulay Library Bird Images\\Images\\' + class_name + '\\' + str(df['ML Catalog Number'][i]) + '.jpg'
        try:

            # Try to read image to ensure it exists and can be read
            image = Image.open(imf)

            # Get Age/Sex
            S = str(df['Age/Sex'][i]).split(' ', 1)

            # Parse out age
            read_age = S[0]
            if read_age == 'Adult':
                age = 'Adult'
            elif read_age == 'Immature':
                age = 'Immature'
            elif read_age == 'Juvenile':
                age = 'Juvenile'
            else:
                age = 'Unknown'

            # Parse out sex if exist
            if len(S) > 1:
                read_sex = S[1]
                if read_sex == 'Male':
                    sex = 'Male'
                elif read_sex == 'Female':
                    sex = 'Female'
                else:
                    sex = 'Unknown'
            else:
                sex = 'Unknown'

            # Not really an entry in the Makaulay dB, so just set to unknown for now
            breeding = 'Unknown'

            data_dict = {'Common name': common_name,
                         'Scientific Name': scientific_name,
                         'Age': age,
                         'Breeding': breeding,
                         'Sex': sex,
                         'Dataset': 'Makaulay Library Bird Images',
                         'Folder': class_name,
                         'Image Filename': str(df['ML Catalog Number'][i]) + '.jpg',
                         'Full Path': imf,
                         'Makaulay CSV': filename}

            # Append data to dataframe
            image_db = image_db.append(data_dict, ignore_index=True)

        except FileNotFoundError:
            print(str(i) + ': ' + imf + " not found, skipping")

        except OSError:
            print(str(i) + ': ' + imf + " is causing an OS Error, skipping")

        except:
            print("Something's bad wrong!")
            raise

df_filename = 'dB of all bird images from all datasets' + '.csv'

image_db.to_csv(r'E:\\Git Repos\\Elbird TF2\\Tools\\Dataset admin\\' + df_filename, index=False)
