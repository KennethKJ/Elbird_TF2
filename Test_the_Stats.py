import pandas as pd
import glob

data_dir = "E:\\Electric Bird Caster\\Data\\"

todays_files = glob.glob(data_dir + '2020_5_16_*csv')


li = []

for filename in todays_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)



# df = pd.read_csv(data_dir + "2020_5_16_$$.csv")


print('Done')
