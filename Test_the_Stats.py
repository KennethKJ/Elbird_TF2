import pandas as pd
import glob

data_dir = "E:\\Electric Bird Caster\\Data\\"

todays_files = glob.glob(data_dir + '2020_5_16_*csv')


li = []

for filename in todays_files:
    df_tmp = pd.read_csv(filename, index_col=None, header=0)
    li.append(df_tmp)

df = pd.concat(li, axis=0, ignore_index=True)

birds_seen_today = df['bird_name'].unique()




# df = pd.read_csv(data_dir + "2020_5_16_$$.csv")


print('Done')
