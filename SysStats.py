import datetime as dt
import pandas as pd
import glob
import numpy as np


class SysStats:

    def __init__(self, stream_folder="E:\\Electric Bird Caster\\", date=None, DEBUG=False):

        # copy data directory to class
        self.stream_folder = stream_folder
        self.sys_data_dir = self.stream_folder + "SysData\\"
        self.image_dir = self.stream_folder + "Captured Images\\"

        self.DEBUG = DEBUG
        # stream_folder = "E:\\Electric Bird Caster\\"

        if date is None:
            # Grab data from files from todays date
            date = str(dt.datetime.today().date())

        data_files = glob.glob(self.sys_data_dir + "sys_data_" + date + '*csv')

        if data_files == []:

            # If no data exist from today, create new and empty df instance
            self.df = pd.DataFrame(columns=['year',
                                            'month',
                                            'day',
                                            'hour',
                                            'minute',
                                            'second',
                                            'microsecond',
                                            'now',
                                            'loop_cycle',
                                            'probabilities'])
        else:

            li = []
            for filename in data_files:
                df_tmp = pd.read_csv(filename, index_col=None, header=0)
                li.append(df_tmp)
            self.df = pd.concat(li, axis=0, ignore_index=True)

    def add_data(self, probabilities, loop):

        data_dict = {'year': dt.datetime.today().year,
                     'month': dt.datetime.today().month,
                     'day': dt.datetime.today().day,
                     'hour': dt.datetime.now().hour,
                     'minute': dt.datetime.now().minute,
                     'second': dt.datetime.now().second,
                     'microsecond': dt.datetime.now().microsecond,
                     'now': dt.datetime.now(),
                     'loop_cycle': loop,
                     'probabilities': probabilities}

        self.df = self.df.append(data_dict, ignore_index=True)

    def save_clock_hour_2_csv(self, clock_hour):

        df_filename = "sys_data_" + str(dt.datetime.today().date()) + "_" + \
                      str(clock_hour) + '.csv'

        df_tmp = self.df[self.df['hour'] == clock_hour]

        # Save data from selected hour to CSV file
        df_tmp.to_csv(r'E:\\Electric Bird Caster\\SysData\\' + df_filename, index=False)
