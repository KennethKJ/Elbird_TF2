import datetime as dt
import pandas as pd
import glob


class BirdStats:

    def __init__(self, data_dir="E:\\Electric Bird Caster\\Data\\", DEBUG = False):

        # copy data directory to class
        self.data_dir = data_dir
        self.DEBUG = DEBUG
        # Grab data from files from todays date
        todays_date = str(dt.datetime.today().date())

        todays_date = todays_date.replace('-', '_')

        if self.DEBUG:
            todays_date = '2020_5_16'

        todays_files = glob.glob(self.data_dir + todays_date + '*csv')

        if todays_files == []:

            # If no data exist from today, create new and empty df instance
            self. df = pd.DataFrame(columns=['year',
                                             'month',
                                             'day',
                                             'hour',
                                             'minute',
                                             'second',
                                             'microsecond',
                                             'now',
                                             'birdID',
                                             'bird_name',
                                             'classification_probability_overall',
                                             'classification_probability_instance',
                                             'just_detected',
                                             'loop_cycle',
                                             'bounding_box',
                                             'image_filename'])
        else:

            li = []

            for filename in todays_files:
                df_tmp = pd.read_csv(filename, index_col=None, header=0)
                li.append(df_tmp)

            self.df = pd.concat(li, axis=0, ignore_index=True)

            self.get_basic_stats()


    def add_data(self, data_dict):

        self.df = self.df.append(data_dict, ignore_index=True)


    def get_basic_stats(self):

        self.birds_seen_today = self.df['bird_name'].unique()


        now = dt.datetime.now()



        # self.birds_seen_last_10_mins

    def save_clock_hour_2_csv(self, clock_hour):

        df_filename = str(dt.datetime.today().year) + '_' + \
                      str(dt.datetime.today().month) + '_' + \
                      str(dt.datetime.today().day) + '_' + \
                      str(clock_hour) + '.csv'

        df_tmp = self.df[self.df['hour'] == clock_hour]

        # Save data from selected hour to CSV file
        df_tmp.to_csv(r'E:\\Electric Bird Caster\\Data\\' + df_filename, index=False)

        # Create new and empty data frame for next hour
        df = pd.DataFrame(columns=['year',
                                   'month',
                                   'day',
                                   'hour',
                                   'minute',
                                   'second',
                                   'microsecond',
                                   'now',
                                   'birdID',
                                   'bird_name',
                                   'classification_probability_overall',
                                   'classification_probability_instance',
                                   'just_detected',
                                   'loop_cycle',
                                   'bounding_box',
                                   'image_filename'])

        # Update current hour

# BS = BirdStats(DEBUG=True)
#
# BS.save_clock_hour_2_csv(18)
#
# print("Done")
#





#
# import pandas as pd
# import glob
#
# class BirdStats:
#
# 	def __init__(self, data_dir = "E:\\Electric Bird Caster\\Data\\"):
#
#         self.data_dir = data_dir
# data_dir = "E:\\Electric Bird Caster\\Data\\"
#
# todays_files = glob.glob(self.data_dir + '2020_5_16_*csv')
#
#
# li = []
#
# for filename in todays_files:
#     df_tmp = pd.read_csv(filename, index_col=None, header=0)
#     li.append(df_tmp)
#
# df = pd.concat(li, axis=0, ignore_index=True)
#
# birds_seen_today = df['bird_name'].unique()
#
#
#
#
# # df = pd.read_csv(data_dir + "2020_5_16_$$.csv")
#
#
# print('Done')
