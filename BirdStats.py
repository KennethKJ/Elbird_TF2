import datetime as dt
import pandas as pd
import glob


class BirdStats:

    def __init__(self, stream_folder="E:\\Electric Bird Caster\\", date=None, DEBUG = False):

        # copy data directory to class
        self.stream_folder = stream_folder
        self.data_dir = self.stream_folder + "Data\\"
        self.image_dir = self.stream_folder + "Captured Images\\"

        self.DEBUG = DEBUG
        stream_folder = "E:\\Electric Bird Caster\\"

        if date is None:
            # Grab data from files from todays date
            date = str(dt.datetime.today().date())
            date = date.replace('-', '_')

        data_files = glob.glob(self.data_dir + date + '*csv')

        if data_files == []:

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

            for filename in data_files:
                df_tmp = pd.read_csv(filename, index_col=None, header=0)
                li.append(df_tmp)

            self.df = pd.concat(li, axis=0, ignore_index=True)

            self.get_basic_stats()


    def add_data(self, data_dict):

        self.df = self.df.append(data_dict, ignore_index=True)


    def get_basic_stats(self):

        df_detected = self.df[self.df['just_detected'] == True]
        self.birds_seen_today = df_detected['bird_name'].unique()

        birds_seen_today_file = open(self.stream_folder + "birds_seen_today.txt", "w+")

        birds_seen_today_txt = ""  # Reset debug info text
        birds_seen_today_txt = birds_seen_today_txt + "*** BIRDS SEEN TODAY *************" + "\n"
        for bird in self.birds_seen_today:
            num_detections = len(df_detected[df_detected['bird_name'] == bird])
            birds_seen_today_txt = birds_seen_today_txt + bird + " (" + str(num_detections) + ")" + "\n"

        birds_seen_today_file.write(birds_seen_today_txt)
        birds_seen_today_file.close()
        # self.birds_seen_today = self.df['bird_name'].unique()


        # now = dt.datetime.now()



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

#
# BS = BirdStats(date="2020_5_17", DEBUG=True)
#
# # BS.save_clock_hour_2_csv(18)
#
# print("Done")





