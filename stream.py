from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from matplotlib import pyplot as plt
import os
from tensorflow.keras.preprocessing import image
import time
import shutil
import datetime
import seaborn as sns
import pandas as pd

model = load_model("C:\\Users\\alert\\Google Drive\ML\\Electric Bird Caster\Model\\my_keras_model.h5")

captures_folder = "C:\\Users\\alert\\AppData\\Roaming\\iSpy\\WebServerRoot\\Media\\Video\\WZQYX\\grabs\\"
                  # "E:\\Electric Bird Caster\\Captured\\video\\TRARO\\grabs\\"
dump_classified_imp_folder = "C:\\Users\\alert\\Google Drive\\ML\\Electric Bird Caster\\"

pretty_names_list = [
    'Crow',
    'Goldfinch breeding M',
    'Goldfinch off duty M or F',
    'No bird ...',
    'Black capped chickadee',
    'Blue jay',
    'Brown headed cowbird F',
    'brown headed cowbird M',
    'Carolina wren',
    'Common Grakle',
    'Downy woodpecker',
    'Eatern Bluebird',
    'Eu starling on-duty Ad',
    'Eu starling off-duty Ad',
    'House finch M',
    'House finch F',
    'House sparrow F/Im',
    'House sparrow M',
    'Mourning dove',
    'Cardinal M',
    'Cardinal F',
    'Norhtern flicker (red)',
    'Pileated woodpecker',
    'Red winged blackbird F/Im',
    'Red winged blackbird M',
    'Squirrel!',
    'Tufted titmouse',
    'White breasted nuthatch']

top_10 = [i for i in range(28)]

bird_history = np.zeros([len(top_10), 60], dtype=np.int16)

fig, ax = plt.subplots(figsize=(17.5, 4.5))

current_bird_count = np.zeros([len(top_10), 1])
latest_labels = ['', '', '', '', '', '', '', '', '', '']

detection_threshold = 0.75

def plot_IDhistory(history, ax, pretty_names_list):
    ax.clear()
    sns.heatmap(history,
                ax=ax,
                annot=True, fmt="d",
                linewidths=.5,
                yticklabels=pretty_names_list,
                cbar=False)

    plt.title("Birds visiting the last hour")
    plt.xlabel("Minutes past now")
    plt.savefig(dump_classified_imp_folder + "classification_graph.png")


def gimme_minute():
    the_time = str(datetime.datetime.now())
    return the_time[14:16]

ref_minute = gimme_minute()

plot_IDhistory(bird_history, ax, pretty_names_list)

while 1 == 1:

    right_now = gimme_minute()
    if ref_minute != right_now:
        ref_minute = gimme_minute()
        bird_history[:, 1:] = bird_history[:, 0:-1]
        bird_history[:, 0] = np.squeeze(current_bird_count)
        current_bird_count = np.zeros([len(top_10), 1])

        # Update figure
        plot_IDhistory(bird_history, ax, pretty_names_list)

    files = os.listdir(captures_folder)
    if files:
        for f in files:
            try:
                img_pil = image.load_img(path=captures_folder + f, target_size=(224, 224, 3))

                img = image.img_to_array(img_pil)

                im_np = preprocess_input(img)

                pred = model.predict(np.expand_dims(im_np, axis=0))
                prob = np.round(np.max(pred) * 100)

                bird_idx = np.argmax(pred)

                if bird_idx != 3:

                    if np.max(pred) > detection_threshold:
                        ti = pretty_names_list[bird_idx] + " (" + str(prob) + "%)"
                    else:
                        ti = pretty_names_list[bird_idx] + "??? (" + str(prob) + "%)"

                    # if bird_idx in top_10:
                    if np.max(pred) > detection_threshold:
                        latest_labels[1:] = latest_labels[0:-1]
                        t = str(datetime.datetime.now())
                        t = t[10:19]
                        latest_labels[0] = t + ": " + ti
                        label_file = open(dump_classified_imp_folder + "label.txt", "w+")
                        for l in latest_labels:
                            label_file.write(l + "\n")
                        label_file.close()


                        current_bird_count[bird_idx] += 1

                    destination_dir = dump_classified_imp_folder + \
                                      "Classified" + "\\" + \
                                      pretty_names_list[bird_idx]

                    if not os.path.isdir(destination_dir):
                        os.makedirs(destination_dir)
                    shutil.copyfile(captures_folder + f, destination_dir + "\\" + f)

                shutil.copyfile(captures_folder + f, dump_classified_imp_folder + "classification_picture.png")
                os.remove(captures_folder + f)

                current_class_file = open(dump_classified_imp_folder + "Current_Classification.txt", "w+")
                if prob > 75:
                    current_class_file.write(pretty_names_list[bird_idx])
                else:
                    current_class_file.write("Not sure")

                current_class_file.close()

            except:
                print("Error reading file!")


    else:
        # If no files were found in the capture folder this round
        print("No files found")
        time.sleep(1)  #

