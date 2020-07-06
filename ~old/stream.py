print("Running stream")
from tensorflow.keras.models import load_model
import numpy as np
# from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from matplotlib import pyplot as plt
import os
from tensorflow.keras.preprocessing import image
import time
import shutil
import datetime
import seaborn as sns
import pandas as pd

print("Loading model")
model = load_model("C:\\Users\\alert\\Google Drive\ML\\Electric Bird Caster\Model\\my_keras_model.h5")
print("Initializing variables")

develop = False

if not develop:
    captures_folder = "C:\\Users\\alert\\AppData\\Roaming\\iSpy\\WebServerRoot\\Media\\Video\\UOOWS\\grabs\\"
else:
    captures_folder = "C:\\Users\\alert\\Google Drive\\ML\\Electric Bird Caster\\Testing\\"

dump_classified_imp_folder = "C:\\Users\\alert\\Google Drive\\ML\\Electric Bird Caster\\"

pretty_names_list = [
    'Crow',
    'Goldfinch breeding M',
    'Goldfinch off duty M or F',
    'No bird detected',
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

current_bird_count = np.zeros([len(top_10), 1])
latest_labels = ['', '', '', '', '', '', '', '', '', '']

detection_threshold = 70

def plot_IDhistory(history, pretty_names_list):
    # ax.clear()

    fig, ax = plt.subplots(figsize=(17.5, 4.5))

    # Find all non zero entries
    idx = np.argmax(history, axis=1) > 0
    idx[3] = True  # set the "No birds one to True so at least one is there

    # Process history matrix accordingly
    h = history[idx, :]
    a = np.zeros((1, len(history[1, :])))
    h = np.concatenate((a, h, a), axis=0)

    # Get corresponding names
    idx2 = [i for i, x in enumerate(idx) if x]
    names = [pretty_names_list[i] for i in idx2]
    names.insert(0, "")
    names.append("")

    sns.heatmap(h.astype(np.int),
                ax=ax,
                annot=True, fmt="d",
                linewidths=.5,
                yticklabels=names,
                cbar=False)

    plt.title("Birds visiting the last hour")
    plt.xlabel("Minutes past now")
    plt.savefig(dump_classified_imp_folder + "classification_graph.png")
    # plt.show()
    plt.close(fig)


def gimme_minute():
    the_time = str(datetime.datetime.now())
    return the_time[14:16]

ref_minute = gimme_minute()

plot_IDhistory(bird_history, pretty_names_list)

pred_history = np.zeros([1, len(pretty_names_list)+1])
print("Starting loop")
while 1 == 1:

    right_now = gimme_minute()
    if ref_minute != right_now:
        ref_minute = gimme_minute()
        bird_history[:, 1:] = bird_history[:, 0:-1]
        bird_history[:, 0] = np.squeeze(current_bird_count)
        current_bird_count = np.zeros([len(top_10), 1])

        # Update figure
        plot_IDhistory(bird_history, pretty_names_list)

    files = os.listdir(captures_folder)
    if files:
        for f in files:
            try:
                # img_pil = image.load_img(path=captures_folder + f)
                img_pil = image.load_img(path=captures_folder + f)
                w, h = img_pil.size
                img_pil = img_pil.resize((int(w/3), int(h/3)))
                # img_pil.show()

                # im1 = im.crop((left, top, right, bottom))
                img = image.img_to_array(img_pil)

                H_anchor = 2 * 224 - 110 + 112 + 50 + 100
                V_anchor = int(112-75+112+75-25)
                move_factor_horizontal = H_anchor
                move_factor_vertical = V_anchor
                sub_img1 = img[0+move_factor_vertical:224+move_factor_vertical, 0+move_factor_horizontal:224+move_factor_horizontal, :]

                move_factor_horizontal = H_anchor + 112
                move_factor_vertical = V_anchor
                sub_img2 = img[0+move_factor_vertical:224+move_factor_vertical, 0+move_factor_horizontal:224+move_factor_horizontal, :]

                move_factor_horizontal = H_anchor + 224
                move_factor_vertical = V_anchor
                sub_img3 = img[0+move_factor_vertical:224+move_factor_vertical, 0+move_factor_horizontal:224+move_factor_horizontal, :]

                doPlot = False
                if doPlot:
                    fig = plt.figure(figsize=(18, 8))

                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.imshow(sub_img1/255)

                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.imshow(sub_img2/255)

                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.imshow(sub_img3/255)

                    plt.show()
                    plt.close(fig)

                im_np1 = preprocess_input(sub_img1)
                im_np2 = preprocess_input(sub_img2)
                im_np3 = preprocess_input(sub_img3)

                im_np1 = np.expand_dims(im_np1, axis=0)
                im_np2 = np.expand_dims(im_np2, axis=0)
                im_np3 = np.expand_dims(im_np3, axis=0)

                im_final = np.concatenate((im_np1, im_np2, im_np3))

                pred = model.predict(im_final)

                preddy1, preddy2, preddy3 = pred

                bird_idx1 = np.argmax(preddy1)
                bird_idx2 = np.argmax(preddy2)
                bird_idx3 = np.argmax(preddy3)

                prob1 = preddy1[bird_idx1]
                prob2 = preddy2[bird_idx2]
                prob3 = preddy3[bird_idx3]

                bird_idx = 3  # will be 3 if the below doesn't change it
                if bird_idx1 != 3:
                    bird_idx = bird_idx1
                    pred = preddy1

                if bird_idx2 != 3 and prob2 > prob1:
                    bird_idx = bird_idx2
                    pred = preddy2

                if bird_idx3 != 3 and prob3 > prob2:
                    bird_idx = bird_idx3
                    pred = preddy3

                if bird_idx != 1000:
                    new_data = np.append(time.time(), pred)
                    new_data = np.reshape(new_data, [1, 29])
                    pred_history = np.append(pred_history, new_data, axis=0)
                    instant_prob = pred[bird_idx]*100
                    prob = np.round(np.mean(pred_history[-3:, bird_idx + 1]) * 100)
                    print("Instant prob: " + pretty_names_list[bird_idx] + " (" + str(int(instant_prob)) + "%)")

                    if prob > 0:  #  detection_threshold:

                        current_bird_count[bird_idx] += 1

                        # Construct destination directory
                        destination_dir = dump_classified_imp_folder + \
                                          "Classified" + "\\" + \
                                          pretty_names_list[bird_idx]

                        # Check if it already exists
                        if not os.path.isdir(destination_dir):
                            os.makedirs(destination_dir)

                        # Copy current picture to the destination
                        shutil.copyfile(captures_folder + f, destination_dir + "\\" + f)

                        # Copy current picture to be displayed in OBS
                        shutil.copyfile(captures_folder + f, dump_classified_imp_folder + "classification_picture.png")

                        current_class_file = open(dump_classified_imp_folder + "Current_Classification.txt", "w+")
                        timmy = str(datetime.datetime.now())
                        timmy = timmy[11:16]

                        if prob > detection_threshold:
                            current_class_file.write(pretty_names_list[bird_idx] + " (" + timmy + "; " + str(int(prob)) + "%)")
                        else:
                            current_class_file.write("Not sure" + " (" + timmy + ";" + str(int(prob)) + "%)")

                        current_class_file.close()

                        #  Save classified pics
                        move_factor_horizontal = H_anchor
                        move_factor_vertical = V_anchor
                        published_img = img[move_factor_vertical:224 + move_factor_vertical,
                                            move_factor_horizontal:2*224 + move_factor_horizontal, :]

                        fig = plt.figure(figsize=(18, 8))
                        fig.imshow(sub_img1 / 255)
                        #
                        plt.savefig(dump_classified_imp_folder + "current_bird.png")
                        # # plt.show()
                        # plt.close(fig)

                        time.sleep(0.1)
                        print(pretty_names_list[bird_idx] + " (" + timmy + "; " + str(int(prob)) + "%)")

                if not develop:
                    os.remove(captures_folder + f)

            except:
                print("Error reading file!")
                time.sleep(0.5)  #

    else:
        # If no files were found in the capture folder this round
        # print("No files found")
        time.sleep(1)  #

