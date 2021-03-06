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
import cv2
import matplotlib.patches as patches
from scipy.spatial import distance
print("Loading model")


model = load_model("C:\\Users\\alert\\Google Drive\ML\\Electric Bird Caster\Model\\my_keras_model.h5")
print("Initializing variables")

develop = True

if not develop:
    captures_folder = "C:\\Users\\alert\\AppData\\Roaming\\iSpy\\WebServerRoot\\Media\\Video\\UOOWS\\grabs\\"
else:
    captures_folder = "C:\\Users\\alert\\Google Drive\\ML\\Electric Bird Caster\\Testing\\"

dump_classified_imp_folder = "E:\\"
stream_folder = "C:\\Users\\alert\\Google Drive\\ML\\Electric Bird Caster\\"

pretty_names_list = [
    'Crow',
    'Goldfinch breeding M',
    'Goldfinch off duty M or F',
    'No bird detected',  # 3
    'Black capped chickadee',
    'Blue jay',
    'Brown headed cowbird F',
    'brown headed cowbird M',
    'Carolina wren',
    'Common Grakle',
    'Downy woodpecker',  # 10
    'Eatern Bluebird',
    'Eu starling on-duty Ad',
    'Eu starling off-duty Ad',
    'House finch M',
    'House finch F',  # 15
    'House sparrow F/Im',
    'House sparrow M',
    'Mourning dove',
    'Cardinal M',
    'Cardinal F',  # 20
    'Norhtern flicker (red)',
    'Pileated woodpecker',
    'Red winged blackbird F/Im',
    'Red winged blackbird M',
    'Squirrel!', # 25
    'Tufted titmouse',
    'White breasted nuthatch']  # 27

top_10 = [i for i in range(28)]

bird_history = np.zeros([len(top_10), 60], dtype=np.int16)

current_bird_count = np.zeros([len(top_10), 1])
latest_labels = ['', '', '', '', '', '', '', '', '', '']

detection_threshold = 60

def plot_IDhistory(history, pretty_names_list):
    # ax.clear()

    fig, ax = plt.subplots(figsize=(17.5, 4.5))
    # idx = np.repeat(False, 28)
    # for i in range(len(history[:, 1, 1])):
    #     # Find all non zero entries
    idx = np.argmax(history, axis=1) > 0
    #     for j in range(len(idx)):
    #         idx[j] = idx[j] or idx_tmp[j]
    if idx.all() == False:
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
    plt.savefig(stream_folder + "classification_graph.png")
    # plt.show()
    plt.close(fig)


def gimme_minute():
    the_time = str(datetime.datetime.now())
    return the_time[14:16]

ref_minute = gimme_minute()

plot_IDhistory(bird_history, pretty_names_list)

pred_history = np.zeros([1, 3, len(pretty_names_list)])



raw_image_shape = (2448, 3264)  #  (height, width)


win_size = (int(224*1.5), int(224*1.5))

target_img_size = (224, 224)

step_size = (int(win_size[0]/10), int(win_size[1]/10))

num_steps = (int(np.floor((raw_image_shape[0]/step_size[0]) - 1)),
             int(np.floor((raw_image_shape[1]/step_size[1]) - 1)))


frame_centers = np.zeros((2, num_steps[0], num_steps[1])).astype(int)

for i in range(num_steps[0]):
    for j in range(num_steps[1]):
        frame_centers[0, i, j] = int(i * step_size[0] + win_size[0] / 2)
        frame_centers[1, i, j] = int(j * step_size[1] + win_size[1] / 2)





num_images = num_steps[0] * num_steps[1]

pred_history_2D = np.zeros((1, num_steps[0], num_steps[1]))
bird_idx_history_2D = np.zeros((1, num_steps[0], num_steps[1])).astype(np.int)

the_situation_idx = np.zeros((num_steps[0], num_steps[1])).astype(np.int)
the_situation_prob = np.zeros((num_steps[0], num_steps[1]))


print("Starting loop")
while 1 == 1:



    files = os.listdir(captures_folder)
    if files:
        for f in files:

            right_now = gimme_minute()
            if ref_minute != right_now:
                ref_minute = gimme_minute()
                bird_history[:, 1:] = bird_history[:, 0:-1]
                bird_history[:, 0] = np.squeeze(current_bird_count)
                current_bird_count = np.zeros([len(top_10), 1])

                # Update figure
                plot_IDhistory(bird_history, pretty_names_list)

            avoidTry = True
            if avoidTry:
            # try:
                # img_pil = image.load_img(path=captures_folder + f)
                img_pil = image.load_img(path=captures_folder + f)
                # cv2.load

                w, h = img_pil.size
                # img_pil = img_pil.resize((int(w/3), int(h/3)))
                # img_pil.show()

                # im1 = im.crop((left, top, right, bottom))
                img = image.img_to_array(img_pil)

                # Grab image snippets accoding to window ans step sizes
                all_image_snippets = np.zeros((num_images, target_img_size[0], target_img_size[1], 3))
                count = 0
                for i_h in range(num_steps[0]):
                    for i_v in range(num_steps[1]):

                        img_snippet = img[i_h * step_size[0]: i_h * step_size[0] + win_size[0],  # height
                                          i_v * step_size[1]: i_v * step_size[1] + win_size[1],  # width
                                          :]  # channels

                        img_snippet = cv2.resize(img_snippet, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

                        all_image_snippets[count, :, :, :] = img_snippet
                        count += 1

                        # # Plot snippet
                        # fig = plt.figure(figsize=(18, 8))
                        # ax1 = fig.add_subplot(1, 1, 1)
                        # ax1.imshow(img_snippet.astype(int))
                        # plt.show()
                        # plt.close(fig)

                # Pre process all image snippets
                all_image_snippets = preprocess_input(all_image_snippets)

                # Run model predictions
                pred = model.predict(all_image_snippets)

                ## BIRD IDX
                # Get index of classified birds/animals
                bird_idx = np.argmax(pred, axis=1)
                # Reshape to grid
                bird_idx = np.reshape(bird_idx, (num_steps[0], num_steps[1]))

                # Update bird idx history
                if len(bird_idx_history_2D[:, 1, 1]) > 10:  # Keep only last 10
                    bird_idx_history_2D = np.delete(bird_idx_history_2D, 0, 0)
                bird_idx_history_2D = np.concatenate((bird_idx_history_2D, np.expand_dims(bird_idx, axis=0)))

                ## BIRD PREDICTIONS
                # Reshape prediction and convert to percentages
                pred = np.reshape(np.max(pred, axis=1), (num_steps[0], num_steps[1]))*100

                # Zero out the "no bird" detections
                pred[bird_idx == 3] = 0

                # Update prediction history
                if len(pred_history_2D[:, 1, 1]) > 10:  # Keep only last 10
                    pred_history_2D = np.delete(pred_history_2D, 0, 0)
                pred_history_2D = np.concatenate((pred_history_2D, np.expand_dims(pred, axis=0)))

                ## MATRIX OF DETECTED CLASSES AND PROPs
                # Find where bird idx was the same the last 3 times (std = 0) and not equal to background (3)
                idx = np.std(bird_idx_history_2D[-3:, :, :], axis=0) + (bird_idx == 3) == 0

                # Calculate mean and use idx to zero out elements of no interets
                pred_mean = np.round(np.mean(pred_history_2D[-3:, :, :], axis=0) * idx).astype(int)

                # Get index of all elements that are larger than threshold
                idx2 = pred_mean > detection_threshold

                # Get teh mean probability and teh index of the final selected classes
                pred_mean = pred_mean * idx2
                classes = bird_idx * idx2

                # Plot snippet
                fig = plt.figure(figsize=(18, 8))
                ax1 = fig.add_subplot(1, 1, 1)
                ax1.imshow(img.astype(int))


                #               i_v * step_size[1]: i_v * step_size[1] + win_size[1],  # width
                cols = "rgbrgbrgbrgbrgbrgbrgbrgb"
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                for k, c in enumerate(np.unique(classes[classes != 0])):
                    idx = np.where(classes == c)
                    coords = []
                    for j in range(len(idx[0])):
                        coords.append((idx[0][j], idx[1][j]))


                    # for i in range(len(idx[0])):
                    cluster_list = []
                    d = distance.cdist(coords, coords, 'euclidean') <= 12

                    current_d_idx = 0
                    while True:

                        cluster_list.append([coords[i] for i, j in enumerate(d[current_d_idx]) if j == True])

                        current_d_idx += len(d[0][d[current_d_idx]])

                        if current_d_idx >= len(d[0]):
                            break



                    for C in cluster_list:
                        C_array = np.array(C)
                        x_min = step_size[1] * np.min(C_array[:, 1])
                        x_max = step_size[1] * np.max(C_array[:, 1]) + win_size[1]
                        y_min = step_size[0] * np.min(C_array[:, 0])
                        y_max = step_size[0] * np.max(C_array[:, 0]) + win_size[0]

                        w = x_max - x_min
                        h = y_max - y_min
                        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor=cols[k], facecolor='none')
                        # rect = patches.Rectangle((500, 1000), 2000, 250, linewidth=1, edgecolor=cols[k], facecolor='none')
                    #
                    # # (xy, width, height, angle=0.0, ** kwargs)[source]¶
                    #
                    #
                    # y = idx[0][i] * step_size[0]
                    # x = idx[1][i]  * step_size[1]
                    # # Create a Rectangle patch
                    # rect = patches.Rectangle((x, y), win_size[0], win_size[1], linewidth=1, edgecolor=cols[k], facecolor='none')

                    # Add the patch to the Axes
                        ax1.add_patch(rect)
                        ax1.text(x_min, y_min, pretty_names_list[c], fontsize=14,
                                verticalalignment='top', bbox=props)
                    # plt.show()

                plt.show()
                plt.close(fig)

            if not develop:
                    os.remove(captures_folder + f)

            # except:
            #     print("Error reading file!")
            #     time.sleep(0.5)  #

    else:
        # If no files were found in the capture folder this round
        # print("No files found")
        time.sleep(1)  #

