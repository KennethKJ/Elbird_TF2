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

from pyimagesearch.motion_detection import singlemotiondetector as smd

import pyttsx3

import matplotlib
matplotlib.use("TkAgg")

doNN = True

# engine = pyttsx3.init()
# engine.say("Initializing bird detector")
# engine.runAndWait()


if doNN:
    print("Loading model ... ")
    model = load_model("C:\\Users\\alert\\Google Drive\ML\\Electric Bird Caster\Model\\my_keras_model.h5")

else:
    print("Skipping model ...")
    model = None

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

detection_threshold = 70

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


# Initialize IP cam
username = "admin"
password = "JuLian50210809"
IP_address = "192.168.10.100"
rtsp_port = "554"
channel = "1"
subtype = "0"
# ss = "rtsp://admin:JuLian50210809@" + IP_address + ":554/cam/realmonitor?channel=1&subtype=00authbasic=YWRtaW46SnVMaWFuNTAyMTA4MDk="


doAVI = True

if doAVI:
    # ss = "E:\Electric Bird Caster\Videos\Test1.avi"
    # ss = "E:\Electric Bird Caster\Videos\Testy.avi"
    ss = "E:\Electric Bird Caster\Videos\MoveDetectAndRecognize.avi"

    cap = cv2.VideoCapture(ss)
    # cap = cv2.VideoCapture("E:\Electric Bird Caster\Videos\sun and birds.avi")
else:
    ss = "rtsp://" + username + ":" + password + "@" + IP_address + \
         ":554/cam/realmonitor?channel=" + channel + "&subtype=" + subtype + "&unicast=true&proto=Onvif"
    cap = cv2.VideoCapture(ss)


frame_no = 0
property = cv2.CAP_PROP_POS_FRAMES
cap.set(property, frame_no)


print("Video source: " + ss)

ret, frame = cap.read()
if frame is None:
    print('Not able to grab images from IP cam!')
    pass

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
h, w, _ = frame.shape  # img_pil = img_pil.resize((int(w/3), int(h/3)))

raw_image_shape = (h, w)  #  (height, width)


# win_size = (int(h/6), int(h/6))
win_size = (int(224*2), int(224*2))

target_img_size = (224, 224)

step_size = (int(win_size[0]/6), int(win_size[1]/6))

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

# Plot snippet
fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(1, 1, 1)


# Motion detectpr
motion = smd.SingleMotionDetector()

plot_objects = None
th = None

frames_btw_obj_detect = 1
counter = 0
print("Starting loop")

# Initializations
ret, frame = cap.read()
frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

motion.update_bg(frame_bw)
motion.update_bg_main(frame_bw)


while 1 == 1:

    # DEBUG SETTINGS

    # counter = 0  #  Preventing any NN

    # Grab next frame from camera
    ret, frame = cap.read()

    if frame is None:
        print('Frame was None')
        cap.release()
        cap = cv2.VideoCapture(ss)
        ret, frame = cap.read()
        counter = 0


    # Increase frame number  up
    counter += 1
    # counter = 1

    try:
        # Change color format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        frame = None
        print("I think there was no frame. Internet might be out. Waiting 10 secs and trying again")
        time.sleep(10)
        continue



    # Clear axis and add current frame
    ax1.clear()
    ax1.imshow(frame.astype(int))

    # Motion detection
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    X = motion.detect(frame_bw)
    if X is not None:
        th, bounding_boxes = X

        # for bb in bounding_boxes:
        #     x, y, w, h = bb
        #     rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        #     ax1.add_patch(rect)

        plot_objects = None

        # If the set # of frames has past, do object detection
        if counter >= frames_btw_obj_detect and bounding_boxes != [] and doNN:
            # Reset counter
            counter = 0


            all_image_snippets = np.zeros((num_images, target_img_size[0], target_img_size[1], 3))
            count = 0
            max_percent_movement = 0
            active_windows = []
            all_image_snippets = np.zeros((0,target_img_size[0], target_img_size[1], 3)).astype(np.int)

            go_vertical = 1
            go_horizontal = 1
            for bb in bounding_boxes:
                x, y, w, h = bb

                mid_x_v = x + np.int(w/2)
                mid_y_h = y + np.int(h/2)

                active_windows.append((mid_y_h, mid_x_v))

                for v in range(1 + 2*go_vertical):
                    for h in range(1 + 2*go_horizontal):

                        # Define horizontal (y) start pixel
                        h_start_pixel = mid_y_h + (v-go_vertical)*step_size[0]
                        # Make sure we don't go past image height
                        h_start_pixel = np.min([h_start_pixel + win_size[0], frame.shape[0]-win_size[0]])
                        # Make sure we don't go below zero
                        h_start_pixel = np.max([h_start_pixel, 0])

                        v_start_pixel = mid_x_v + (h-go_horizontal)*step_size[1]
                        # Make sure we don't go past image width
                        v_start_pixel = np.min([v_start_pixel + win_size[1], frame.shape[1]-win_size[1]])
                        # Make sure we don't go below zero
                        v_start_pixel = np.max([v_start_pixel, 0])

                        # Grab window from the frame
                        img_snippet = frame[h_start_pixel: h_start_pixel + win_size[0],  # height
                                            v_start_pixel: v_start_pixel + win_size[1],  # width
                                            :]  # channels

                        # Resize to NN image size requirement
                        img_snippet = cv2.resize(img_snippet, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

                        # Stack onto collection of images to run NN on
                        all_image_snippets = np.concatenate((all_image_snippets, np.expand_dims(img_snippet, axis=0)))
                        count += 1


                # th_snippet = th[i_h * step_size[0]: i_h * step_size[0] + win_size[0],  # height
                #                   i_v * step_size[1]: i_v * step_size[1] + win_size[1]] /255 # channels

                # percent_movement = np.sum(th_snippet)/th.size * 100
                # max_percent_movement = np.max([max_percent_movement, percent_movement])
                # if percent_movement > 1:
                    # active_windows.append((i_h, i_v))
                    # img_snippet = frame[i_h * step_size[0]: i_h * step_size[0] + win_size[0],  # height
                    #                   i_v * step_size[1]: i_v * step_size[1] + win_size[1],  # width
                    #                   :]  # channels



                    # # Plot snippet
                    # fig = plt.figure(figsize=(18, 8))
                    # ax1 = fig.add_subplot(1, 1, 1)
                    # ax1.imshow(img_snippet.astype(int))
                    # plt.show()
                    # plt.close(fig)

            print('____________________________________________________')
            print('Time: ' + str(datetime.datetime.now()))
            print('Max percent movement: {}'.format(max_percent_movement))
            print('Number of windows: {}'.format(count))

            if all_image_snippets.shape[0] != 0:
                # Pre process all image snippets
                # print('Preprocessing images ...')
                all_image_snippets = preprocess_input(all_image_snippets)

                # Run model predictions
                # print('Running model ...')
                pred = model.predict(all_image_snippets)
                # print('Done')

                ## BIRD IDX
                # Get index of classified birds/animals
                bird_idx = np.argmax(pred, axis=1)
                pred_max = np.max(pred, axis=1)*100

                bird_idx_2D = np.zeros((1, num_steps[0], num_steps[1])).astype(np.int)
                pred_max_2D = np.zeros((1, num_steps[0], num_steps[1])).astype(np.int)

                for i, bi in enumerate(bird_idx):
                    h, v = active_windows[i]
                    bird_idx_2D[0, h, v] = bi
                    pred_max_2D[0, h, v] = pred_max[i]


                # Update bird idx history
                if len(bird_idx_history_2D[:, 1, 1]) > 10:  # Keep only last 10
                    bird_idx_history_2D = np.delete(bird_idx_history_2D, 0, 0)
                bird_idx_history_2D = np.concatenate((bird_idx_history_2D, bird_idx_2D))

                ## BIRD PREDICTIONS
                # Reshape prediction and convert to percentages
                # pred = np.reshape(np.max(pred, axis=1), (num_steps[0], num_steps[1]))*100

                # Zero out the "no bird" detections
                pred_max_2D[bird_idx_2D == 3] = 0

                # Update prediction history
                if len(pred_history_2D[:, 1, 1]) > 10:  # Keep only last 10
                    pred_history_2D = np.delete(pred_history_2D, 0, 0)
                pred_history_2D = np.concatenate((pred_history_2D, pred_max_2D))

                ## MATRIX OF DETECTED CLASSES AND PROPs
                # Find where bird idx was the same the last 3 times (std = 0) and not equal to background (3)
                idx = np.std(bird_idx_history_2D[-3:, :, :], axis=0) + (bird_idx_2D == 3) == 0

                # Calculate mean and use idx to zero out elements of no interets
                pred_mean = np.round(np.mean(pred_history_2D[-3:, :, :], axis=0) * idx).astype(int)
                pred_mean = np.squeeze(pred_mean)

                # Get index of all elements that are larger than threshold
                idx2 = pred_mean > detection_threshold

                # Get teh mean probability and teh index of the final selected classes
                pred_mean = pred_mean * idx2
                classes = bird_idx_2D * idx2
                classes = np.squeeze(classes)

                cols = "rgbrgbrgbrgbrgbrgbrgbrgb"
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

                plot_objects = []

                # Looping over each detected class
                for k, c in enumerate(np.unique(classes[classes != 0])):

                    # Find windows where current class is present
                    idx = np.where(classes == c)

                    # Convert idx variable to a list of tuple (y, x) coordinates (idx is two arrays of [y] and [x]'s)
                    coords = []
                    for j in range(len(idx[0])):
                        coords.append((idx[0][j], idx[1][j]))

                    # Calculate the euclidian distance between windows with current class ID'ed
                    #   and test which ones are less that the specified distance apart
                    d = distance.cdist(coords, coords, 'euclidean') <= 6

                    #  Then make a list of the clusters of the current class that are the specified distance apart (i.e. j == True)_
                    cluster_list = []
                    current_d_idx = 0
                    while True:
                        cluster_list.append([coords[i] for i, j in enumerate(d[current_d_idx]) if j == True])
                        current_d_idx += len(d[0][d[current_d_idx]])
                        if current_d_idx >= len(d[0]):
                            break

                    # For each cluster of the current class, calculate a "bounding box" and make a rect object
                    for C in cluster_list:
                        C_array = np.array(C)
                        x_min = step_size[1] * np.min(C_array[:, 1])
                        x_max = step_size[1] * np.max(C_array[:, 1]) + win_size[1]
                        y_min = step_size[0] * np.min(C_array[:, 0])
                        y_max = step_size[0] * np.max(C_array[:, 0]) + win_size[0]

                        w = x_max - x_min #- int(win_size[1]/2)
                        h = y_max - y_min #- int(win_size[0]/2)
                        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1, edgecolor=cols[k], facecolor='none')

                        cluster_prop = np.mean([pred_mean[xy[0], xy[1]] for xy in C_array]).astype(np.int)

                        class_text = pretty_names_list[c] + ' (' + str(cluster_prop) + '%)'
                        plot_objects.append((rect, (x_min, y_min, class_text)))

        # if th is not None:
        #     ax1.imshow(th, alpha=0.3)

    if plot_objects is not None:
        for rc, txt_info in plot_objects:

            ax1.add_patch(rc)

            x_min, y_min, txt = txt_info
            ax1.text(x_min,
                     y_min,
                     txt,
                     fontsize=14,
                     verticalalignment='top',
                     bbox=props)
            print(txt + '(' + str(datetime.datetime.now()) + ')')
            print('____________________________________________________')



    if motion.updated:
        props2 = dict(boxstyle='round', facecolor='white', alpha=0.5)

        ax1.text(10,
                 10,
                 "UPDATED",
                 fontsize=16,
                 verticalalignment='top',
                 bbox=props2)

        ax1.text(10,
                 100,
                 "Thres sum main :" + str(int(motion.sum_thresh_bg_main)),
                 fontsize=16,
                 verticalalignment='top',
                 bbox=props2)

        ax1.text(10,
                 200,
                 "Thresh sum: " + str(int(motion.sum_thresh_bg)),
                 fontsize=16,
                 verticalalignment='top',
                 bbox=props2)

    else:
        props2 = dict(boxstyle='round', facecolor='red', alpha=0.5)

        ax1.text(10,
                 10,
                 "NOT UPDATED",
                 fontsize=16,
                 verticalalignment='top',
                 bbox=props2)

        ax1.text(10,
                 100,
                 "Thres sum main :" + str(int(motion.sum_thresh_bg_main)),
                 fontsize=16,
                 verticalalignment='top',
                 bbox=props2)

        ax1.text(10,
                 200,
                 "Thresh sum: " + str(int(motion.sum_thresh_bg)),
                 fontsize=16,
                 verticalalignment='top',
                 bbox=props2)



    # # Check if it is time to update stats
    # right_now = gimme_minute()
    # if ref_minute != right_now:
    #     ref_minute = gimme_minute()
    #     bird_history[:, 1:] = bird_history[:, 0:-1]
    #     bird_history[:, 0] = np.squeeze(current_bird_count)
    #     current_bird_count = np.zeros([len(top_10), 1])
    #
    #     # Update figure
    #     plot_IDhistory(bird_history, pretty_names_list)

    plt.draw()
    plt.pause(0.02)
    plt.ioff()
    plt.show()


    # plt.close(fig)

    # time.sleep(0.5)  #
