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
from itertools import compress
from pyimagesearch.motion_detection import singlemotiondetector as smd
from Tools.color_constants import gimme_color
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
#
# pretty_names_list = [
#     'Crow',
#     'Goldfinch breeding M',
#     'Goldfinch off duty M or F',
#     'No bird detected',  # 3
#     'Black capped chickadee',
#     'Blue jay',
#     'Brown headed cowbird F',
#     'brown headed cowbird M',
#     'Carolina wren',
#     'Common Grakle',
#     'Downy woodpecker',  # 10
#     'Eatern Bluebird',
#     'Eu starling on-duty Ad',
#     'Eu starling off-duty Ad',
#     'House finch M',
#     'House finch F',  # 15
#     'House sparrow F/Im',
#     'House sparrow M',
#     'Mourning dove',
#     'Cardinal M',
#     'Cardinal F',  # 20
#     'Norhtern flicker (red)',
#     'Pileated woodpecker',
#     'Red winged blackbird F/Im',
#     'Red winged blackbird M',
#     'Squirrel!',  # 25
#     'Tufted titmouse',
#     'White breasted nuthatch']  # 27

ct = gimme_color()


pretty_names_list = [
    ['Crow', ct['black']],
    ['Goldfinch breeding M', ct['gold1']],
    ['Goldfinch off duty M or F', ct['gold4']],
    ['No bird detected', ct['black']],  # 3
    ['Black capped chickadee', ct['gray69']],
    ['Blue jay', ct['peacock']],
    ['Brown headed cowbird F', ct['chocolate4']],
    ['Brown headed cowbird M', ct['chartreuse4']],
    ['Carolina wren', ct['chartreuse4']],
    ['Common Grakle', ct['chartreuse4']],
    ['Downy woodpecker', ct['chartreuse4']],   # 10
    ['Eastern Bluebird', ct['chartreuse4']],
    ['Eu starling on-duty Ad', ct['chartreuse4']],
    ['Eu starling off-duty Ad', ct['chartreuse4']],
    ['House finch M', ct['chartreuse4']],
    ['House finch F',  ct['chartreuse4']],  # 15
    ['House sparrow F/Im', ct['chartreuse4']],
    ['House sparrow M', ct['chartreuse4']],
    ['Mourning dove', ct['chartreuse4']],
    ['Cardinal M', ct['chartreuse4']],
    ['Cardinal F', ct['chartreuse4']],   # 20
    ['Northern flicker (red)', ct['chartreuse4']],
    ['Pileated woodpecker', ct['chartreuse4']],
    ['Red winged blackbird F/Im', ct['chartreuse4']],
    ['Red winged blackbird M', ct['chartreuse4']],
    ['Squirrel!', ct['chartreuse4']],   # 25
    ['Tufted titmouse', ct['chartreuse4']],
    ['White breasted nuthatch', ct['chartreuse4']]]   # 27



latest_labels = ['', '', '', '', '', '', '', '', '', '']

detection_threshold = 50

def plot_IDhistory(history, pretty_names_list):
    # ax.clear()

    fig, ax = plt.subplots(figsize=(18.5, 5.5))
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
    names = [pretty_names_list[i,0] for i in idx2]
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

# plot_IDhistory(bird_history, pretty_names_list)

pred_history = np.zeros([1, 3, len(pretty_names_list)])

IP_start = 101
IP = IP_start
# Initialize IP cam
username = "admin"
password = "JuLian50210809"
IP_address = "192.168.10." + str(IP_start)
rtsp_port = "554"
channel = "1"
subtype = "0"
# ss = "rtsp://admin:JuLian50210809@" + IP_address + ":554/cam/realmonitor?channel=1&subtype=00authbasic=YWRtaW46SnVMaWFuNTAyMTA4MDk="


doAVI = False
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


frame_no = 150
property = cv2.CAP_PROP_POS_FRAMES
cap.set(property, frame_no)


print("Video source: " + ss)

ret, frame = cap.read()
if frame is None:
    print('Not able to grab images from IP cam!')
    pass

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
h_frame, w_frame, _ = frame.shape  # img_pil = img_pil.resize((int(w/3), int(h/3)))

raw_image_shape = (h_frame, w_frame)  #  (height, width)


# win_size = (int(h/6), int(h/6))
win_size = (int(224*2), int(224*2))

target_img_size = (224, 224)

step_size = (int(win_size[0]/3), int(win_size[1]/3))

num_steps = (int(np.floor((raw_image_shape[0]/step_size[0]) - 1)),
             int(np.floor((raw_image_shape[1]/step_size[1]) - 1)))


frame_centers = np.zeros((2, num_steps[0], num_steps[1])).astype(int)


for i in range(num_steps[0]):
    for j in range(num_steps[1]):
        frame_centers[0, i, j] = int(i * step_size[0] + win_size[0] / 2)
        frame_centers[1, i, j] = int(j * step_size[1] + win_size[1] / 2)


num_images = num_steps[0] * num_steps[1]

pred_history_2D = np.zeros((0, num_steps[0], num_steps[1]))
bird_idx_history_2D = np.zeros((0, num_steps[0], num_steps[1])).astype(np.int)

the_situation_idx = np.zeros((num_steps[0], num_steps[1])).astype(np.int)
the_situation_prob = np.zeros((num_steps[0], num_steps[1]))

# Plot snippet
fig = plt.figure(figsize=(18, 8))
ax1 = fig.add_subplot(1, 1, 1)


# Motion detectpr
motion = smd.SingleMotionDetector()



th = None

frames_btw_obj_detect = 1
counter = 0
print("Starting loop")

# Initializations
ret, frame = cap.read()
frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

motion.update_bg(frame_bw)
motion.update_bg_main(frame_bw)
colly = "rgbwymcrgbwymcrgbwymcrgbwymcrgbwymcrgbwymcrgbwymcrgbwymcrgbwymcrgbwymcrgbwymcrgbwymc"

colly_rgb = []

loop_count = 0
ID_stay_time = 20  # Seconds before a positive ID has faded away in the species ID panel
classes_seen = np.zeros(len(pretty_names_list),)
system_msg = "xxx"
restart_no = 0
plot_objects = None
plot_labels = None

numClassifications = np.zeros((len(pretty_names_list),))
currentCertainty = np.zeros((len(pretty_names_list),))

while 1 == 1:

    img_count = -1
    # DEBUG SETTINGS

    # counter = 0  #  Preventing any NN

    # Grab next frame from camera
    ret, frame = cap.read()

    if frame is None:
        IP = IP_start-1
        while frame is None:
            print('Frame was None')
            cap.release()

            ax1.clear()

            ax1.set_xlim(0, w_frame)
            ax1.set_ylim(h_frame, 0)

            props_tech_issue = dict(boxstyle='round', facecolor='white', alpha=1)

            ax1.text(50,
                     50,
                     "Sorry. Technical issue!",
                     fontsize=16,
                     verticalalignment='top',
                     bbox=props_tech_issue)

            ax1.text(50,
                     150,
                     "Was not able to capture new image",
                     fontsize=16,
                     verticalalignment='top',
                     bbox=props_tech_issue)


            ax1.text(50,
                     250,
                     "Trying again in 10 seconds ...",
                     fontsize=16,
                     verticalalignment='top',
                     bbox=props_tech_issue)

            plt.draw()
            plt.pause(0.02)
            plt.ioff()
            plt.show()

            print('Trying again in 10 secs ...')
            time.sleep(10)

            IP += 1
            if IP > 110:
                IP = int(100)
            IP_address = "192.168.10." + str(IP)
            ss = "rtsp://" + username + ":" + password + "@" + IP_address + \
                 ":554/cam/realmonitor?channel=" + channel + "&subtype=" + subtype + "&unicast=true&proto=Onvif"

            print('Retrying with IP = ' + str(IP))
            ax1.text(50,
                     350,
                     'Retrying with IP = ' + str(IP),
                     fontsize=16,
                     verticalalignment='top',
                     bbox=props_tech_issue)

            plt.draw()
            plt.pause(0.02)
            plt.ioff()
            plt.show()

            cap = cv2.VideoCapture(ss)
            ret, frame = cap.read()

            counter = 0
        else:
            restart_no += 1

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

    if 1 == 1:

        # Motion detection
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        X = motion.detect(frame_bw)
        if X is not None:
            th, bounding_boxes = X

            # for bb in bounding_boxes:
            #     x, y, w, h = bb
            #     rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            #     ax1.add_patch(rect)

            # plot_objects = None
            # plot_labels = None

            # If the set # of frames has past, do object detection
            if bounding_boxes != [] and doNN:
                # Reset counter


                # all_image_snippets = np.zeros((num_images, target_img_size[0], target_img_size[1], 3))
                max_percent_movement = 0
                active_windows = []
                bb_centers = []
                all_image_snippets = np.zeros((0, target_img_size[0], target_img_size[1], 3)).astype(np.int)


                bb_count = 0
                grid = np.zeros((num_steps[0], num_steps[1])).astype(np.bool)

                for bb in bounding_boxes:

                    x, y, w, h = bb

                    low_x = np.floor(x / step_size[1]).astype(np.int)
                    high_x = np.ceil((x + w) / step_size[1]).astype(np.int)

                    low_y = np.floor(y / step_size[0]).astype(np.int)
                    high_y = np.ceil((y + h) / step_size[0]).astype(np.int)

                    grid[low_y: high_y, low_x: high_x] = True

                    mid_x_v = x + np.int(w/2)
                    mid_y_h = y + np.int(h/2)

                    bb_centers.append((mid_y_h, mid_x_v))

                    circ = patches.Circle((mid_x_v, mid_y_h), radius=30, linewidth=1, edgecolor='w', facecolor='none')
                    ax1.add_patch(circ)

                    bb_count += 1
                    # rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='c', facecolor='none')
                    # ax1.add_patch(rect)

                if counter >= frames_btw_obj_detect:
                    counter = 0
                    img_count = 0

                    x_s = []
                    y_s = []
                    for i in range(num_steps[0]):
                        for j in range(num_steps[1]):
                            if grid[i, j]:
                                x_s.append(j)
                                y_s.append(i)

                                # Grab window from the frame
                                img_snippet = frame[i*step_size[0]: i*step_size[0] + win_size[0],  # height
                                                    j*step_size[1]: j*step_size[1] + win_size[1],  # width
                                                    :]  # channels

                                # Resize to NN image size requirement
                                if img_snippet.shape[0:2] != target_img_size:
                                    img_snippet = cv2.resize(img_snippet, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

                                # Stack onto collection of images to run NN on
                                all_image_snippets = np.concatenate((all_image_snippets, np.expand_dims(img_snippet, axis=0)))
                                img_count += 1


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

                    bird_idx_2d = np.zeros(num_steps).astype(np.int)
                    bird_idx_2d[y_s, x_s] = bird_idx

                    # Get rid of background detections
                    bird_idx_2d[bird_idx_2d == 3] = 0

                    pred_max = np.max(pred, axis=1)*100

                    pred_max_2D = np.zeros(num_steps).astype(np.int)
                    pred_max_2D[y_s, x_s] = pred_max

                    # Get rid of background detections
                    pred_max_2D[bird_idx_2d == 0] = 0

                    # # Get index of all elements that are larger than threshold
                    idx2 = pred_max_2D < detection_threshold
                    pred_max_2D[idx2] = 0
                    bird_idx_2d[idx2] = 0

                    depth = 3
                    if len(pred_history_2D[:, 1, 1]) == depth:  # Keep only last 10
                        pred_history_2D = np.delete(pred_history_2D, 0, 0)
                        bird_idx_history_2D = np.delete(bird_idx_history_2D, 0, 0)

                    bird_idx_history_2D = np.concatenate((bird_idx_history_2D, np.expand_dims(bird_idx_2d, axis=0)))
                    pred_history_2D = np.concatenate((pred_history_2D, np.expand_dims(pred_max_2D, axis=0)))

                    # Continue if the full depth of the history is filled
                    if len(pred_history_2D[:, 1, 1]) >= depth:
                        classes = bird_idx_history_2D
                        plot_objects = []
                        plot_labels = []
                        y_plot_level = 20

                        # classes_seen = np.zeros(len(pretty_names_list), )

                        # Looping over each detected class
                        for k, c in enumerate(np.unique(classes[classes != 0])):

                            # Find windows where current class is present
                            idx = np.where(classes == c)

                            # Convert idx variable to a list of tuple (y, x) coordinates (idx is two arrays of [y] and [x]'s)
                            coords = []
                            for j in range(len(idx[0])):
                                coords.append((idx[0][j], idx[1][j], idx[2][j]))

                            #  Then make a list of the clusters of the current class that are the specified distance apart (i.e. j == True)_
                            cluster_list = []
                            max_dist = 7
                            run_through = 0
                            while coords != []:
                                # Calculate the euclidian distance between windows with current class ID'ed
                                #   and test which ones are less that the specified distance apart
                                dist = distance.cdist(coords, coords, 'euclidean')

                                d = np.logical_and(dist <= (run_through + 1) * max_dist, dist >= run_through*max_dist)
                                # cluster_list.append([coords[i] for i, j in enumerate(d[current_d_idx]) if j == True])
                                cluster_list.append([coords[i] for i, j in enumerate(d[0]) if j == True])
                                coords = [coords[i] for i, j in enumerate(d[0]) if j == False]


                            # For each cluster of the current class, calculate a "bounding box" and make a rect object
                            for C in cluster_list:

                                C_array = np.array(C)

                                # if any(C_array[:, 0] == depth-1):
                                #     break

                                the_situation = [pred_history_2D[xyz[0], xyz[1], xyz[2]] for xyz in C_array]
                                cluster_prop = np.mean(the_situation).astype(np.int)

                                if len(the_situation) > 5 and cluster_prop > 70: # HIT!


                                    # Raise the detected flag!
                                    classes_seen[c] = ID_stay_time


                                    currentCertainty[c] = cluster_prop

                                    numClassifications[c] += 1


                                    # pred_max_2D

                                    max_situation_idx = np.argmax(the_situation)

                                    # x_min = step_size[1] * np.min(C_array[max_situation_idx][2])
                                    # x_max = x_min + win_size[1]
                                    # y_min = step_size[0] * np.min(C_array[max_situation_idx][1])
                                    # y_max = y_min + win_size[0]

                                    x_min = step_size[1] * np.min(C_array[:, 2])
                                    x_max = step_size[1] * np.max(C_array[:, 2]) + win_size[1]
                                    y_min = step_size[0] * np.min(C_array[:, 1])
                                    y_max = step_size[0] * np.max(C_array[:, 1]) + win_size[0]

                                    w = x_max - x_min  # - int(win_size[1]/2)
                                    h = y_max - y_min  # - int(win_size[0]/2)

                                    x_mid = int(x_min + w/2)
                                    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=1.5, edgecolor=colly[c],
                                                             facecolor='none')

                                    class_text = pretty_names_list[c][0] + ' (' + str(cluster_prop) + '%)'
                                    plot_objects.append((rect, (x_min, y_min-45, class_text)))

                                    # plot_labels.append((10, y_plot_level, class_text, 'white'))
                                    # y_plot_level += 50
                # ax1.text(x_min,
                #          y_min,
                #          txt,
                #          fontsize=14,
                #          verticalalignment='top',
                #          bbox=props)


    cols = "rgbrgbrgbrgbrgbrgbrgbrgb"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # if th is not None:
    #     ax1.imshow(th, alpha=0.3)
    doPlotBoundingBoxes = False
    if plot_objects is not None and doPlotBoundingBoxes:
        for rc, txt_info in plot_objects:

            ax1.add_patch(rc)

            x_min, y_min, txt = txt_info
            # ax1.text(x_min,
            #          y_min,
            #          txt,
            #          fontsize=14,
            #          verticalalignment='top',
            #          bbox=props)

            print(txt + '(' + str(datetime.datetime.now()) + ')')
            print('____________________________________________________')

    doPlotLabels = False
    if plot_labels is not None and doPlotLabels:
        for pl in plot_labels:
            x, y, txt, col = pl

            props2 = dict(boxstyle='round', facecolor=col, alpha=0.5)

            ax1.text(x, y,
                     txt,
                     fontsize=14,
                     verticalalignment='top',
                     bbox=props2)


    # if motion.updated:
    props2 = dict(boxstyle='round', facecolor='white', alpha=1)

    ax1.text(10,
             1100,
             "Frame #: " + str(loop_count),
             fontsize=16,
             verticalalignment='top',
             bbox=props2)

    ax1.text(400,
             1100,
             "# imgs: " + str(img_count),
             fontsize=16,
             verticalalignment='top',
             bbox=props2)

    system_msg = "Status: " + "Num restarts = " + str(restart_no) + '; IP=' + str(IP)
    ax1.text(800,
             1100,
             system_msg,
             fontsize=16,
             verticalalignment='top',
             bbox=props2)



    # if plot_objects == []:
    #     msg = 'I see nothing!'
    # else:
    #     msg = ''
        
    # ax1.text(50,
    #          50,
    #          "Message from NN: " + msg,
    #          fontsize=16,
    #          verticalalignment='top',
    #          bbox=props2)

    props3 = dict(boxstyle='round', facecolor='white', alpha=0.8)
    props4 = dict(boxstyle='round', facecolor='green', alpha=1)

    y = 25
    subtractor = 0
    x_pos = 10
    y_step = 50
    for i, c in enumerate(classes_seen):

        if i == 0 or i == 3 or i == 18 or i == 22 or i == 23 or i == 24 or i == 26:

            subtractor += 1
            continue

        total_classifications = int(numClassifications[i])
        txt = pretty_names_list[i][0]

        ax1.text(x_pos,
                 y + y_step * (i - subtractor),
                 txt,
                 fontsize=12,
                 color='black',
                 verticalalignment='center',
                 bbox=props3)

        if c:
            props4['alpha'] = c/ID_stay_time

            ax1.text(x_pos,
                     y + y_step*(i-subtractor),
                     txt,
                     fontsize=12,
                     color='black',
                     verticalalignment='center',
                     bbox=props4)
            classes_seen[i] -= 1






        # ax1.text(300,
        #          1000,
        #          "Thres sum main :" + str(int(motion.sum_thresh_bg_main)),
        #          fontsize=16,
        #          verticalalignment='top',
        #          bbox=props2)
        #
        # ax1.text(600,
        #          1000,
        #          "Thresh sum: " + str(int(motion.sum_thresh_bg)),
        #          fontsize=16,
        #          verticalalignment='top',
        #          bbox=props2)

    # else:
    #     props2 = dict(boxstyle='round', facecolor='red', alpha=0.5)
    #
    #     ax1.text(10,
    #              10,
    #              "Frame #: " + str(loop_count),
    #              fontsize=16,
    #              verticalalignment='top',
    #              bbox=props2)
    #
    #     ax1.text(10,
    #              100,
    #              "Thres sum main :" + str(int(motion.sum_thresh_bg_main)),
    #              fontsize=16,
    #              verticalalignment='top',
    #              bbox=props2)
    #
    #     ax1.text(10,
    #              200,
    #              "Thresh sum: " + str(int(motion.sum_thresh_bg)),
    #              fontsize=16,
    #              verticalalignment='top',
    #              bbox=props2)
    #

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
    loop_count += 1

    # plt.close(fig)

    # time.sleep(0.5)  #
