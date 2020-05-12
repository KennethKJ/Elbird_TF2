from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import sys
import datetime
import pandas as pd
import cv2
from scipy.spatial import distance
from pyimagesearch.motion_detection import singlemotiondetector as smd
from Tools.Emailing import Emailer
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import matplotlib
matplotlib.use("TkAgg")

print("Running Electric Birder")

## Settings

# Misc
doNN = True
doAVI = False
plot_mode_on = True
DEBUG = False

minimum_prob = 0  # The minimum probability for selection
main_prob_criteria = 90  # Main criteria for a final classsification (mean of num_classifications)
num_classifications = 5  # number of images across space and time classified within current cluster
ID_stay_time = 1  # Cycles before a positive ID has faded away in the species ID panel
num_cycles_in_history = 3  # Number of cycles in history
max_dist = 5  # max allowed distance when looking for classification clusters of same species

alpha_detect = 0.8
alpha_decay = 0.01

# Folders
stream_folder = "E:\\Electric Bird Caster\\"
image_capture_folder = stream_folder + "Captured Images\\"

# IP Camera parameters
IP_start = 101
IP = IP_start
username = "admin"
password = "JuLian50210809"
IP_address = "192.168.10." + str(IP_start)
rtsp_port = "554"
channel = "1"
subtype = "0"

# Image windowing parameters
win_size = (int(224*2), int(224*2))
target_img_size = (224, 224)
num_image_steps = (3, 8)  # (int(win_size[0]/2), int(win_size[1]/2))

# Load model
print('Neural network starting up, please wait ... ' + "\n")
label_file = open(stream_folder + "label.txt", "w+")
label_file.write('Restarting, please wait ... :| ' + "\n")
label_file.close()

debug_file = open(stream_folder + "debug_info.txt", "w+")
debug_file.write(str(datetime.datetime.now()) + '\n')
debug_file.write('The neural network is starting up \n')
# debug_file.write('This takes a minute or two' + "\n")
debug_file.write('Please wait ... ' + "\n")
debug_file.close()

if doNN:
    print("Loading model ... ")
    model = load_model("C:\\Users\\alert\\Google Drive\ML\\Electric Bird Caster\Model\\my_keras_model.h5")

else:
    print("Skipping model ...")
    model = None

pretty_names_list = [
    'Crow',
    'Goldfinch (breeding M)',
    'Goldfinch (non-breeding M or F)',
    'No bird detected',  # 3
    'Black-capped chickadee',
    'Blue jay',
    'Brown-headed cowbird (F)',
    'Brown-headed cowbird (M)',
    'Carolina wren',
    'Common grakle',
    'Downy woodpecker',  # 10
    'Eastern bluebird',
    'Eu starling',
    'Eu starling off-duty Ad',
    'House finch (M)',
    'House finch (F)',  # 15
    'House sparrow (F/Im)',
    'House sparrow (M)',
    'Mourning dove',
    'Cardinal (M)',
    'Cardinal (F)',  # 20
    'Northern flicker',
    'Pileated woodpecker',
    'Red winged blackbird (F/Im)',
    'Red winged blackbird (M)',
    'Squirrel >:o',  # 25
    'Tufted titmouse',
    'White-breasted nuthatch']  # 27

# Species specific thresholds/hysteresis
prob_hysteresis_higher = []
prob_hysteresis_lower = []
default_higher = 90
default_difference_to_lower = 10
for c in range(len(pretty_names_list)):

    # Troublesome species that need higher criteria
    if (c == 5 or c == 18 or c == 22 or c == 24 or c == 24 or c == 26 or c == 27):

        prob_hysteresis_higher.append(default_higher + 7)
        prob_hysteresis_lower.append(default_higher + 7 - default_difference_to_lower)

    # Regular
    else:

        prob_hysteresis_higher.append(default_higher)
        prob_hysteresis_lower.append(default_higher - default_difference_to_lower)



print("Initializing camera")
if doAVI:
    # ss = "E:\Electric Bird Caster\Videos\Test1.avi"
    # ss = "E:\Electric Bird Caster\Videos\Testy.avi"
    # ss = "E:\Electric Bird Caster\Videos\MoveDetectAndRecognize.avi"
    ss = "E:\\Electric Bird Caster\\Videos\\apr 29.avi"

    cap = cv2.VideoCapture(ss)
    # cap = cv2.VideoCapture("E:\Electric Bird Caster\Videos\sun and birds.avi")
else:
    # ss = "rtsp://" + username + ":" + password + "@" + IP_address + \
    #      ":554/cam/realmonitor?channel=" + channel + "&subtype=" + subtype + "&unicast=true&proto=Onvif"

    ss = "rtsp://admin:JuLian50210809@192.168.0.200:554/Streaming/Channels/101/"

    cap = cv2.VideoCapture(ss)

print("Video source: " + ss)

print("reading frame ... ")
ret, frame = cap.read()
if frame is None:
    print('Not able to grab images from IP cam!')
    pass
print("Camera initialized")

frame_rate = cap.get(cv2.CAP_PROP_FPS)
print("Frame rate = " + str(frame_rate))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Initialize motion detector
motion = smd.SingleMotionDetector()
# motion.update_bg(frame_bw)
# motion.update_bg_main(frame_bw)

# Get image size and calc windowing related parameters
frame_height, frame_width, _ = frame.shape  # img_pil = img_pil.resize((int(w/3), int(h/3)))
step_size = (int(np.floor(frame_height / num_image_steps[0])), int(np.floor(frame_width / num_image_steps[1])))
raw_image_shape = (frame_height, frame_width)  #  (height, width)

print("Image overlap vertical = " + str(np.round(100-(step_size[0] / win_size[0]) * 100).astype(np.int)) + "%")
print("Image overlap horizontal = " + str(np.round(100-(step_size[1] / win_size[1]) * 100).astype(np.int)) + "%")
print("Max images = " + str(num_image_steps[0]*num_image_steps[1]))

# Initialize vars related to classification
pred_probs_2D_history = np.zeros((0, num_image_steps[0], num_image_steps[1]))
bird_classes_2D_history = np.zeros((0, num_image_steps[0], num_image_steps[1])).astype(np.int)

# Below var is set to ID_stay_time when a positive ID is made and then "fades" with 1 per cycle.
# Works as a "low pass filter" to make the displaying of labels less flimsy
birds_seen_lately = np.zeros(len(pretty_names_list), )

# Keep track of number of classifications if each species
bird_classifications_count = np.zeros(len(pretty_names_list), )

# Initialize data frame
df = pd.DataFrame(columns=['year',
                           'month',
                           'day',
                           'hour',
                           'minute',
                           'second',
                           'birdID',
                           'bird_name',
                           'classification_probability_overall',
                           'classification_probability_instance',
                           'loop_cycle',
                           'bounding_box',
                           'image_filename'])

# Initializing E-mailer
e_mailer = Emailer()
debug_email = 'kenneth.kragh.jensen@gmail.com'

# Initialize figure
if plot_mode_on:
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(1, 1, 1)

# Initializing loop variables
current_hour = datetime.datetime.now().hour
last_loop_count = -1
nuthins_seen = 0
restart_no = 0
loop_count = 0

WIN_MODE_FIXED = 0
WIN_MODE_FLOATING = 1
mode = WIN_MODE_FLOATING

if mode == WIN_MODE_FLOATING:
    pred_probs_floating = np.zeros((len(pretty_names_list), 1))
    images_saved = np.zeros((len(pretty_names_list), 1))

print("Starting loop")
try:

    while 1 == 1:

        # Update t
        t = datetime.datetime.now()

        # Reset loop variables
        img_count = 0
        model_pred_time = 0
        frame_rate = cap.get(cv2.CAP_PROP_FPS)

        num_frames = 0
        if doAVI:
            ret, frame = cap.read()
        else:
            # Skip over old frames
            grab_delay = 0
            # print("Loop No " + str(loop_count) + "*********************")
            while grab_delay < 1/(frame_rate+1):
                T = datetime.datetime.now()
                ret = cap.grab()
                dT = datetime.datetime.now() - T
                grab_delay = dT.total_seconds()
                # print("Grab delay = " + str(grab_delay) + " Thr = " + str(1/(frame_rate+1)))
                num_frames += 1
                if num_frames > 50*frame_rate:
                    grab_delay = 1
                    print("Looks like it's caught indefinitely in frame reading loop. Skipping it! " + str(T))

            # Then get most recent frame
            ret, frame = cap.read()
            num_frames += 1

            if DEBUG:
                dt = datetime.datetime.now() - t
                print("Frame read: " + str(dt.total_seconds()))

        # Retry connecting to capture device if frame was none
        if frame is None:

            print('Frame was None ' + str(datetime.datetime.now()))

            wait_time = 0.01
            IP = IP_start - 1
            while frame is None:

                # Let go of capture object
                print(' * Releasing current camera object')
                cap.release()

                # Inform
                print(' * Writing label to streaming SW')
                label_file = open(stream_folder + "label.txt", "w+")
                label_file.write('< Camera connection issues > \n < Retrying to connect in ' + str(wait_time) + ' seconds >')
                label_file.close()

                # print('Trying again in ' + str(wait_time) + ' secs ...')
                print(' * Trying to connect again ...')
                time.sleep(wait_time)

                # Increase last 3 IP numbers
                IP += 1
                if IP > 105:
                    IP = int(100)

                # Generate new string
                IP_address = "192.168.10." + str(IP)
                # ss = "rtsp://" + username + ":" + password + "@" + IP_address + \
                #      ":554/cam/realmonitor?channel=" + channel + "&subtype=" + subtype + "&unicast=true&proto=Onvif"
                ss = "rtsp://admin:JuLian50210809@192.168.0.200:554/Streaming/Channels/101/"
                # Inform
                print(' * Retrying with IP = ' + str(IP))
                label_file = open(stream_folder + "label.txt", "w+")
                label_file.write('< Retrying camera connection with IP = ' + str(IP) + ' >')
                label_file.close()

                # Try again
                cap = cv2.VideoCapture(ss)
                ret, frame = cap.read()

                if frame is None:
                    print(" * Connection unsuccessful ...")

            else:
                restart_no += 1

                # Inform
                label_file = open(stream_folder + "label.txt", "w+")
                label_file.write('< Connection re-established >')
                label_file.close()
                print(' * Connection re-established ' + str(datetime.datetime.now()))

        try:
            # Change color format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            frame = None
            print("I think there was no frame. Internet might be out. Waiting 10 secs and trying again")
            time.sleep(10)
            continue

        # Clear axis and add current frame
        if plot_mode_on:
            ax1.clear()
            ax1.imshow(frame.astype(int))

        ## Motion detection

        # Detect motion
        mt = datetime.datetime.now()

        # Convert to black and white
        frame_bw = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2GRAY)

        # Reduce size to save time on processing
        frame_bw = cv2.resize(frame_bw, dsize=((int(frame_width/2), int(frame_height/2))), interpolation=cv2.INTER_CUBIC)

        th, bounding_boxes = motion.detect(frame_bw)

        dmt = datetime.datetime.now() - mt
        motion_detection_time = dmt.total_seconds()

        if DEBUG:
            dt = datetime.datetime.now() - t
            print("Motion detetion done: " + str(dt.total_seconds()))

        if plot_mode_on and th is not None:

            # Restore size
            th = cv2.resize(th, dsize=(frame_width, frame_height), interpolation=cv2.INTER_CUBIC)

            ax1.imshow(th, alpha=0.25)

        # dmt = datetime.datetime.now() - mt
        # print("Motion detection time: " + str(dmt.total_seconds()))
        motion_detected = False
        if bounding_boxes is not None:  # Motion is detected

            motion_detected = True

            # Initialize image container
            all_image_snippets = np.zeros((0, target_img_size[0], target_img_size[1], 3)).astype(np.int)

            if mode == WIN_MODE_FIXED:

                # Initialize movement indicator grid
                grid = np.zeros((num_image_steps[0], num_image_steps[1])).astype(np.bool)
                for bb in bounding_boxes:

                    # Unpack the single bounding box
                    x, y, w, h = bb

                    # Add bounding box to plot
                    if plot_mode_on:
                        rect = patches.Rectangle((x, y), w, h,
                                                 linewidth=1.5,
                                                 edgecolor="white",
                                                 facecolor='none')
                        ax1.add_patch(rect)

                    # Prevent too many images will be activated
                    if w > win_size[1]:
                        x = x + w / 2 - win_size[1] / 2
                        w = win_size[1]

                    if h > win_size[0]:
                        y = y + h / 2 - win_size[0] / 2
                        w = win_size[0]

                    # Calculate x & y positions on grid ("pixels to idx")
                    # x
                    low_x = np.floor(x / step_size[1]).astype(np.int)
                    high_x = np.ceil((x + w) / step_size[1]).astype(np.int)
                    # y
                    low_y = np.floor(y / step_size[0]).astype(np.int)
                    high_y = np.ceil((y + h) / step_size[0]).astype(np.int)

                    # Set grid values overlapping with bounding box to true
                    grid[low_y: high_y, low_x: high_x] = True

                # Grab images where bounding boxes moving object
                img_count = 0  # reset image count
                x_s = []  # list of x values included in grid (s is for selected)
                y_s = []  # list of y values included in grid
                for i in range(num_image_steps[0]):
                    for j in range(num_image_steps[1]):
                        if grid[i, j]:
                            x_s.append(j)
                            y_s.append(i)

                            # Grab window from the frame
                            img_snippet = frame[i*step_size[0]: i*step_size[0] + win_size[0],  # height
                                                j*step_size[1]: j*step_size[1] + win_size[1],  # width
                                                :]  # channels

                            # Resize to NN image size requirement
                            # TODO this needs to be done only once with all images
                            if img_snippet.shape[0:2] != target_img_size:
                                img_snippet = cv2.resize(img_snippet, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

                            # Stack onto collection of images to run NN on
                            all_image_snippets = np.concatenate((all_image_snippets, np.expand_dims(img_snippet, axis=0)))

                            # Increase image count
                            img_count += 1

                if all_image_snippets.shape[0] != 0:  # Images are present (aren't they always at this stage?)

                    # Preprocess images according to model requirement
                    all_image_snippets = preprocess_input(all_image_snippets)

                    # Run (and time) model predictions
                    # print("Running model on " + str(img_count) + " images")
                    mt = datetime.datetime.now()
                    pred = model.predict(all_image_snippets)
                    dmt = datetime.datetime.now() - mt
                    model_pred_time = dmt.total_seconds()
                    # print("Done after " + str(model_pred_time) + " seconds")

                    # Get index of classified birds/animals
                    bird_classes = np.argmax(pred, axis=1)

                    # Translate class identifications into a matrix ("2D")
                    bird_classes_2D = np.zeros(num_image_steps).astype(np.int)  # reset
                    bird_classes_2D[y_s, x_s] = bird_classes

                    # Get rid of background detections
                    bird_classes_2D[bird_classes_2D == 3] = 0

                    # Get maxima of props and convert to %
                    pred_probs = np.max(pred, axis=1)*100
                    # Translate probabilities into a matrix ("2D")
                    pred_probs_2D = np.zeros(num_image_steps).astype(np.int)  # reset
                    pred_probs_2D[y_s, x_s] = pred_probs

                    # Get rid of background detections
                    pred_probs_2D[bird_classes_2D == 0] = 0

                    # Get index of all elements that are larger than minimum prob threshold
                    idx2 = pred_probs_2D < minimum_prob
                    # Zero out classifications and probs not meeting criteria
                    pred_probs_2D[idx2] = 0
                    bird_classes_2D[idx2] = 0

                    # Purge to only keep set number of cycles in history
                    if len(pred_probs_2D_history[:, 1, 1]) == num_cycles_in_history:
                        pred_probs_2D_history = np.delete(pred_probs_2D_history, 0, 0)
                        bird_classes_2D_history = np.delete(bird_classes_2D_history, 0, 0)

                    # Add detected classes to history
                    bird_classes_2D_history = np.concatenate((bird_classes_2D_history, np.expand_dims(bird_classes_2D, axis=0)))

                    # Add prediction probabilities to history
                    pred_probs_2D_history = np.concatenate((pred_probs_2D_history, np.expand_dims(pred_probs_2D, axis=0)))

                    # Continue only if the full depth of the history is filled
                    if len(pred_probs_2D_history[:, 1, 1]) >= num_cycles_in_history:
                        # print("Line 391")

                        # Looping over detected classes
                        for k, c in enumerate(np.unique(bird_classes_2D_history[bird_classes_2D_history != 0])):

                            # Find the windows where the current class is present
                            idx = np.where(bird_classes_2D_history == c)

                            # Convert idx variable to a list of tuple (y, x) coordinates for use below (idx is two arrays of [y] and [x]'s)
                            coords = []
                            for j in range(len(idx[0])):
                                coords.append((idx[0][j], idx[1][j], idx[2][j]))

                            #  Make a list of the clusters of the current class that are the specified distance apart (i.e. j == True)_
                            cluster_list = []
                            run_through = 0
                            while coords:

                                # Calculate the euclidian distance between windows with current class ID'ed
                                dist = distance.cdist(coords, coords, 'euclidean')

                                # Test which ones are less that the specified distance apart
                                d = np.logical_and(dist <= (run_through + 1) * max_dist, dist >= run_through*max_dist)

                                # Append found cluster of close enough coords to cluster list
                                cluster_list.append([coords[i] for i, j in enumerate(d[0]) if j == True])

                                # Remove found cluster of coords from coords
                                coords = [coords[i] for i, j in enumerate(d[0]) if j == False]

                            # For each cluster of the current class, calculate a "bounding box" and make a rect object
                            for C in cluster_list:

                                # Convert current cluster to a numpy array
                                C_array = np.array(C)

                                # Use array to make a list of probabilities associated with the current cluster
                                the_situation = [pred_probs_2D_history[xyz[0], xyz[1], xyz[2]] for xyz in C_array]

                                # Calculate the mean probability of all predictions within the cluster
                                cluster_prop = np.mean(the_situation).astype(np.int)

                                # Test if criteria is met for a "real" detection
                                if len(the_situation) > num_classifications and cluster_prop > main_prob_criteria: # HIT! (more than 5 detections and over 70% mean certainty)

                                    # Apply higher restrictions on troublesome classes
                                    if (c == 5 or c == 18 or c == 22 or c == 24 or c == 24 or c == 26 or c == 27) and cluster_prop < 90:  # Stricter rule for blue jay and others due to many false alarms
                                        continue

                                    # recategorizing of "off duty" Starlings
                                    if c == 13:
                                        c = 12

                                    # Tracker for e-mailing system
                                    bird_classifications_count[c] += 1

                                    # Raise the detected flag!
                                    birds_seen_lately[c] = ID_stay_time

                                    # Calculate position of detected birdie
                                    x_min = step_size[1] * np.min(C_array[:, 2])
                                    x_max = step_size[1] * np.max(C_array[:, 2]) + win_size[1]
                                    y_min = step_size[0] * np.min(C_array[:, 1])
                                    y_max = step_size[0] * np.max(C_array[:, 1]) + win_size[0]

                                    w = x_max - x_min  # - int(win_size[1]/2)
                                    h = y_max - y_min  # - int(win_size[0]/2)

                                    x_mid = int(x_min + w/2)
                                    y_mid = int(y_min + h/2)

                                    # Save frame to image file (if this is a new frame/loop)
                                    if loop_count != last_loop_count:

                                        # Generate filename for saving image
                                        filename = str(datetime.datetime.today().year) + '_' + \
                                                   str(datetime.datetime.today().month) + '_' +  \
                                                   str(datetime.datetime.today().day) + '_' +  \
                                                   str(datetime.datetime.now().hour) + '_' +  \
                                                   str(datetime.datetime.now().minute) + '_' +  \
                                                   str(datetime.datetime.now().second) + '_' +  \
                                                   str(datetime.datetime.now().microsecond) + '_' +  \
                                                   'loop_' + str(loop_count) + \
                                                   '.jpg'

                                        # Save frame to image
                                        cv2.imwrite(image_capture_folder + filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                                    # Save loop count
                                    last_loop_count = loop_count

                                    # Assemble dict of data for dataframe
                                    data_dict = {'year': datetime.datetime.today().year,
                                                 'month': datetime.datetime.today().month,
                                                 'day': datetime.datetime.today().day,
                                                 'hour': datetime.datetime.now().hour,
                                                 'minute': datetime.datetime.now().minute,
                                                 'second': datetime.datetime.now().second,
                                                 'birdID': c,
                                                 'bird_name': pretty_names_list[c][0],
                                                 'classification_probability': cluster_prop,
                                                 'loop_cycle': loop_count,
                                                 'horz_location': x_mid,
                                                 'vert_location': y_mid,
                                                 'image_filename': filename}

                                    # Append data to dataframe
                                    df = df.append(data_dict, ignore_index=True)

            elif mode == WIN_MODE_FLOATING:

                if DEBUG:
                    print('Num BBs = ' + str(len(bounding_boxes)))

                for bb in bounding_boxes:

                    # Unpack the single bounding box
                    x, y, w, h = bb

                    # Restore actual size from the motion detection image size reduction
                    x *= 2
                    y *= 2
                    w *= 2
                    h *= 2

                    if plot_mode_on:
                        rect = patches.Rectangle((x, y), w, h,
                                                 linewidth=1.5,
                                                 edgecolor="black",
                                                 facecolor='none')
                        ax1.add_patch(rect)

                    # Find bb center
                    x_c = x + w/2
                    y_c = y + h/2

                    growth_factor = 1.1
                    # growth_factor = 1 + (frame_height/h - 1)
                    # Make bb square and apply growth factor
                    if w > h:
                        h = w*growth_factor
                        w = h

                    else:
                        w = h*growth_factor
                        h = w

                    # Limit the height and width
                    h = min(h, frame_height)
                    w = min(w, frame_width)

                    # Calculate new x,y and make sure they are above 0
                    x = max(0, x_c - w/2)
                    y = max(0, y_c - h/2)

                    #
                    # # Redefine bb
                    # h = win_size[0]
                    # w = win_size[1]
                    # y = y_c - h/2
                    # x = x_c - w/2

                    # Assure the new bb still fits within image
                    if x + w > frame_width:
                        x = x - (x+w - frame_width)

                    if y + h > frame_height:
                        y = y - (y+h - frame_height)

                    # Assure integer values
                    h = int(h)
                    w = int(w)
                    x = int(x)
                    y = int(y)

                    # Add new bounding box to plot
                    rect = patches.Rectangle((x, y), w, h,
                                             linewidth=2,
                                             edgecolor="white",
                                             facecolor='none')
                    ax1.add_patch(rect)

                    if not doNN:
                        continue

                    # Grab image
                    img_snippet = frame[y: y + h,  # height
                                        x: x + w,  # width
                                        :]  # channels

                    # Resize image
                    if (x < 0) or (y < 0):
                        print("WRONG!")
                        print(x)
                        print(y)
                    else:
                        img_snippet = cv2.resize(img_snippet, dsize=target_img_size, interpolation=cv2.INTER_CUBIC)

                    # Stack onto collection of images to run NN on
                    try:
                        all_image_snippets = np.concatenate((all_image_snippets, np.expand_dims(img_snippet, axis=0)))
                    except:
                        print("oops")

                    # Increase image count
                    img_count += 1

                if all_image_snippets.shape[0] != 0:  # Images are present (aren't they always at this stage?)

                    # Preprocess images according to model requirement
                    all_image_snippets = preprocess_input(all_image_snippets)

                    # Run (and time) model predictions
                    # print("Running model on " + str(img_count) + " images")
                    mt = datetime.datetime.now()
                    pred = model.predict(all_image_snippets)
                    dmt = datetime.datetime.now() - mt
                    model_pred_time = dmt.total_seconds()
                    # print("Done after " + str(model_pred_time) + " seconds")

                    if DEBUG:
                        dt = datetime.datetime.now() - t
                        print("Model predictions done: " + str(dt.total_seconds()))

                    # Get index of classified birds/animals
                    bird_classes = np.argmax(pred, axis=1)

                    # Get maxima of props and convert to %
                    pred_probs = np.max(pred, axis=1) * 100

                    # Remove background classifications
                    pred_probs = pred_probs[bird_classes != 3]
                    bird_classes = bird_classes[bird_classes != 3]

                    # Cycle through each classification and update filter
                    for i, c in enumerate(bird_classes):

                        if pred_probs[i] > pred_probs_floating[c][0]:

                            # One pole filter
                            pred_probs_floating[c][0] = alpha_detect * pred_probs[i] + \
                                (1-alpha_detect) * pred_probs_floating[c][0]

                    for c, p in enumerate(pred_probs_floating):

                        if p >= prob_hysteresis_higher[c]:

                            # Tracker for e-mailing system
                            bird_classifications_count[c] = 1

                            birds_seen_lately[c] = 1



                    # Cycle through each classification again for data saving
                    filename = ''
                    for i, c in enumerate(bird_classes):

                        # Save frame to image file (if this is a new frame/loop and anything was classified)
                        if loop_count != last_loop_count and birds_seen_lately[c] and  pred_probs[i] > 80 and images_saved[c][0] < 100:

                            images_saved[c][0] += 1

                            # Generate filename for saving image
                            filename = str(datetime.datetime.today().year) + '_' + \
                                       str(datetime.datetime.today().month) + '_' + \
                                       str(datetime.datetime.today().day) + '_' + \
                                       str(datetime.datetime.now().hour) + '_' + \
                                       str(datetime.datetime.now().minute) + '_' + \
                                       str(datetime.datetime.now().second) + '_' + \
                                       str(datetime.datetime.now().microsecond) + '_' + \
                                       'loop_' + str(loop_count) + \
                                       '.jpg'

                            # Save frame to image
                            cv2.imwrite(image_capture_folder + filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                        # Save loop count
                        last_loop_count = loop_count

                        # Save to data frame if detection has been made
                        if birds_seen_lately[c] and pred_probs[i] > 70:
                            # Assemble dict of data for data frame
                            data_dict = {'year': datetime.datetime.today().year,
                                         'month': datetime.datetime.today().month,
                                         'day': datetime.datetime.today().day,
                                         'hour': datetime.datetime.now().hour,
                                         'minute': datetime.datetime.now().minute,
                                         'second': datetime.datetime.now().second,
                                         'birdID': c,
                                         'bird_name': pretty_names_list[c],
                                         'classification_probability_overall': pred_probs_floating[c][0],
                                         'classification_probability_instance': pred_probs[i],
                                         'loop_cycle': loop_count,
                                         'bounding_box': bounding_boxes[i],
                                         'image_filename': filename}

                            # Append data to dataframe
                            df = df.append(data_dict, ignore_index=True)

        for c in range(len(pred_probs_floating)):

            # Apply decay
            pred_probs_floating[c][0] = alpha_decay * minimum_prob + \
                                        (1 - alpha_decay) * pred_probs_floating[c][0]

            if pred_probs_floating[c][0] < prob_hysteresis_lower[c]:

                 birds_seen_lately[c] = 0


        # Create label file for OBS
        label_txt = ''
        if np.sum(birds_seen_lately) > 0:

            for c, yes in enumerate(birds_seen_lately):

                # # Subtract one to "forget" some over each cycle
                # if birds_seen_lately[c] > 0:
                #     birds_seen_lately[c] -= 1

                if yes:
                    # running_prob = str(np.around(pred_probs_floating[c][0], decimals=1))
                    # label_txt = label_txt + pretty_names_list[c] + " - (" + running_prob + "%)" + "\n"
                    label_txt = label_txt + pretty_names_list[c] + "\n"
                    nuthins_seen = 0

            label_file = open(stream_folder + "label.txt", "w+")
            label_file.write(label_txt)
            label_file.close()

        elif nuthins_seen == 0:  # When no detections has been made

            label_file = open(stream_folder + "label.txt", "w+")
            label_file.write(' - - - ' + "\n")
            label_file.close()

            nuthins_seen = 1  # This switch is to ensure to write this ti label file only once if a strech of nothing is going on at the feeder

        # Save dataframe to csv file on disk every 5 minutes
        if loop_count % 300 == 0:
            # print("Saving data frame to disk ")

            df_filename = str(datetime.datetime.today().year) + '_' + \
                          str(datetime.datetime.today().month) + '_' + \
                          str(datetime.datetime.today().day) + '_' + \
                          str(current_hour) + '.csv'

            df.to_csv(r'E:\\Electric Bird Caster\\Data\\' + df_filename, index=False)

            if DEBUG:
                dt = datetime.datetime.now() - t
                print("Data frame interim saving done: " + str(dt.total_seconds()))

        # Save data frame and create a new if a new hour of the day has started
        if current_hour != datetime.datetime.now().hour:
            print("Saving data frame to disk and create new for next hour")
            df_filename = str(datetime.datetime.today().year) + '_' + \
                          str(datetime.datetime.today().month) + '_' + \
                          str(datetime.datetime.today().day) + '_' + \
                          str(current_hour) + '.csv'

            # Save current data frame to CSV file
            df.to_csv(r'E:\\Electric Bird Caster\\Data\\' + df_filename, index=False)

            # Create new and empty data frame for next hour
            df = pd.DataFrame(columns=['year',
                                       'month',
                                       'day',
                                       'hour',
                                       'minute',
                                       'second',
                                       'birdID',
                                       'bird_name',
                                       'classification_probability_overall',
                                       'classification_probability_instance',
                                       'loop_cycle',
                                       'bounding_box',
                                       'image_filename'])

            # Update current hour
            current_hour = datetime.datetime.today().hour

            if DEBUG:
                dt = datetime.datetime.now() - t
                print("Data frame hourly saving done: " + str(dt.total_seconds()))

        # E-mail notification
        bird_of_interest = 22
        bird_of_interest_threshold = 5
        for i, c in enumerate(bird_classifications_count):
            if i == bird_of_interest and c >= bird_of_interest_threshold:

                emailContent = "A pretty " + pretty_names_list[i] + " has been seen " + str(bird_classifications_count[i]) \
                               + " times now. Most recently now at " + str(datetime.datetime.now()) + "\n" \
                               + "See attached image :) "
                e_mailer.sendmail("kenneth.kragh.jensen@gmail.com", "Electric Birder Alert!", emailContent, image_capture_folder + filename)
                e_mailer.sendmail("jensenmeredithl@gmail.com", "Electric Birder Alert!", emailContent, image_capture_folder + filename)

                # Reset count
                bird_classifications_count[i] = 0

                if DEBUG:
                    dt = datetime.datetime.now() - t
                    print("E-mail processing done: " + str(dt.total_seconds()))


        # Update debug info
        debug_txt = ""  # Reset debug info text
        debug_txt = debug_txt + "Loop " + str(loop_count) + "\n"
        debug_txt = debug_txt + "Frame jumps : " + str(num_frames) + "\n"
        if motion_detected:
            debug_txt = debug_txt + "Motion detected" + "\n"
        else:
            debug_txt = debug_txt + "No motion" + "\n"

        debug_txt = debug_txt + "Mtn. dtc. time: " + str(np.round(motion_detection_time*1000).astype(np.int)) + ' ms' + "\n"
        debug_txt = debug_txt + "Model images: " + str(img_count) + "\n"
        debug_txt = debug_txt + "Mdl. pred. time: " + str(np.round(model_pred_time*1000).astype(np.int)) + ' ms' + "\n"
        # if motion.is_stagnant:
        #     debug_txt = debug_txt + "Stagnant BB eliminated" + "\n"
        # else:
        #     debug_txt = debug_txt + "BB Ok" + "\n"

        debug_txt = debug_txt + "Frame rate = " + str(frame_rate) + ' fps' + "\n"

        # Add info on motion detection background state
        # debug_txt = debug_txt + "BG Updated = " + str(motion.updated) + "\n"
        # debug_txt = debug_txt + "BG Sum = " + str(motion.sum_thresh_bg) + "\n"
        # debug_txt = debug_txt + "BG Sum main = " + str(motion.sum_thresh_bg_main) + "\n"
        # Get loop delay


        # Finish off debug info
        dt = datetime.datetime.now() - t
        delay = np.round(dt.total_seconds()*1000).astype(np.int)

        debug_txt = debug_txt + "Loop time = " + str(delay) + ' ms'

        debug_file = open(stream_folder + "debug_info.txt", "w+")
        debug_file.write(debug_txt)
        debug_file.close()

        if DEBUG:
            dt = datetime.datetime.now() - t
            print("Debug text processing done: " + str(dt.total_seconds()))

        # Update figure
        if plot_mode_on:
            ax1.set_title(str(datetime.datetime.now()))

            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax1.text(10,
                     10,
                     debug_txt,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=props)

            ax1.text(frame_width - 400,
                     10,
                     label_txt,
                     fontsize=12,
                     verticalalignment='top',
                     bbox=props)

            plt.draw()
            plt.pause(0.002)
            plt.ioff()
            plt.show()

            if DEBUG:
                dt = datetime.datetime.now() - t
                print("Plotting done: " + str(dt.total_seconds()))

        # Inc loop count
        loop_count += 1
        # print("Line 608, end of loop")

except:

    print("SHIT!!!")
    raise

    e = sys.exc_info()[0]
    emailContent = "Error = " + str(e)
    e_mailer.sendmail(debug_email, "ELECTRIC BIRDER ERROR!", emailContent)
