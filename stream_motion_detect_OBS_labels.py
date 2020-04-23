from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import datetime
import pandas as pd
import cv2
from scipy.spatial import distance
from pyimagesearch.motion_detection import singlemotiondetector as smd

print("Running Electric Birder")

## Settings

# Misc
doNN = True
doAVI = False
minimum_prob = 50
ID_stay_time = 5  # Cycles before a positive ID has faded away in the species ID panel
num_cycles_in_history = 3  # Number of cycles in history
max_dist = 7  # max allowed distance when looking for classification clusters of same species

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
step_size = (int(win_size[0]/3), int(win_size[1]/3))

# Load model
print('Neural network starting up, please wait ... ' + "\n")
label_file = open(stream_folder + "label.txt", "w+")
label_file.write('Neural network starting up, please wait ... ' + "\n")
label_file.close()
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
    'Brown headed cowbird (F)',
    'brown headed cowbird (M)',
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
    'Squirrel :o',  # 25
    'Tufted titmouse',
    'White-breasted nuthatch']  # 27

print("Initializing camera")
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

print("Video source: " + ss)

print("reading frame ... ")
ret, frame = cap.read()
if frame is None:
    print('Not able to grab images from IP cam!')
    pass
print("Camera initialized")

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Initialize motion detector
motion = smd.SingleMotionDetector()
motion.update_bg(frame_bw)
motion.update_bg_main(frame_bw)

# Get image size and calc windowing related parameters
h_frame, w_frame, _ = frame.shape  # img_pil = img_pil.resize((int(w/3), int(h/3)))
raw_image_shape = (h_frame, w_frame)  #  (height, width)
num_steps = (int(np.floor((raw_image_shape[0]/step_size[0]) - 1)),
             int(np.floor((raw_image_shape[1]/step_size[1]) - 1)))

# Initialize vars related to classification
pred_probs_2D_history = np.zeros((0, num_steps[0], num_steps[1]))
bird_classes_2D_history = np.zeros((0, num_steps[0], num_steps[1])).astype(np.int)

the_situation_idx = np.zeros((num_steps[0], num_steps[1])).astype(np.int)
the_situation_prob = np.zeros((num_steps[0], num_steps[1]))

# Below var is set to ID_stay_time when a positive ID is made and then "fades" with 1 per cycle.
# Works as a "low pass filter" to make the displaying of labels less flimsy
birds_seen_lately = np.zeros(len(pretty_names_list), )

# Initialize data frame
df = pd.DataFrame(columns=['year',
                           'month',
                           'day',
                           'hour',
                           'minute',
                           'second',
                           'birdID',
                           'bird_name',
                           'classification_probability',
                           'loop_cycle',
                           'horz_location',
                           'vert_location',
                           'image_filename'])

# Initializing loop variables
current_hour = datetime.datetime.now().hour
last_loop_count = -1
nuthins_seen = 0
restart_no = 0
loop_count = 0
frame_rate = 6

print("Starting loop")
while 1 == 1:

    # Update t for next loop
    t = datetime.datetime.now()

    # Reset loop variables
    img_count = -1
    model_pred_time = 0

    # Skip over old frames
    grab_delay = 0
    num_frames = 0
    while grab_delay < 0.1:
        T = datetime.datetime.now()
        ret = cap.grab()
        dT = datetime.datetime.now() - T
        grab_delay = dT.total_seconds()
        # print(delay_grab)
        num_frames += 1
        if num_frames > 500:
            grab_delay = 1
            print("Looks like it's caught indefinitely in frame reading loop. Skipping it! " + T.strftime())

    # Then get most recent frame
    ret, frame = cap.read()
    num_frames += 1

    # Retry connecting to capture device if frame was none
    if frame is None:

        print('Frame was None ' + str(datetime.datetime.now()))

        wait_time = 3
        IP = IP_start - 1
        while frame is None:

            # Let go of capture object
            cap.release()

            # Inform
            label_file = open(stream_folder + "label.txt", "w+")
            label_file.write('< Camera connection issues > \n < Retrying to connect in ' + str(wait_time) + ' seconds >')
            label_file.close()

            print('Trying again in ' + str(wait_time) + ' secs ...')
            time.sleep(wait_time)

            # Increase last 3 IP numbers
            IP += 1
            if IP > 105:
                IP = int(100)

            # Generate new string
            IP_address = "192.168.10." + str(IP)
            ss = "rtsp://" + username + ":" + password + "@" + IP_address + \
                 ":554/cam/realmonitor?channel=" + channel + "&subtype=" + subtype + "&unicast=true&proto=Onvif"

            # Inform
            print('Retrying with IP = ' + str(IP))
            label_file = open(stream_folder + "label.txt", "w+")
            label_file.write('< Retrying camera connection with IP = ' + str(IP) + ' >')
            label_file.close()

            # Try again
            cap = cv2.VideoCapture(ss)
            ret, frame = cap.read()

        else:
            restart_no += 1

            # Inform
            label_file = open(stream_folder + "label.txt", "w+")
            label_file.write('< Connection re-established >')
            label_file.close()
            print('Connection re-established ' + str(datetime.datetime.now()))

    try:
        # Change color format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except:
        frame = None
        print("I think there was no frame. Internet might be out. Waiting 10 secs and trying again")
        time.sleep(10)
        continue

    ## Motion detection
    # Convert to black and white
    frame_bw = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect motion
    # mt = datetime.datetime.now()
    X = motion.detect(frame_bw)
    # dmt = datetime.datetime.now() - mt
    # print("Motion detection time: " + str(dmt.total_seconds()))

    if X is not None:  # Motion is detected

        # Unpack group of bounding boxes
        _, bounding_boxes = X


        if bounding_boxes != [] and doNN:

            # Initialize image container
            all_image_snippets = np.zeros((0, target_img_size[0], target_img_size[1], 3)).astype(np.int)

            # Initialize movement indicator grid
            grid = np.zeros((num_steps[0], num_steps[1])).astype(np.bool)

            for bb in bounding_boxes:

                # Unpack the single bounding box
                x, y, w, h = bb

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
                mt = datetime.datetime.now()
                pred = model.predict(all_image_snippets)
                dmt = datetime.datetime.now() - mt
                model_pred_time=dmt.total_seconds()

                # Get index of classified birds/animals
                bird_classes = np.argmax(pred, axis=1)

                # Translate class identifications into a matrix ("2D")
                bird_classes_2D = np.zeros(num_steps).astype(np.int)  # reset
                bird_classes_2D[y_s, x_s] = bird_classes

                # Get rid of background detections
                bird_classes_2D[bird_classes_2D == 3] = 0

                # Get maxima of props and convert to %
                pred_probs = np.max(pred, axis=1)*100
                # Translate probabilities into a matrix ("2D")
                pred_probs_2D = np.zeros(num_steps).astype(np.int)  # reset
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
                            if len(the_situation) > 5 and cluster_prop > 70: # HIT! (more than 5 detections and over 70% mean certainty)

                                # Apply higher restrictions on troublesome classes
                                if (c == 5 or c == 18 or c == 22 or c == 24 or c == 24 or c == 26 or c==27) and cluster_prop < 90:  # Stricter rule for blue jay and others due to many false alarms
                                    continue

                                # recategorizing of "off duty" Starlings
                                if c == 13:
                                    c = 12

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

    # Write labels to file for OBS
    if np.sum(birds_seen_lately) > 0:

        label_file = open(stream_folder + "label.txt", "w+")

        for i, c in enumerate(birds_seen_lately):

            # Subtract one to "forget" some over each cycle
            if birds_seen_lately[i] > 0:
                birds_seen_lately[i] -= 1

            # Add too labels if active
            if c:
                label_file.write(pretty_names_list[i] + "\n")
                nuthins_seen = 0

        label_file.close()

    elif nuthins_seen == 0:  # When no detections has been made

        label_file = open(stream_folder + "label.txt", "w+")
        label_file.write('< --- >' + "\n")
        label_file.close()

        nuthins_seen = 1  # This switch is to ensure to write this ti label file only once if a strech of nothing is going on at the feeder

    # Save dataframe to csv every 5 minutes
    if loop_count % 300 == 0:

        df_filename = str(datetime.datetime.today().year) + '_' + \
                      str(datetime.datetime.today().month) + '_' + \
                      str(datetime.datetime.today().day) + '_' + \
                      str(current_hour) + '.csv'

        df.to_csv(r'E:\\Electric Bird Caster\\Data\\' + df_filename, index=False)

    # Save data frame and create a new if a new hour of the day has started
    if current_hour != datetime.datetime.now().hour:

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
                                   'classification_probability',
                                   'loop_cycle',
                                   'horz_location',
                                   'vert_location',
                                   'image_filename'])

        # Update current hour
        current_hour = datetime.datetime.today().hour

    ## Update debug info
    debug_txt = ""  # Reset debug info text
    debug_txt = debug_txt + "LOOP no. " + str(loop_count) + "\n"
    debug_txt = debug_txt + "Num frames run through: " + str(num_frames) + "\n"
    debug_txt = debug_txt + "Total images for model: " + str(img_count) + "\n"
    debug_txt = debug_txt + "Model prediction time: " + str(model_pred_time) + "\n"

    # Get loop delay
    dt = datetime.datetime.now() - t
    delay = dt.total_seconds()
    # print("Loop time = " + str(delay))
    debug_txt = debug_txt + "Loop time = " + str(delay) + "\n"

    debug_file = open(stream_folder + "debug_info.txt", "w+")
    debug_file.write(debug_txt)
    debug_file.close()

    # Inc loop count
    loop_count += 1
