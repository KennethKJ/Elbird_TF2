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
import subprocess


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

src = 0
cap = cv2.VideoCapture(src)

CV_CAP_PROP_FRAME_WIDTH = 3
CV_CAP_PROP_FRAME_HEIGHT = 4

cap.set(CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

command = ['ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s','640x480',
            '-i','-',
            '-ar', '44100',
            '-ac', '2',
            '-acodec', 'pcm_s16le',
            '-f', 's16le',
            '-ac', '2',
            '-i','/dev/zero',
            '-acodec','aac',
            '-ab','128k',
            '-strict','experimental',
            '-vcodec','h264',
            '-pix_fmt','yuv420p',
            '-g', '50',
            '-vb','1000k',
            '-profile:v', 'baseline',
            '-preset', 'ultrafast',
            '-r', '30',
            '-f', 'flv',
            'rtmp://a.rtmp.youtube.com/live2/67qv-p2wf-uqd0-7qx6']

# pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
#
# while True:
#     _, frame = cap.read()
#     pipe.stdin.write(frame.tostring())
#
# pipe.kill()
# cap.release()

print("Starting loop")
while 1 == 1:
    # Capture frame-by-frame
    ret, frame_BGR = cap.read()
    frame = cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2RGB)
    # pipe.stdin.write(frame.tostring())
    
    dim = (224, 224)
    frame_rsz = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    frame_ML = np.expand_dims(frame_rsz, axis=0)
    frame_ML_preproc = preprocess_input(frame_ML)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run predictions
    pred = model.predict(frame_ML_preproc)
    bird_idx = np.argmax(pred, axis=1)

    # # Update prediction history
    # if len(pred_history[:]) > 10:
    #     pred_history = np.delete(pred_history, 0, 0)
    # pred_history = np.concatenate((pred_history, np.expand_dims(pred, axis=0)))

    # instant_prob = pred[bird_idx]*100

    # Calculate classification probability as average of last 3 images
    # prob = np.round(np.mean(pred_history[-3:, [0, 1, 2], bird_idx], axis=0) * 100)

    # prob = pred[[0, 1, 2], bird_idx]

    frame_BGR = cv2.putText(frame_BGR,  pretty_names_list[bird_idx[0]], (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imshow('frame', frame_BGR)
    cv2.waitKey(1)

    doPlot = False
    if doPlot:

        fig = plt.figure(figsize=(18, 8))

        ax1 = fig.add_subplot(1, 3, 1)
        # P = sub_img1
        # P = (sub_img1 + 1) / 2
        ax1.imshow(frame_rsz)
        # ax1.text(5, 10, pretty_names_list[bird_idx[0]] + " (" + str(prob[0]) + ")", fontsize=16)
        ax1.text(5, 10, pretty_names_list[bird_idx[0]], fontsize=16)
        #
        # ax2 = fig.add_subplot(1, 3, 2)
        # ax2.imshow((sub_img2 + 1) / 2)
        # ax2.text(5, 10, pretty_names_list[bird_idx[1]] + " (" + str(prob[1]) + ")", fontsize=16)
        #
        # ax3 = fig.add_subplot(1, 3, 3)
        # ax3.imshow((sub_img3 + 1) / 2)
        # ax3.text(5, 10, pretty_names_list[bird_idx[2]] + " (" + str(prob[2]) + ")", fontsize=16)

        plt.show()

        plt.close(fig)

        # # Display the resulting frame
    # cv2.imshow('frame', gray)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

        # When everything done, release the capture
        #
        # right_now = gimme_minute()
        # if ref_minute != right_now:
        #     ref_minute = gimme_minute()
        #     bird_history[:, 1:] = bird_history[:, 0:-1]
        #     bird_history[:, 0] = np.squeeze(current_bird_count)
        #     current_bird_count = np.zeros([len(top_10), 1])
        #
        #     # Update figure
        #     plot_IDhistory(bird_history, pretty_names_list)
        #
        # try:
        #     # img_pil = image.load_img(path=captures_folder + f)
        #     img_pil = image.load_img(path=captures_folder + f)
        #     w, h = img_pil.size
        #     img_pil = img_pil.resize((int(w/3), int(h/3)))
        #     # img_pil.show()
        #
        #     # im1 = im.crop((left, top, right, bottom))
        #     img = image.img_to_array(img_pil)
        #
        #     H_anchor = 2 * 224 - 110 + 112 + 50 + 100
        #     V_anchor = int(112-75+112+75-25)
        #     move_factor_horizontal = H_anchor
        #     move_factor_vertical = V_anchor
        #     sub_img1 = img[0+move_factor_vertical:224+move_factor_vertical, 0+move_factor_horizontal:224+move_factor_horizontal, :]
        #
        #     img = image.img_to_array(img_pil)
        #
        #     move_factor_horizontal = H_anchor + 112
        #     move_factor_vertical = V_anchor
        #     sub_img2 = img[0+move_factor_vertical:224+move_factor_vertical, 0+move_factor_horizontal:224+move_factor_horizontal, :]
        #
        #     img = image.img_to_array(img_pil)
        #
        #     move_factor_horizontal = H_anchor + 224
        #     move_factor_vertical = V_anchor
        #     sub_img3 = img[0+move_factor_vertical:224+move_factor_vertical, 0+move_factor_horizontal:224+move_factor_horizontal, :]
        #
        #     im_np1 = preprocess_input(sub_img1)
        #     im_np2 = preprocess_input(sub_img2)
        #     im_np3 = preprocess_input(sub_img3)
        #
        #     im_np1 = np.expand_dims(im_np1, axis=0)
        #     im_np2 = np.expand_dims(im_np2, axis=0)
        #     im_np3 = np.expand_dims(im_np3, axis=0)
        #
        #     im_final = np.concatenate((im_np1, im_np2, im_np3))
        #
        #     # Run predictions
        #     pred = model.predict(im_final)
        #     bird_idx = np.argmax(pred, axis=1)
        #
        #     # Update prediction history
        #     if len(pred_history[:]) > 10:
        #         pred_history = np.delete(pred_history, 0, 0)
        #     pred_history = np.concatenate((pred_history, np.expand_dims(pred, axis=0)))
        #
        #     # instant_prob = pred[bird_idx]*100
        #
        #     # Calculate classification probability as average of last 3 images
        #     prob = np.round(np.mean(pred_history[-3:, [0, 1, 2], bird_idx], axis=0) * 100)
        #
        #     # prob = pred[[0, 1, 2], bird_idx]
        #
        #     doPlot = False
        #     if doPlot:
        #         fig = plt.figure(figsize=(18, 8))
        #
        #         ax1 = fig.add_subplot(1, 3, 1)
        #         P = sub_img1
        #         P = (sub_img1+1)/2
        #         ax1.imshow(P)
        #         ax1.text(5, 10, pretty_names_list[bird_idx[0]] + " (" + str(prob[0]) + ")", fontsize=16)
        #
        #         ax2 = fig.add_subplot(1, 3, 2)
        #         ax2.imshow((sub_img2+1)/2)
        #         ax2.text(5, 10, pretty_names_list[bird_idx[1]] + " (" + str(prob[1]) + ")", fontsize=16)
        #
        #         ax3 = fig.add_subplot(1, 3, 3)
        #         ax3.imshow((sub_img3+1)/2)
        #         ax3.text(5, 10, pretty_names_list[bird_idx[2]] + " (" + str(prob[2]) + ")", fontsize=16)
        #
        #         plt.show()
        #         plt.close(fig)
        #
        #
        #     sections = []
        #     birds = []
        #     for x in range(3):
        #         if (bird_idx[x] != 3) and (prob[x] > detection_threshold):
        #             sections.append(x)
        #             birds.append(bird_idx[x])
        #
        #
        #
        #     if len(sections) > 0:  #  If no bird is detetion in any section, just skip all
        #         del_idx = []
        #         if len(sections) > 1:
        #             for j in range(len(sections)-1):
        #                 if birds[j] == birds[j+1]:
        #                     if prob[sections[j]] >= prob[sections[j+1]]:
        #                         del_idx.append(j+1)
        #                     else:
        #                         del_idx.append(j)
        #             sections = np.delete(sections, del_idx)
        #
        #
        #         #
        #         # if len(sections) > 1:
        #         #     for j in range(len(sections)-1):
        #         #         if birds[j] == birds[j+1]:
        #         #             if prob[sections[j]] >= prob[sections[j+1]]:
        #         #                 sections = np.delete(sections, j+1)
        #         #             else:
        #         #                 sections = np.delete(sections, j)
        #
        #         # print("Instant prob: " + pretty_names_list[bird_idx] + " (" + str(int(instant_prob)) + "%)")
        #
        #         current_class_file = open(stream_folder + "Current_Classification.txt", "w+")
        #
        #         # Get time stamp
        #         timmy = str(datetime.datetime.now())
        #         timmy = timmy[11:16]
        #
        #         img = image.img_to_array(img_pil)  # Band aid for now
        #
        #         move_factor_horizontal = H_anchor
        #         move_factor_vertical = V_anchor
        #         published_img = img[move_factor_vertical: 224 + move_factor_vertical,
        #                         move_factor_horizontal: 2 * 224 + move_factor_horizontal, :]
        #         # Cycle through all
        #         for i in range(len(sections)):
        #
        #             if (prob[sections[i]] > detection_threshold):
        #                 # Write line to label files used in OBS
        #                 current_class_file.write("Sec" + str(i+1) + pretty_names_list[birds[i]] + " (" + timmy + "; " + str(int(prob[i])) + "%) \n")
        #
        #                 # Add to counter
        #                 current_bird_count[birds[i]] += 1
        #
        #                 # Construct destination directory and copy classified image to folder
        #                 destination_dir = dump_classified_imp_folder + \
        #                                   "Classified" + "\\" + \
        #                                   pretty_names_list[birds[i]]
        #
        #                 # Check if it already exists
        #                 if not os.path.isdir(destination_dir):
        #                     os.makedirs(destination_dir)
        #
        #                 # Copy current picture to the destination
        #                 shutil.copyfile(captures_folder + f, destination_dir + "\\" + f)
        #
        #                 # else:
        #                 #     # Write line to label files used in OBS
        #                 #     current_class_file.write("Not sure" + " (" + timmy + ";" + str(int(prob[i])) + "%) \n")
        #
        #
        #                 # print(pretty_names_list[bird_idx] + " (" + timmy + "; " + str(int(prob)) + "%)")
        #
        #                 published_img[:, sections[i] * 112 :sections[i] * 112 + 2, :] = 0
        #                 published_img[:, sections[i] * 112 :sections[i] * 112 + 2, sections[i]] = 255
        #
        #                 published_img[:, sections[i] * 112 + 224 -2 : sections[i] * 112 + 224, :] = 0
        #                 published_img[:, sections[i] * 112 + 224 -2 : sections[i] * 112 + 224, sections[i]] = 255
        #
        #                 published_img[0:2, sections[i] * 112:sections[i] * 112 + 224, :] = 0
        #                 published_img[0:2, sections[i] * 112:sections[i] * 112 + 224, sections[i]] = 255
        #
        #                 published_img[-2:, sections[i] * 112:sections[i] * 112 +224, :] = 0
        #                 published_img[-2:, sections[i] * 112:sections[i] * 112 + 224, sections[i]] = 255
        #
        #
        #         fig = plt.figure(figsize=(12, 6))
        #
        #         ax = plt.Axes(fig, [0., 0., 1., 1.])
        #         ax.set_axis_off()
        #         fig.add_axes(ax)
        #         plt.imshow(published_img.astype(int))
        #         for i in range(len(sections)):
        #             ax.text(sections[i] *112+5, sections[i]*12 + 12, pretty_names_list[birds[i]] + " (" + str(prob[sections[i]]) + ")", fontsize=24)
        #         plt.savefig(stream_folder + "The View.png")
        #         # plt.show()
        #         plt.close(fig)
        #
        #         current_class_file.close()
        #
        #     if not develop:
        #         os.remove(captures_folder + f)

#         except:
#             print("Error reading file!")
#             time.sleep(0.5)  #
#
# else:
#     # If no files were found in the capture folder this round
#     # print("No files found")
#     time.sleep(1)  #

cap.release()
cv2.destroyAllWindows()
