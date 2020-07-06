import os
import random
import shutil

GD_path = ""
DB_path = "E:\\ML Training Data\\(2) Bird Photo Booth Users Group_files\\"
random.seed(1)

main_Keras_Dir = "E:\\ML Training Data\\Keras\\"
if not(os.path.isdir(main_Keras_Dir)):
    os.mkdir(main_Keras_Dir)

train_eval_test_proportions = [0.7, 0.2, 0.1]
train_eval_test_folders = ["train", "eval", "test"]

file_list = []
label_list = []
filename_classes = "C:/Users/alert/Google Drive/ML/Databases/Birds_dB/Mappings/minimal_bird_list.txt"
LIST_OF_CLASSES = [line.strip() for line in open(filename_classes, 'r')]

for folder in os.listdir(GD_path + DB_path):
    if folder in LIST_OF_CLASSES:
        files = os.listdir(GD_path + DB_path + folder)
        random.shuffle(files)
        currentIdx = 0
        for i in range(3):
            currentPath = main_Keras_Dir + train_eval_test_folders[i]
            if not (os.path.isdir(currentPath)):
                os.mkdir(currentPath)
            currentPath += "/" + folder
            if not os.path.isdir((currentPath)):
                os.mkdir(currentPath)

            startIdx = currentIdx
            num_files = round(train_eval_test_proportions[i] * len(files))
            endIdx = currentIdx + num_files
            currentFiles = files[startIdx:endIdx]
            currentIdx += num_files

            for file_name in currentFiles:
                full_file_name = GD_path + DB_path + folder + "/" + file_name
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, currentPath)

            print("DONE!")
print('The End')