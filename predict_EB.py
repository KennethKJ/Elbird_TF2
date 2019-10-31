from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.inception_v3 import preprocess_input
from matplotlib import pyplot as plt
import os
import skimage
from tensorflow.keras.preprocessing import image
from random import shuffle
# model.summary()
model = load_model("E:\\KerasOutput\\run_2019_10_26_21_58\\my_keras_model.h5")

doImageGen = False

if doImageGen:
    eval_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    # testDir = 'C:/Users/alert/Google Drive/ML/Databases/Birds_dB/Keras2/test/'
    testDir = "E:\\ML Training Data\\Keras\\test\\"
              # 'C:\\Users\\alert\\Google Drive\\ML\\Databases\\Photo Booth User Group Photos\\'
    # test_path = "E:\\ML Training Data\\Keras\\eval\\"

    label_list = []
    filename_classes = "C:/Users/alert/Google Drive/ML/Databases/Birds_dB/Mappings/minimal_bird_list.txt"
    LIST_OF_CLASSES = [line.strip() for line in open(filename_classes, 'r')]

    test_generator = eval_test_datagen.flow_from_directory(
        testDir,
        target_size=(224, 224),
        batch_size=1,
        class_mode='categorical')

    labels = (test_generator.class_indices)
    labels = dict((v,k) for k,v in labels.items())

    count = 0
    filepaths = test_generator.filepaths
    for img, label in test_generator:

        pred = model.predict(img)
        c = LIST_OF_CLASSES[np.argmax(pred)]

        im = Image.open(test_generator.filepaths[test_generator.index_array[count]])
        plt.figure()
        # plt.subplot(211)
        # plt.imshow(np.squeeze(img))
        # # plt.title(c)
        # plt.subplot(212)
        plt.imshow(np.squeeze(im))

        prob = np.round(np.max(pred)*100)
        if np.max(pred) > 0.50:
            ti = LIST_OF_CLASSES[np.argmax(pred)] + " (" + str(prob) + "%)"
        else:
            ti = "hmm, not sure, ... " + LIST_OF_CLASSES[np.argmax(pred)] + " maybe? (" + str(prob) + "%)"

        # plt.title("Real: " + LIST_OF_CLASSES[np.argmax(label)] + "; Pred: " + c)
        plt.title(ti)

        plt.show()
        print(c)
        # #
        # plt.imshow(np.squeeze(img))
        # plt.title(c)
        # plt.show()
        # print(c)
        count += 1
else:
    testDir = 'C:\\Users\\alert\\Google Drive\\ML\\Databases\\Photo Booth User Group Photos\\(2) Bird Photo Booth Users Group_files\\'
    # testDir = 'C:\\Users\\alert\Google Drive\\ML\\Databases\\Birds_dB\\Keras2\\test\\blue_jay\\'
    # testDir = "C:\\Users\\alert\\Google Drive\\ML\\Databases\\Own Feeder BirdPics\\Demo\\"
    # testDir = "E:\\ML Training Data\\(2) Bird Photo Booth Users Group_files\\_UNSORTED\\"

    file_list = []
    label_list = []
    filename_classes = "C:/Users/alert/Google Drive/ML/Databases/Birds_dB/Mappings/minimal_bird_list.txt"
    LIST_OF_CLASSES = [line.strip() for line in open(filename_classes, 'r')]


    shortNamesList = [
        'Crow',
        'Goldfinch breeding M',
        'Goldfinch off duty M or F',
        'Do not see a bird here :/',
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
        'Caridal M',
        'Cardinal F',
        'Norhtern flicker (red)',
        'Pileated woodpecker',
        'Red winged blackbird F/Im',
        'Red winged blackbird M',
        'Squirrel!',
        'Tufted titmouse',
        'White breasted nuthatch']

    files = os.listdir(testDir)
    shuffle(files)

    # eval_test_datagen = ImageDataGenerator(
    #     preprocessing_function=preprocess_input)



    for f in files:
        img_pil = image.load_img(path=testDir + f, target_size=(224, 224, 3))
        img = image.img_to_array(img_pil)

        im_np = preprocess_input(img)

        pred = model.predict(np.expand_dims(im_np,axis=0))


        plt.figure(1)
        # plt.subplot(211)
        x = np.arange(0, 28)
        y = np.squeeze(pred*100)

        plt.bar(x, y)
        plt.xticks(x, shortNamesList, rotation=90)
        plt.xlim(-1, 29)
        plt.ylim(-1, 101)
        plt.ylabel("Classification certainty (%)")
        plt.xlabel("Class name")
        plt.title("Classification overview")


        plt.figure(2)

        prob = np.round(np.max(pred)*100)
        if np.max(pred) > 0.5:
            ti = shortNamesList[np.argmax(pred)] + " (" + str(prob) + "%)"
        else:
            ti = "hmm, not sure, ... " + shortNamesList[np.argmax(pred)] + " maybe? (" + str(prob) + "%)"



        plt.title(ti)

        # plt.subplot(212)
        plt.imshow(img_pil)
        # plt.title(LIST_OF_CLASSES[np.argmax(pred)])
        plt.show()
        print(LIST_OF_CLASSES[np.argmax(pred)])

        print("")
print("The End")