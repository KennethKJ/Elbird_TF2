from tf_explain.core.grad_cam import GradCAM
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from matplotlib import pyplot as plt
from random import shuffle
import os
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

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

file_list = []
label_list = []
filename_classes = "C:/Users/alert/Google Drive/ML/Databases/Birds_dB/Mappings/minimal_bird_list.txt"
LIST_OF_CLASSES = [line.strip() for line in open(filename_classes, 'r')]

explainer = GradCAM()
explainer_occ = OcclusionSensitivity()

model = load_model("E:\\KerasOutput\\run_2019_11_25_07_33\\my_keras_model.h5")
model.summary()

testDir = "C:\\Users\\alert\\Google Drive\\ML\\Electric Bird Caster\\Classified\Carolina wren\\"

files = os.listdir(testDir)
shuffle(files)

# f = "2019-11-12_07-25-14_840.jpg"
for f in files:

    img_pil = image.load_img(path=testDir + f, target_size=(224, 224, 3))

    img = image.img_to_array(img_pil)

    im_np = preprocess_input(img)

    pred = model.predict(np.expand_dims(im_np, axis=0))
    prob = np.round(np.max(pred) * 100)
    ti = shortNamesList[np.argmax(pred)]

    grid1 = explainer.explain((np.expand_dims(im_np, axis=0), None), model, 'mixed10', np.argmax(pred))
    grid_occ = explainer_occ.explain((np.expand_dims(im_np, axis=0), None), model, np.argmax(pred), 75)

    fig = plt.figure(figsize=(18, 8))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img_pil)
    ax1.imshow(grid1, alpha=0.6)

    # ax2 = fig.add_subplot(1, 3, 2)
    # ax2.imshow(grid1, alpha=0.6)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img_pil)
    ax2.imshow(grid_occ, alpha=0.6)

    plt.title(ti + " " + str(prob) + "%")

    plt.show()
    print('Done')