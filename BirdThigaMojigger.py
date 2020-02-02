import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

train_path = "E:\\ML Training Data\\Keras\\train\\"
eval_path = "E:\\ML Training Data\\Keras\\eval\\"
test_path = "E:\\ML Training Data\\Keras\\test\\"

num_train_files = sum([len(files) for r, d, files in os.walk(train_path)])
num_eval_files = sum([len(files) for r, d, files in os.walk(eval_path)])

num_classes = 28
learning_rate = 0.0000075
batch_size = 16
num_epochs = 50
dropout_pc = 0.65
optimizer = 'Adam'
loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
selected_model = "MobileNetV2"
output_activation ='softmax'

train_from_beginning = True

if train_from_beginning:


    if selected_model=="InceptionV3":

        from tensorflow.keras.applications.inception_v3 import preprocess_input

        # create the base pre-trained model
        base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = BatchNormalization()(x)
        x = Dropout(dropout_pc)(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization()(x)
        x = Dropout(dropout_pc)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(dropout_pc)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(num_classes, activation=output_activation)(x)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in model.layers:
            layer.trainable = True
        # for layer in model.layers[65:]:
        #     layer.trainable = True

    elif selected_model == "VGG16":

        from tensorflow.keras.applications.vgg16 import preprocess_input

        base_model = tf.keras.applications.VGG16(weights='imagenet')

        x = base_model.get_layer('fc2').output
        x = BatchNormalization()(x)

        # x = Dropout(dropout_pc)(x)
        # x = Dense(1000, activation='relu')(x)
        # x = BatchNormalization()(x)

        x = Dropout(dropout_pc)(x)
        x = Dense(500, activation='relu')(x)
        x = BatchNormalization()(x)

        x = Dropout(dropout_pc)(x)
        predictions = Dense(num_classes, activation="softmax", name="sm_out")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in model.layers:
            layer.trainable = True
        # for layer in model.layers[-36:]:
        #     layer.trainable = True

    elif selected_model == "MobileNetV2":

        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        # create the base pre-trained model
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
        # base_model.summary()
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = BatchNormalization()(x)
        x = Dropout(dropout_pc)(x)
        # let's add a fully-connected layer
        # x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(dropout_pc)(x)
        # x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(dropout_pc)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(num_classes, activation=output_activation)(x)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in model.layers:
            layer.trainable = False
        for layer in model.layers[2:]:
            layer.trainable = True



    model.summary()

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # for layer in base_model.layers:
    #     layer.trainable = False



    # compile the model (should be done *after* setting layers to non-trainable)

    if optimizer == 'Adam':
        optim = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optim,
        loss=loss,
        metrics=[metrics.categorical_accuracy])



from scipy.ndimage.filters import gaussian_filter
from skimage.filters import gaussian as gau
from matplotlib import pyplot as plt
from PIL import Image

def my_preprocess(img):
    s = np.random.rand()*1.5
    img = gau(img, sigma=s, multichannel=True)
    img = preprocess_input(img)
    return img

# train the model on the new data for a few epochs
train_datagen = ImageDataGenerator(
    preprocessing_function=my_preprocess,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.4,
    width_shift_range=0.2,
    height_shift_range=0.2,
    # fill_mode='reflect',
    brightness_range=[0.2, 0.8],
    channel_shift_range=50,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')# train the model on the new data for a few epochs

# for a, b in train_generator:
#     print(a)
#     print(b)
#     plt.figure()
#     for i in range(batch_size):
#         plt.cla
#         immy = a[i, :, :, :]
#         immy = immy.astype(int)
#         # normy= ma.colors.Normalize(vmin=0, vmax=1.)
#         plt.imshow(immy)
#         plt.show()
#         # img = Image.fromarray(immy, 'RGB')
#         # img.show()
#         print(str(i))



labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())

from tensorflow.keras.applications.inception_v3 import preprocess_input

eval_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

eval_generator = eval_test_datagen.flow_from_directory(
    eval_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = eval_test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M")
    root_logdir = 'E:\\KerasOutput\\'
    full_path = os.path.join(root_logdir, run_id)
    os.mkdir(full_path,  mode=0o777)
    os.chdir(full_path)
    return full_path

run_logdir = get_run_logdir() + "_LR_" + str(learning_rate) + "_BS_" + str(batch_size)
# run_logdir = 'C:\\EstimatorOutput\\test\\'
tensorboard_cb = keras.callbacks.TensorBoard(
    run_logdir,
    #update_freq=1000,  # samples
    histogram_freq=1)

import time
log_id = time.strftime("log_run_%Y_%m_%d_%H_%M")

f= open("Run conditions" + log_id + ".txt", "w+")

f.write("Model = %s \n" % selected_model)

f.write("Learning rate = %1.20f \n" % learning_rate)
f.write("Batch size = %d \n" % batch_size)
f.write("Num epochs = %d \n" % num_epochs)
f.write("Dropout proportion = %1.2f \n" % dropout_pc)
f.write("Optimizer = %s \n" % optimizer)
f.write("Loss = %s \n" % loss)
f.write("Output activation function = %s \n" % output_activation)

f.close()

tensorboard_cb = keras.callbacks.TensorBoard(run_logdir, histogram_freq=1)

if not train_from_beginning:
    saved_model = "E:\\KerasOutput\\run_2019_12_06_12_11\\my_keras_model.h5"
    model = load_model(saved_model)

if optimizer == 'Adam':
    optim = keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(
    optimizer=optim,
    loss=loss,
    metrics=[metrics.categorical_accuracy])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=round(num_train_files/batch_size),
    epochs=num_epochs,
    validation_data=eval_generator,
    validation_steps=100,
    callbacks=[checkpoint_cb, tensorboard_cb]
)

model.save(run_logdir + "\\" + log_id + '.h5')
print("The End")
