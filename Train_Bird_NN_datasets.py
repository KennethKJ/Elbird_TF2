import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import preprocess_input

from tensorflow.keras.mixed_precision import experimental as mixed_precision

import os
import numpy as np
import pandas as pd
from skimage.filters import gaussian as gau
import time
from get_dataset import get_dataset

import winsound

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

print("Starting training session :) ")

# train_path = "E:\\ML Training Data\\Keras\\train\\"
# eval_path = "E:\\ML Training Data\\Keras\\eval\\"
# test_path = "E:\\ML Training Data\\Keras\\test\\"

# num_train_files = sum([len(files) for r, d, files in os.walk(train_path)])
# num_eval_files = sum([len(files) for r, d, files in os.walk(eval_path)])

apply_mixed_precicision = False
if apply_mixed_precicision:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)

# SET PARAMETERS
keras_root_output_folder = "E:\\KerasOutput\\"
training_folder_name = 'July_11th_2020_2'
saved_model_filename = "saved model.h5"

# if os.path.isfile(keras_root_output_folder + training_folder_name + "\\" + saved_model_filename):
# train_from_beginning = False
# else:
train_from_beginning = False

params = {}
params['num_epochs'] = 100
params['initial_epoch'] = 90
params['steps_per_epoch'] = 1000
params['validation_steps'] = 100

params['learning_rate'] = 0.0000001
params['dropout_pc_hidden'] = 0.6
params['dropout_pc_FC'] = 0.6

params['batch_size'] = 16*2*2


params['buffer_size'] = 10000  # int(params['steps_per_epoch'] * params['batch_size'])
print("Buffer size = " + str(params['buffer_size']))

params['num_parallel_calls'] = 12  # tf.data.experimental.AUTOTUNE
params['batch_prefetch'] = 12  # tf.data.experimental.AUTOTUNE
params['num_classes'] = 56

params['model_input_key'] = "input_1"
params['selected_model'] = "MobileNetV2"

params['image_size'] = [224, 224, 3]

# Augmentation parameters
params['max_delta'] = 63.0 / 2 / 255.0  # Image brightness augmentation parameter
params['contrast_lower'] = 0.2
params['contrast_higher'] = 1.8 / 2
params['image_max_rotation'] = 25
params['image_crop_margin'] = 50



optimizer = 'Adam'
loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
selected_model = "MobileNetV2"
output_activation ='softmax'


# SELECT MODEL

if train_from_beginning:

    if params['selected_model'] == "InceptionV3":

        from tensorflow.keras.applications.inception_v3 import preprocess_input

        # create the base pre-trained model
        base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = BatchNormalization()(x)
        x = Dropout(params['dropout_pc_FC'])(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization()(x)
        x = Dropout(params['dropout_pc_FC'])(x)
        # x = Dense(1024, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dropout(params['dropout_pc_FC'])(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(params['num_classes'], activation=output_activation)(x)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in model.layers:
            layer.trainable = True
        # for layer in model.layers[65:]:
        #     layer.trainable = True

    elif params['selected_model'] == "VGG16":

        from tensorflow.keras.applications.vgg16 import preprocess_input

        base_model = tf.keras.applications.VGG16(weights='imagenet')

        x = base_model.get_layer('fc2').output
        x = BatchNormalization()(x)

        # x = Dropout(params['dropout_pc_FC'])(x)
        # x = Dense(1000, activation='relu')(x)
        # x = BatchNormalization()(x)

        x = Dropout(params['dropout_pc_FC'])(x)
        x = Dense(500, activation='relu')(x)
        x = BatchNormalization()(x)

        x = Dropout(params['dropout_pc_FC'])(x)
        predictions = Dense(params['num_classes'], activation="softmax", name="sm_out")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in model.layers:
            layer.trainable = True
        # for layer in model.layers[-36:]:
        #     layer.trainable = True

    elif params['selected_model'] == "MobileNetV2":

        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        # create the base pre-trained model
        base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
        # base_model.summary()
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # x = BatchNormalization()(x)
        x = Dropout(params['dropout_pc_FC'])(x)
        # let's add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout_pc_FC'])(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(params['dropout_pc_FC'])(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(params['num_classes'], activation=output_activation)(x)
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        # for layer in model.layers:
        #     layer.trainable = False
        for layer in model.layers:  # [2:]:
            layer.trainable = True

else:

    # New training with the Maculay images included
    saved_model_file = keras_root_output_folder + training_folder_name + "\\saved model.h5"
    # saved_model = "E:\\KerasOutput\\run_2019_12_06_12_11\\my_keras_model.h5"

    print("Loading saved model from: " + saved_model_file)
    model = load_model(saved_model_file)

# Display a summary of the model
model.summary()

# COMPILE
# (should be done *after* setting layers to non-trainable)
if optimizer == 'Adam':
    optim = keras.optimizers.Adam(learning_rate=params['learning_rate'])

model.compile(
    optimizer=optim,
    loss=loss,
    metrics=[metrics.categorical_accuracy])


# DATASET RETRIEVAL & PROCESSING

# Define preprocessing fn (happens after augmentation)
def my_preprocess(img):

    # Add blur
    s = np.random.rand()*1.5
    img = gau(img, sigma=s, multichannel=True)

    # Apply Keras pre-defined preprocessing fn
    img = preprocess_input(img)
    return img


# DEFINE CALLBACKS
run_logdir = keras_root_output_folder + training_folder_name + "\\"
if not os.path.isdir(run_logdir):
    os.mkdir(run_logdir, mode=0o777)
os.chdir(run_logdir)
# run_logdir = 'C:\\EstimatorOutput\\test\\'

tensorboard_cb = keras.callbacks.TensorBoard(
    run_logdir,
    profile_batch='500,510',
    update_freq=500,
    write_images=False,
    embeddings_freq=1,
    histogram_freq=1)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    run_logdir + saved_model_filename,
    save_best_only=True,
    save_freq='epoch')

csv_logger = keras.callbacks.CSVLogger(
    'train log.csv',
    append=True,
    separator=';')


# WRITE LOG FILE FOR CURRENT RUN
log_id = time.strftime("log_run_%Y_%m_%d_%H_%M")
f = open(run_logdir + log_id + ".txt", "w+")
f.write("Model = %s \n" % selected_model)
f.write("Learning rate = %1.20f \n" % params['learning_rate'])
f.write("Batch size = %d \n" % params['batch_size'])
f.write("Num epochs = %d \n" % params['num_epochs'])
f.write("Dropout proportion = %1.2f \n" % params['dropout_pc_FC'])
f.write("Optimizer = %s \n" % optimizer)
f.write("Loss = %s \n" % loss)
f.write("Output activation function = %s \n" % output_activation)
f.close()


# COMPILE
if optimizer == 'Adam':
    optim = keras.optimizers.Adam(learning_rate=params['learning_rate'])

model.compile(
    optimizer=optim,
    loss=loss,
    metrics=[metrics.categorical_accuracy])

# # FIT
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=1000,
#     epochs=num_epochs,
#     validation_data=eval_generator,
#     validation_steps=100,
#     callbacks=[checkpoint_cb, tensorboard_cb]
# )

# os.chdir('E:\\Virtual Envs\\TF2.2\\Scripts\\')
# os.system('tensorboard --logdir ' + '"' + keras_root_output_folder + training_folder_name + '"')

# FIT
history = model.fit(
    get_dataset("E:\\ML Training Data\\Train CSVs\\train.csv", params=params),
    steps_per_epoch=params['steps_per_epoch'],
    epochs=params['num_epochs'],
    initial_epoch=params['initial_epoch'],
    validation_data=get_dataset("E:\\ML Training Data\\Train CSVs\\eval.csv", params=params),
    validation_steps=params['validation_steps'],
    # workers=2,
    # use_multiprocessing=True,
    # max_queue_size=2,
    callbacks=[checkpoint_cb, tensorboard_cb, csv_logger]
)


# SAVE MODEL AT END
model.save(run_logdir + "saved model at end of training session.h5")

try:
    pd.DataFrame(history.history).to_csv("history.csv")
except:
    print('Panda thing didnt work')

frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

print("The End")

# Notes
# at epoch 13 for June 24th run