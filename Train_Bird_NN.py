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

import os
import numpy as np
import pandas as pd
from skimage.filters import gaussian as gau
import time
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# train_path = "E:\\ML Training Data\\Keras\\train\\"
# eval_path = "E:\\ML Training Data\\Keras\\eval\\"
# test_path = "E:\\ML Training Data\\Keras\\test\\"

# num_train_files = sum([len(files) for r, d, files in os.walk(train_path)])
# num_eval_files = sum([len(files) for r, d, files in os.walk(eval_path)])

# SET PARAMETERS
keras_root_output_folder = "E:\\KerasOutput\\"
training_folder_name = 'June_10th_2020'
saved_model_filename = "saved model.h5"

if os.path.isfile(keras_root_output_folder + training_folder_name + saved_model_filename):
    train_from_beginning = False
else:
    train_from_beginning = True

num_classes = 44
# learning_rate = 0.0000075
learning_rate = 0.0001
batch_size = 16
num_epochs = 50
dropout_pc = 0.1
optimizer = 'Adam'
loss = 'categorical_crossentropy'
# loss = 'binary_crossentropy'
selected_model = "MobileNetV2"
output_activation ='softmax'


# SELECT MODEL


if train_from_beginning:

    if selected_model == "InceptionV3":

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
    optim = keras.optimizers.Adam(learning_rate=learning_rate)

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

# Train data
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
    horizontal_flip=True,
    )

# ImageDataGenerator.flow_from_dataframe(
#     dataframe,
#     directory=None,
#     x_col="filename",
#     y_col="class",
#     weight_col=None,
#     target_size=(256, 256),
#     color_mode="rgb",
#     classes=None,
#     class_mode="categorical",
#     batch_size=32,
#     shuffle=True,
#     seed=None,
#     save_to_dir=None,
#     save_prefix="",
#     save_format="png",
#     subset=None,
#     interpolation="nearest",
#     validate_filenames=True,
#     **kwargs
# )



train_df = pd.read_csv("E:\\ML Training Data\\" + "train.csv", index_col=None, header=0)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,  # Set to None because the absolute path is provided
    x_col='Full Path',
    y_col='Common name',
    shuffle=True,
    seed=42,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    save_to_dir=None  # "E:\\tmp"
)

#
# images, labels = next(train_generator)
# print(images.dtype, images.shape)
# print(labels.dtype, labels.shape)

# ds = tf.data.Dataset.from_generator(
#     train_generator,
#     output_types=(tf.float32, tf.float32),
#     output_shapes=([16, 224, 224, 3], [16, 44])
# )

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

# labels = (train_generator.class_indices)
# labels = dict((v,k) for k,v in labels.items())

# Eval data
eval_test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)
eval_df = pd.read_csv("E:\\ML Training Data\\" + "eval.csv", index_col=None, header=0)
eval_generator = eval_test_datagen.flow_from_dataframe(
    dataframe=eval_df,
    directory=None,
    x_col='Full Path',
    y_col='Common name',
    shuffle=True,
    seed=42,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# Test data
test_df = pd.read_csv("E:\\ML Training Data\\" + "test.csv", index_col=None, header=0)
test_generator = eval_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=None,
    x_col='Full Path',
    y_col='Common name',
    shuffle=True,
    seed=42,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# def get_run_logdir():
#     import time
#     run_id = time.strftime("run_%Y_%m_%d_%H_%M")
#     root_logdir = 'E:\\KerasOutput\\'
#     full_path = os.path.join(root_logdir, run_id)
#     os.mkdir(full_path,  mode=0o777)
#     os.chdir(full_path)
#     return full_path

# run_logdir = get_run_logdir() + "_LR_" + str(learning_rate) + "_BS_" + str(batch_size)

# DEFINE CALLBACKS
run_logdir = keras_root_output_folder + training_folder_name + "\\"
if not os.path.isdir(run_logdir):
    os.mkdir(run_logdir, mode=0o777)
os.chdir(run_logdir)
# run_logdir = 'C:\\EstimatorOutput\\test\\'

tensorboard_cb = keras.callbacks.TensorBoard(
    run_logdir,
    update_freq=500,
    write_images=False,
    embeddings_freq=1,
    histogram_freq=1)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    run_logdir + saved_model_filename,
    save_best_only=True,
    save_freq='epoch'  # 1000
)

# WRITE LOG FILE FOR CURRENT RUN
log_id = time.strftime("log_run_%Y_%m_%d_%H_%M")
f = open(run_logdir + log_id + ".txt", "w+")
f.write("Model = %s \n" % selected_model)
f.write("Learning rate = %1.20f \n" % learning_rate)
f.write("Batch size = %d \n" % batch_size)
f.write("Num epochs = %d \n" % num_epochs)
f.write("Dropout proportion = %1.2f \n" % dropout_pc)
f.write("Optimizer = %s \n" % optimizer)
f.write("Loss = %s \n" % loss)
f.write("Output activation function = %s \n" % output_activation)
f.close()


# COMPILE
if optimizer == 'Adam':
    optim = keras.optimizers.Adam(learning_rate=learning_rate)

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


# FIT
history = model.fit(
    train_generator,
    steps_per_epoch=1000,
    epochs=num_epochs,
    initial_epoch=15,
    validation_data=eval_generator,
    validation_steps=100,
    # workers=2,
    # use_multiprocessing=True,
    # max_queue_size=2,
    callbacks=[checkpoint_cb, tensorboard_cb]
)

# SAVE MODEL AT END
model.save(run_logdir + "saved model at end of training session.h5")
print("The End")
