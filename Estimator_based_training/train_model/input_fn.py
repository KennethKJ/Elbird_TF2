import tensorflow as tf
# from keras.applications.vgg16 import preprocess_input as vgg16_input_processing
from tensorflow.keras.applications.mobilenet import preprocess_input as vgg16_input_processing
from tensorflow.keras.applications.inception_v3 import preprocess_input as inceptionV3_preprocessing
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_preprocessing

tf.logging.set_verbosity(v=tf.logging.INFO)


def read_and_preprocess_with_augment(image_bytes, label=None, pr=None):
    return read_and_preprocess(image_bytes, label,  pr, augment=True)


def read_and_preprocess(image_bytes, label=None, pr=None, augment=False):

    image = tf.image.decode_jpeg(contents=image_bytes,
                                 channels=pr['num channels'])
    image = tf.image.convert_image_dtype(image=image,
                                         dtype=tf.float32)  # 0-1
    image = tf.expand_dims(input=image,
                           axis=0)  # resize_bilinear needs batches

    if augment:

        # Resize to slightly larger than target size
        image = tf.image.resize_bilinear(images=image,
                                         size=[pr['image size'][0] + 50, pr['image size'][1] + 50],
                                         align_corners=False)

        # Image random rotation
        degree_angle = tf.random.uniform((), minval=-25, maxval=25, dtype=tf.dtypes.float32)
        radian = degree_angle * 3.14 / 180
        image = tf.contrib.image.rotate(image, radian, interpolation='NEAREST')

        # remove batch dimension
        image = tf.squeeze(input=image, axis=0)

        # Random Crop
        image = tf.random_crop(value=image, size=[pr['image size'][0],  pr['image size'][1],  pr['num channels']])
        # Random L-R flip
        image = tf.image.random_flip_left_right(image=image)
        # Random brightness
        image = tf.image.random_brightness(image=image, max_delta=63.0/2/ 255.0)
        # Random contrast
        image = tf.image.random_contrast(image=image, lower=0.2, upper=1.8/2)

    else:
        image = tf.image.resize_bilinear(images=image, size=[pr['image size'][0], pr['image size'][1]], align_corners=False)
        image = tf.squeeze(input=image, axis=0)  # remove batch dimension

    image = tf.round(image * 255)

    # image = tf.clip_by_value(image, 0, 255, name='Clipper')
    # image = image / 255

    # Pixel values are in range [0,1], convert to [-1,1]
    # image = tf.subtract(x=image, y=0.5)
    # image = tf.multiply(x=image, y=2.0)
    #
    # image = tf.cast(tf.round(image * 255), tf.int32)
    # image = tf.expand_dims(input=image, axis=0)  #
    if pr['Transfer learning model'] == "VGG16":
        image = vgg16_input_processing(image)

    elif pr['Transfer learning model'] == "InceptionV3":
        image = inceptionV3_preprocessing(image)

    elif pr['Transfer learning model'] == "Nasnet":
        image = nasnet_preprocessing(image)

    else:
        print("Oops")

    label = tf.one_hot(tf.strings.to_number(label, out_type=tf.int32), depth=pr['num classes'])

    return {"input_1": image}, label


def make_input_fn(csv_of_filenames, mode, pr, augment=False):
    def _input_fn():
        def decode_csv(csv_row):
            filename, label = tf.decode_csv(records=csv_row, record_defaults=[[""], [""]])
            image_bytes = tf.read_file(filename=pr['dB_path'] + filename)
            return image_bytes, label

        # Create tf.data.dataset from filename
        dataset = tf.data.TextLineDataset(filenames=csv_of_filenames)\
            .map(map_func=decode_csv,
                 num_parallel_calls=pr['num parallel calls'])

        if augment:
            dataset = dataset.map(map_func=lambda x, y: read_and_preprocess_with_augment(x, y, pr), num_parallel_calls=pr['num parallel calls'])
        else:  # (lambda x: mapfunc(x, [7])
            dataset = dataset.map(map_func=lambda x, y: read_and_preprocess(x, y, pr), num_parallel_calls=pr['num parallel calls'])

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None  # indefinitely
            dataset = dataset.shuffle(buffer_size=10000)
        else:
            num_epochs = 1  # end-of-input after this

        dataset = dataset.repeat(count=num_epochs).batch(batch_size=pr["batch size"]).prefetch(2)
        images, labels = dataset.make_one_shot_iterator().get_next()

        return images, labels
    return _input_fn


test = False

if test:
    from matplotlib import pyplot as plt
    model_num = 65
    params = {}
    params['train csv'] = "C:/Users/alert/Google Drive/ML/ElBird/Data_proc/train_set_local.csv"
    params['eval csv'] = "C:/Users/alert/Google Drive/ML/ElBird/Data_proc/eval_set_local.csv"
    params['output path'] = "C:/EstimatorOutput/" + str(model_num) + "/"
    params['data path'] = "C:/Users/alert/Google Drive/ML/Databases/Birds_dB/Images"
    params['num channels'] = 3
    params["batch size"] = 16
    params['use random flip'] = True
    params['learning rate'] = 0.0002  #
    params['dropout rate'] = 0.50
    params['num classes'] = 123
    params['train steps'] = 10000
    params['eval steps'] = 100
    params['eval_throttle_secs'] = 600
    params['isRunOnCloud'] = False
    params['num parallel calls'] = 4
    params['dB_path'] = "C:/Users/alert/Google Drive/ML/Databases/Birds_dB/Images/"
    params['Transfer learning model'] = "VGG16"

    if params['Transfer learning model'] == "InceptionV3":
        params['image size'] = [299, 299]
    elif params['Transfer learning model'] == "VGG16":
        params['image size'] = [244, 224]
    elif params['Transfer learning model'] == "Nasnet":
        params['image size'] = [331, 331]

    print("Hi, this is task.py talking")
    print("Will train for {} steps using batch_size={}".format(params['train steps'], params['batch size']))
    print("********************************")
    print("PARAMETER SETTINGS:")
    for key, value in params.items():
        print(key + ": " + str(value))
    print("********************************")

    a1 = make_input_fn("C:/Users/alert/Google Drive/ML/ElBird/Data_proc/test_set_local.csv",  tf.estimator.ModeKeys.TRAIN, params, augment=True)
    b, c = a1()

    filename = "C:/Users/alert/Google Drive/ML/Databases/Birds_dB/Mappings/classes.txt"
    LIST_OF_LABELS = [line.strip() for line in open(filename, 'r')]

    labels_table_2 = tf.contrib.lookup.index_table_from_tensor(
        mapping=tf.constant(value=LIST_OF_LABELS, dtype=tf.string))

    with tf.Session() as sess:
        tf.tables_initializer().run()

        while 1 == 1:

            imgs, labls = sess.run(a1())

            for i in range(10):
                immy = imgs['input_1'][i, :, :, :]
                plt.imshow(immy)
                plt.title(labls[i])
                plt.show(block=False)
                print("Done")

        print("Done")



def get_image(filename):
    image_bytes = tf.read_file(filename=filename)
    image = tf.image.decode_jpeg(contents=image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image=image, dtype=tf.float32)  # 0-1
    image = tf.expand_dims(input=image, axis=0)  # resize_bilinear needs batches

    image = tf.image.resize_bilinear(images=image, size=[HEIGHT, WIDTH], align_corners=False)
    image = tf.squeeze(input=image, axis=0)  # remove batch dimension

    image = tf.cast(tf.round(image * 255), tf.int32)
    image = preprocess_input(image)

    return image