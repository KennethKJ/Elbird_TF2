import tensorflow as tf
import tensorflow_addons as tfa


def get_dataset(csv_of_filenames, params=None, augment=True, mode='TRAIN'):

    if params is None:
        params = {}
        params['buffer_size'] = 10000
        params['num_parallel_calls'] = 4
        params['batch_size'] = 16
        params['batch_prefetch'] = 10
        params['image_size'] = [224, 224, 3]
        params['max_delta'] = 63.0 / 2 / 255.0  # Image brightness augmentation parameter
        params['contrast_lower'] = 0.2
        params['contrast_higher'] = 1.8 / 2
        params['image_max_rotation'] = 25
        params['image_crop_margin'] = 50
        params['num_classes'] = 56
        params['model_input_key'] = "input_1"
        params['selected_model'] = "MobileNetV2"

    if params['selected_model'] == "MobileNetV2" or params['selected_model'] == "InceptionV3":
        from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocessing
    elif params['selected_model'] == "VGG16":
        from tensorflow.keras.applications.vgg16 import preprocess_input as preprocessing

    params['preprocessing_fn'] = preprocessing

    # Create tf.data.dataset from filename
    dataset = tf.data.TextLineDataset(filenames=csv_of_filenames) \
        .map(map_func=decode_csv,
             num_parallel_calls=params['num_parallel_calls'])

    if augment:
        dataset = dataset.map(map_func=lambda x, y, z=params: read_and_preprocess_with_augment(x, y, z),
                              num_parallel_calls=params['num_parallel_calls'])
    else:
        dataset = dataset.map(map_func=lambda x, y, z=params: read_and_preprocess(x, y, z),
                              num_parallel_calls=params['num_parallel_calls'])

    if mode == 'TRAIN':
        num_epochs = None  # indefinitely
        dataset = dataset.shuffle(buffer_size=params['buffer_size'])
    else:
        num_epochs = 1  # end-of-input after this

    dataset = dataset.repeat(count=num_epochs).batch(batch_size=params['batch_size']).prefetch(params['batch_prefetch'])
    # images, labels = dataset.make_one_shot_iterator().get_next()

    return dataset


def read_and_preprocess_with_augment(image_bytes, label, params):
    return read_and_preprocess(image_bytes, label,  params, augment=True)


def read_and_preprocess(image_bytes, label=None, params=None, augment=False):

    image = tf.image.decode_jpeg(contents=image_bytes,
                                 channels=params['image_size'][2])
    image = tf.image.convert_image_dtype(image=image,
                                         dtype=tf.float32)  # 0-1
    image = tf.expand_dims(input=image,
                           axis=0)  # resize_bilinear needs batches

    if augment:

        # Resize to slightly larger than target size
        # image = tf.image.resize(images=image,
        #                         method=tf.image.ResizeMethod.BILINEAR,
        #                         size=[224 + 50, 224 + 50],
        #                         preserve_aspect_ratio=True)
        image = tf.image.resize_with_pad(image,
                                         params['image_size'][0] + params['image_crop_margin'],
                                         params['image_size'][1] + params['image_crop_margin'])

        # Image random rotation
        degree_angle = tf.random.uniform((), minval=-params['image_max_rotation'],
                                         maxval=params['image_max_rotation'], dtype=tf.dtypes.float32)
        radian = degree_angle * 3.14 / 180
        image = tfa.image.rotate(image, radian, interpolation='NEAREST')

        # remove batch dimension
        image = tf.squeeze(input=image, axis=0)

        # Random Crop
        image = tf.image.random_crop(value=image, size=params['image_size'])

        # Random L-R flip
        image = tf.image.random_flip_left_right(image=image)

        # Random brightness
        image = tf.image.random_brightness(image=image, max_delta=params['max_delta'])

        # Random contrast
        image = tf.image.random_contrast(image=image,
                                         lower=params['contrast_lower'],
                                         upper=params['contrast_higher'])

    else:
        image = tf.image.resize_bilinear(images=image,
                                         size=[params['image_size'][0], params['image_size'][1]],
                                         align_corners=False)
        image = tf.squeeze(input=image, axis=0)  # remove batch dimension

    image = tf.round(image * 255)

    preprocessing_fn = params['preprocessing_fn']
    image = preprocessing_fn(image)

    label = tf.one_hot(tf.strings.to_number(label, out_type=tf.int32), depth=params['num_classes'])

    return {params['model_input_key']: image}, label


# def _input_fn():
def decode_csv(csv_row):
    path_and_filename, label = tf.io.decode_csv(records=csv_row, select_cols=[1, 2], record_defaults=[[""], [""]])
    # print(path_and_filename)
    # print(label)
    image_bytes = tf.io.read_file(filename=path_and_filename)
    return image_bytes, label
