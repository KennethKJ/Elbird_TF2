import tensorflow as tf
from tensorflow.python.keras import estimator as kes
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from train_model.input_fn import make_input_fn


def create_estimator(params):

    if params['Transfer learning model'] == "VGG16":
        from tensorflow.python.keras.applications.vgg16 import VGG16 as TL_model
        base_model = TL_model(weights='imagenet')

    elif params['Transfer learning model'] == "InceptionV3":
        from tensorflow.python.keras.applications.inception_v3 import InceptionV3 as TL_model
        base_model = TL_model(weights='imagenet', include_top=False, input_shape=[224, 224, 3])

    elif params['Transfer learning model'] == "Nasnet":
        from tensorflow.python.keras.applications.nasnet import nasnet
        base_model = nasnet(weights='imagenet', input_shape=[331, 331, 3])

    # Import selected model for transfer learning
    base_model.summary()

    if params['Transfer learning model'] == "VGG16":
        x = base_model.get_layer('fc2').output

        x = Dropout(params['dropout rate'])(x)

        x = Dense(1000, activation='relu')(x)

        x = Dropout(params['dropout rate'])(x)

        x = Dense(500, activation='relu')(x)

        x = Dropout(params['dropout rate'])(x)

        predictions = Dense(params['num classes'], activation="softmax", name="sm_out")(x)

    elif params['Transfer learning model'] == "InceptionV3":
        print("Building Inception V3")
        # base_model.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D(trainable=True)(x)
        x = Dropout(params['dropout rate'])(x)

        x = Dense(1000, activation='relu', trainable=True)(x)

        x = Dropout(params['dropout rate'])(x)

        predictions = Dense(params['num classes'], activation="softmax", trainable=True, name="sm_out")(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    if params['Transfer learning model'] == "VGG16":
        for layer in model.layers:
            layer.trainable = True

    elif params['Transfer learning model'] == "InceptionV3":
        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.train.AdamOptimizer(params['learning rate'],
                                         beta1=0.9,
                                         beta2=0.999),
        # optimizer=tf.train.AdadeltaOptimizer(params['learning rate']),
        # optimizer=tf.train.RMSPropOptimizer(params['learning rate']),
        metrics=["categorical_accuracy"]
    )

    if params['isRunOnCloud']:

        run_config = tf.estimator.RunConfig(
            model_dir=params['output path']
        )
    else:

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        # session_config = tf.contrib.learn.RunConfig(session_config=config)

        run_config = tf.estimator.RunConfig(
            session_config=config,
            model_dir=params['output path'],
            # save_checkpoints_steps=1000
        )

    # Convert to Estimator (https://cloud.google.com/blog/products/gcp/new-in-tensorflow-14-converting-a-keras-model-to-a-tensorflow-estimator)
    estimator_model = kes.model_to_estimator(
        keras_model=model,
        config=run_config
    )

    return estimator_model


def go_train(params):
    # Create the estimator
    Est = create_estimator(params)

    # Set up Estimator train and evaluation specifications
    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(params['train csv'], tf.estimator.ModeKeys.TRAIN, params, augment=True),
        max_steps=params['train steps']
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(params['eval csv'], tf.estimator.ModeKeys.EVAL, params, augment=True),
        steps=params['eval steps'],  # Evaluates on "eval steps" batches
        throttle_secs=params['eval_throttle_secs']
    )

    # Set logging level
    tf.logging.set_verbosity(tf.logging.DEBUG)

    print("Starting training and evaluation ...")

    # Run training and evaluation
    tf.estimator.train_and_evaluate(Est, train_spec, eval_spec)

    print("Training and evaluation round is done")


def go_predict(params):
    # Create the estimator
    Est = create_estimator(params)

    # Set logging level
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # Run preduiction
    return Est.predict(input_fn=make_input_fn(params['test csv'], tf.estimator.ModeKeys.PREDICT, params, augment=False))
