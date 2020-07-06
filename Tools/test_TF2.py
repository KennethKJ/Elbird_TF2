import tensorflow as tf
from tensorflow import keras

import pandas as pd
import matplotlib.pyplot as plt


print("TensorFlow Version = " + tf.__version__)
print("Keras Version = " + tf.keras.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[: 5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[: 5000], y_train_full[5000:]

class_names = [" T-shirt/ top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))  # check out keras.activations for other activation functions
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))

# w = model.layers[1].weights
# model.get_layer('dense') is hidden1
model.summary()
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

history = model.fit(X_train,
                    y_train,
                    epochs=3,
                    validation_data=(X_valid, y_valid)
                    )

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1] plt.show()
plt.show()

# Run evaluation on test set
e = model.evaluate(X_test, y_test)

# Run specific predictions
X_new = X_test[: 3]
y_proba = model.predict(X_new)
y_proba.round(2)

print("The End")


