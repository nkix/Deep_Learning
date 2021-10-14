import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, datasets, models
from keras.utils.np_utils import to_categorical

# import mnist dataset
(x, y), (x_test, y_test) = datasets.mnist.load_data()
# make info logs printable
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

# normalization
x, x_test = x/255.0, x_test/255.0

x = x.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
y = to_categorical(y)
y_test = to_categorical(y_test)

# create model and add convolution layers and pooling layers
model = models.Sequential()
model.add(layers.Conv2D(32, (5, 5), activation="relu", input_shape=(28, 28, 1)))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, (3, 3), activation="relu"))

model.add(layers.MaxPooling2D((2, 2)))

# fully connected layer
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

# show the model struct
# model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, batch_size=128, validation_split=0.2, epochs=10)

model.save('cnn_mnist_save1.h5')

model.evaluate(x_test, y_test)


