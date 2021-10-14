import pandas as pd
import cv2
from keras import models, layers, callbacks, optimizers, initializers, regularizers
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
import matplotlib.pyplot as plt

input_shape = (224, 224, 3)
target_size = (224, 224)
batch_size = 8
"""train = pd.read_csv("C:/Users/Nrx03/Desktop/deep_learning/proj/ISIC_2019_Training_set.csv")
train['image'] += '.jpg'

test = pd.read_csv("C:/Users/Nrx03/Desktop/deep_learning/proj/Test_set.csv")
test['image'] += '.jpg'

datagen = image.ImageDataGenerator(rescale=1./255., validation_split=0.25,
                                   channel_shift_range=10,
                                   shear_range=0.3)
train_generator = datagen.flow_from_dataframe(dataframe=train,
                                              directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/train",
                                              x_col='image',
                                              y_col=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'],
                                              class_mode='raw',
                                              subset='training',
                                              target_size=(img_size, img_size),
                                              shuffle=True,
                                              batch_size=batch_size)

valid_generator = datagen.flow_from_dataframe(dataframe=train,
                                              directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/train",
                                              x_col='image',
                                              y_col=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'],
                                              class_mode='raw',
                                              subset='validation',
                                              target_size=(img_size, img_size),
                                              batch_size=batch_size,
                                              shuffle=True)

test_datagen = image.ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(dataframe=test,
                                                  directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/test",
                                                  x_col='image',
                                                  y_col=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'],
                                                  class_mode='raw',
                                                  target_size=(img_size, img_size),
                                                  shuffle=True,
                                                  batch_size=batch_size)
"""

datagen = image.ImageDataGenerator(rescale=1./255., validation_split=0.2,
                                   vertical_flip=True,
                                   brightness_range=(0.5, 1.5),
                                   fill_mode='nearest')

train_generator = datagen.flow_from_directory(directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/train/",
                                              class_mode='categorical',
                                              target_size=target_size,
                                              shuffle=True,
                                              seed=3,
                                              batch_size=batch_size,
                                              subset='training')
valid_generator = datagen.flow_from_directory(directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/train/",
                                              class_mode='categorical',
                                              target_size=target_size,
                                              shuffle=True,
                                              seed=3,
                                              batch_size=batch_size,
                                              subset='validation')


checkpoint_path = '/cp.ckpt'
call_back = callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

test_datagen = image.ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_directory(directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/test/",
                                                  class_mode='categorical',
                                                  target_size=target_size,
                                                  shuffle=True,
                                                  seed=3,
                                                  batch_size=batch_size)


model = models.Sequential(name='vgg16-sequential')

model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(8, activation='softmax'))

model.summary()


# model.compile(optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizers.rmsprop_v2, loss='categorical_crossentropy', metrics=['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
history = model.fit(train_generator,  steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator, validation_steps=STEP_SIZE_VALID,
                    epochs=30, shuffle=True,
                    callbacks=[call_back])

model.evaluate(test_generator, steps=STEP_SIZE_TEST)

model.save('cnn_cancer_save1.h5')

# plot loss and accuracy for model
epochs = range(len(history.history['accuracy']))
plt.figure()
plt.plot(epochs, history.history['accuracy'], 'b', label="Training accuracy")
plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation accuracy')
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig('model_acc.jpg')

plt.figure()
plt.plot(epochs, history.history['loss'], 'b', label="Training loss")
plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
plt.title("Training and validation loss")
plt.legend()
plt.savefig('model_loss.jpg')

