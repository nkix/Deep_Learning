import pandas as pd
import cv2
from tensorflow.keras import models, layers, optimizers, applications, regularizers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt


train = pd.read_csv("C:/Users/Nrx03/Desktop/deep_learning/proj/ISIC_2019_Training_set.csv")
train['image'] += '.jpg'

test = pd.read_csv("C:/Users/Nrx03/Desktop/deep_learning/proj/Test_set.csv")
test['image'] += '.jpg'

img_size = 224
batch_size = 32

datagen = image.ImageDataGenerator(rescale=1./255., validation_split=0.25,
                                   channel_shift_range=10,
                                   vertical_flip=True,
                                   width_shift_range=0.2,
                                   shear_range=0.3,
                                   fill_mode='nearest')

train_generator = datagen.flow_from_dataframe(dataframe=train,
                                              directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/train",
                                              x_col='image',
                                              y_col=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'],
                                              class_mode='raw',
                                              subset='training',
                                              target_size=(img_size, img_size),
                                              shuffle=True,
                                              batch_size=batch_size)

valid_generator = datagen.flow_from_dataframe(dataframe=train,
                                              directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/train",
                                              x_col='image',
                                              y_col=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'],
                                              class_mode='raw',
                                              subset='validation',
                                              target_size=(img_size, img_size),
                                              batch_size=batch_size,
                                              shuffle=True)

test_datagen = image.ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_dataframe(dataframe=test,
                                                  directory="C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/test",
                                                  x_col='image',
                                                  y_col=['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'],
                                                  class_mode='raw',
                                                  target_size=(img_size, img_size),
                                                  shuffle=True,
                                                  batch_size=32)

conv_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

for layer in conv_base.layers[:165]:
    layer.trainable = False
for layer in conv_base.layers[165:]:
    layer.trainable = True

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(9, activation='softmax'))

# model = ResNet50(input_shape=(128, 128, 3), weights='imagenet', classes=9, include_top=False, pooling='max')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=valid_generator, epochs=30)

model.evaluate(valid_generator)

model.save('cnn_cancer_save2.h5')

# plot loss and accuracy for model
epochs = range(len(history.history['accuracy']))
plt.figure()
plt.plot(epochs, history.history['accuracy'], 'b', label="Training accuracy")
plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation accuracy')
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig('model_acc_res.jpg')

plt.figure()
plt.plot(epochs, history.history['loss'], 'b', label="Training loss")
plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
plt.title("Training and validation loss")
plt.legend()
plt.savefig('model_loss_res.jpg')

