from keras import layers, models, regularizers, optimizers, backend
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator

input_size = (224, 224, 3)
target_size = (224, 224)
batch_size = 32

train_path = ''
test_path = ''

datagen = ImageDataGenerator(rescale=1./225, validation_split=0.2, vertical_flip=True)
train_data = datagen.flow_from_directory(train_path,
                                         class_mode='sparse',
                                         subset='training',
                                         shuffle=True,
                                         batch_size=batch_size,
                                         target_size=target_size)
validation_data = datagen.flow_from_directory(train_path,
                                              class_mode='sparse',
                                              subset='validation',
                                              shuffle=True,
                                              batch_size=batch_size,
                                              target_size=target_size)
test_datagen = ImageDataGenerator(rescale=1./225)
test_data = test_datagen.flow_from_directory(test_path,
                                             class_mode='sparse',
                                             shuffle=False,
                                             target_size=target_size,
                                             batch_size=66)


model = models.Sequential()

model.add(layers.Conv2D(64, (3, 3), input_shape=input_size, padding='same'))
model.add(layers.Conv2D)
