from tensorflow.keras import layers, models, optimizers
from keras.preprocessing import image
import matplotlib.pyplot as plt


img_size = 64
batch_size = 128


datagen = image.ImageDataGenerator(rescale=1./255., validation_split=0.2,
                                   vertical_flip=True,
                                   brightness_range=(0.5, 1.5),
                                   fill_mode='nearest')

train_generator = datagen.flow_from_directory(directory="C:/Users/Nrx03/Desktop/deep_learning/proj/data/train",
                                              class_mode='sparse',
                                              target_size=(img_size, img_size),
                                              shuffle=True,
                                              seed=3,
                                              batch_size=batch_size)
valid_generator = datagen.flow_from_directory(directory="C:/Users/Nrx03/Desktop/deep_learning/proj/data/train",
                                              class_mode='sparse',
                                              target_size=(img_size, img_size),
                                              shuffle=True,
                                              seed=3,
                                              batch_size=batch_size)


test_datagen = image.ImageDataGenerator(rescale=1./255.)
test_generator = test_datagen.flow_from_directory(directory="C:/Users/Nrx03/Desktop/deep_learning/proj/data/test",
                                                  class_mode='sparse',
                                                  target_size=(img_size, img_size),
                                                  shuffle=True,
                                                  seed=3,
                                                  batch_size=batch_size)


model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization())

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

# model.compile(optimizers.RMSprop(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizers.SGD(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, validation_data=valid_generator, epochs=60)

model.evaluate(test_generator)

model.save('cnn_cancer_binary_save2.h5')

# plot loss and accuracy for model
epochs = range(len(history.history['accuracy']))
plt.figure()
plt.plot(epochs, history.history['accuracy'], 'b', label="Training accuracy")
plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation accuracy')
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig('model2_acc.jpg')

plt.figure()
plt.plot(epochs, history.history['loss'], 'b', label="Training loss")
plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
plt.title("Training and validation loss")
plt.legend()
plt.savefig('model2_loss.jpg')
