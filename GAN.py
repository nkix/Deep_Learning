import os

import tensorflow as tf
import time
import pathlib
import random
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display


data_path = pathlib.Path("C:/Users/Nrx03/Desktop/科研/as4")
all_image_paths = list(data_path.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
image_count = len(all_image_paths)
ds = tf.data.Dataset.from_tensor_slices(all_image_paths[:50])

# preprocess data
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (112, 112))

    return tf.expand_dims(image[:, :, 0], axis=2)


image_ds = ds.map(load_and_preprocess_image)
ds = image_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
BATCH_SIZE = 32
ds = ds.batch(BATCH_SIZE)


def change_range(image):
    return 2*image-1


train_ds = ds.map(change_range)
x_train = next(iter(train_ds))

EPOCHES = 10
num_examples = 16

seed = tf.random.normal([num_examples, 100])


# deconvolutional neural network
def make_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 512)))

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    # model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False))

    return model


# convolutional neural network
def make_discriminator():
    model = tf.keras.Sequential()

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[112, 112, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.BatchNormalization())

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


generator = make_generator()
discriminator = make_discriminator()


def find_discriminator_loss(real, fake):
    real = tf.sigmoid(real)
    fake = tf.sigmoid(fake)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = loss(tf.ones_like(real), real)
    fake_loss = loss(tf.zeros_like(fake), fake)

    return real_loss+fake_loss


def find_generator_loss(fake):
    fake = tf.sigmoid(fake)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return loss(tf.ones_like(fake), fake)


def generate_save_image(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def train(dataset, epochs):
    generator_optimizer = tf.keras.optimizers.Adam()
    discriminator_optimizer = tf.keras.optimizers.Adam()

    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                     discriminator_optimizer = discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    for e in range(epochs):
        start = time.time()
        print(e)

        for img_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, 100])

            with tf.GradientTape() as generative_tape, tf.GradientTape() as discriminative_tape:
                generated = generator(noise)
                real = discriminator(img_batch)
                fake = discriminator(generated)

                generator_loss = find_generator_loss(fake)
                discriminator_loss = find_discriminator_loss(real, fake)

                gradients_generator = generative_tape.gradient(generator_loss, generator.trainable_variables)
                gradients_discriminator = discriminative_tape.gradient(discriminator_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients((zip(gradients_discriminator, discriminator.trainable_variables)))

        display.clear_output(wait=True)
        generate_save_image(generator, e+1, seed)
        print('Time for epoch {} is {} sec'.format(e+1, time.time()-start))

    display.clear_output(wait=True)
    generate_save_image(generator, epochs, seed)


train(train_ds, EPOCHES)

