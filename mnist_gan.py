import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers
import time
from IPython import display

# load data
(x, y), (x_test, y_test) = datasets.mnist.load_data()
# normalization
x = x.reshape(-1, 28, 28, 1).astype('float32')
x = (x-127.5)/127.5

BATCH_SIZE = 100
x = tf.data.Dataset.from_tensor_slices(x).shuffle(len(x)).batch(BATCH_SIZE)


def create_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(1, (5, 5), activation='tanh', padding='same'))

    model.summary()

    return model


def create_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    model.summary()

    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


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


EPOCHES = 25
noise_dim = 100
num_examples_to_generate = 9

seed = tf.random.normal([num_examples_to_generate, noise_dim])
generator = create_generator()
discriminator = create_discriminator()


def train(dataset, epochs):
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    for e in range(epochs):
        start = time.time()

        for img_batch in dataset:
            noise = tf.random.normal([BATCH_SIZE, 100])

            with tf.GradientTape() as generative_tape, tf.GradientTape() as discriminative_tape:
                generated = generator(noise, training=True)
                real = discriminator(img_batch, training=True)
                fake = discriminator(generated, training=True)

                generator_loss = find_generator_loss(fake)
                discriminator_loss = find_discriminator_loss(real, fake)

                gradients_generator = generative_tape.gradient(generator_loss, generator.trainable_variables)
                gradients_discriminator = discriminative_tape.gradient(discriminator_loss,
                                                                       discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients((zip(gradients_discriminator,
                                                             discriminator.trainable_variables)))

        display.clear_output(wait=True)
        generate_save_image(generator, e+1, seed)
        print('Time for epoch {} is {} sec'.format(e+1, time.time()-start))

    display.clear_output(wait=True)
    generate_save_image(generator, epochs, seed)


def generate_save_image(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(3, 3))

    for i in range(predictions.shape[0]):
        plt.subplot(3, 3, i+1)
        plt.imshow(predictions[i, :, :, 0] * 225.0, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))


def display_image(epoch_no):
    return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


train(x, EPOCHES)
display_image(EPOCHES)

