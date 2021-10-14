from keras import layers, models, regularizers, optimizers, backend
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score, f1_score
import numpy as np

input_shape = (200, 200, 3)


def model_buid():
    backend.set_learning_phase(0)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, )

    backend.set_learning_phase(1)
    x = base_model.output
    x = layers.GlobalAveragePooling2D(name='average_pool')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.0001), )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), )(x)
    x = layers.BatchNormalization()(x)

    predictions = layers.Dense(8, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    return model
