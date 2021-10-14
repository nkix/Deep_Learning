import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, accuracy_score, f1_score
import numpy as np
import itertools
from keras import models
import os


def print_aic(y_true, y_pred):
    fpr, tpr, thresholds_keras = roc_curve(y_true, y_pred)
    auc_img = auc(fpr, tpr)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='S3< val (AUC = {:.3f})'.format(auc))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')

    plt.show()


def print_acc_loss(history):
    # plot loss and accuracy for model
    epochs = range(len(history.history['accuracy']))
    plt.figure()
    plt.plot(epochs, history.history['accuracy'], 'b', label="Training accuracy")
    plt.plot(epochs, history.history['val_accuracy'], 'r', label='Validation accuracy')
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.savefig('model_bi_acc.jpg')

    plt.figure()
    plt.plot(epochs, history.history['loss'], 'b', label="Training loss")
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation loss')
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig('model_bi_loss.jpg')


def get_confusion_matrix(y_true, y_pred):
    n_classes = len(np.unique(y_true))
    conf = np.zeros((n_classes, n_classes))
    for actual, pred in zip(y_true, y_pred):
        conf[int(actual)][int(pred)] += 1
    return conf.astype('int')


def print_confusion_matrix(classes, y_true, y_pred):
    conf = get_confusion_matrix(y_true, y_pred)

    plt.imshow(conf, interpolation='nearest', cmap='g')
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = conf.max() / 2.
    for i, j in itertools.product(range(conf.shape[0]), range(conf.shape[1])):
        plt.text(j, i, format(conf[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if conf[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')








