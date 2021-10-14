import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

model = tf.keras.models.load_model('cnn_cancer_save1.h5')

test = pd.read_csv("C:/Users/Nrx03/Desktop/deep_learning/proj/Test_set.csv")
test['image'] += '.jpg'


def input_pic(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 96))
    img = img / 255

    return img.reshape(-1, 128, 96, 3)


real_outcome = []
for i in range(len(test['image'])):
    if test['MEL'][i] == 1 :
        real_outcome.append(0)
    elif test['NV'][i] == 1:
        real_outcome.append(1)
    elif test['BCC'][i] == 1:
        real_outcome.append(2)
    elif test['AK'][i] == 1:
        real_outcome.append(3)
    elif test['BKL'][i] == 1:
        real_outcome.append(4)
    elif test['DF'][i] == 1:
        real_outcome.append(5)
    elif test['VASC'][i] == 1:
        real_outcome.append(6)
    elif test['SCC'][i] == 1:
        real_outcome.append(7)
    else:
        real_outcome.append(8)

prediction_outcome = []

for image in tqdm(test['image']):
    path = 'C:\\Users\\Nrx03\\Desktop\\deep_learning\\proj\\Test_set\\' + image
    x = input_pic(path)
    prediction = []
    for i in range(50):
        y = model.predict(x)
        prediction.append(np.argmax(y))

    predict = max(prediction, key=prediction.count)
    prediction_outcome.append(predict)

    # print(image + 'predict result: ' + str(predict))

print(prediction_outcome)

count = 0
for i in tqdm(range(len(prediction_outcome))):
    if prediction_outcome[i] == real_outcome[i]:
        count += 1

accuracy = count/len(prediction_outcome)
print("accuracy = " + str(accuracy))
"""
path = 'C:\\Users\\Nrx03\\Pictures\\1.jpg'
x = input_pic(path)
y = model.predict(x)
print('predict result: ' + str(np.argmax(y)))


path = 'C:\\Users\\Nrx03\\Pictures\\2.jpg'
x = input_pic(path)
y = model.predict(x)
print('predict result: ' + str(np.argmax(y)))
path = 'C:\\Users\\Nrx03\\Pictures\\3.jpg'
x = input_pic(path)
y = model.predict(x)
print('predict result: ' + str(np.argmax(y)))
path = 'C:\\Users\\Nrx03\\Pictures\\6.jpg'
x = input_pic(path)
y = model.predict(x)
print('predict result: ' + str(np.argmax(y)))
path = 'C:\\Users\\Nrx03\\Pictures\\8.jpg'
x = input_pic(path)
y = model.predict(x)
print('predict result: ' + str(np.argmax(y)))
path = 'C:\\Users\\Nrx03\\Pictures\\9.jpg'
x = input_pic(path)
y = model.predict(x)
print('predict result: ' + str(np.argmax(y)))"""


