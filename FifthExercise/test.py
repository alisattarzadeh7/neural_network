#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 05:42:19 2020

@author: alisattarzadeh
"""

train_path = "/media/alisattarzadeh/New Volume/dataset/seg_train/seg_train/"
test_path = "/media/alisattarzadeh/New Volume/dataset/seg_test/seg_test/"

from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks

model_checkPoint = callbacks.ModelCheckpoint('model.h5', save_best_only=True)
logger = callbacks.CSVLogger('training.log')
tensorboard = callbacks.TensorBoard(log_dir='./tensorboard')

class_names = ['sea', 'street', 'forest', 'buildings', 'mountain']
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(150, 150), classes=class_names,
                                                         batch_size=10000)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(150, 150), classes=class_names,
                                                        batch_size=2000)


def plots(ims, figsize=(12, 6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0, 2, 3, 1))
    f = plt.figure(figsize=figsize)
    cols = len(ims) // rows if len(ims) % 2 == 0 else len(ims) // rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i + 1)
        sp.axis('off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


X_train, y_train = next(train_batches)
X_test, y_test = next(test_batches)
# plots(X_train,titles=y_train)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.losses import binary_crossentropy
import tensorflow as tf

myInput = Input(shape=(150, 150, 3))

conv1 = Conv2D(16, 3, activation='relu', padding='same')(myInput)

pool1 = MaxPooling2D(pool_size=2)(conv1)
conv2 = Conv2D(32, 3, activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D(pool_size=2)(conv2)

flat = Flatten()(pool2)
out_layer = Dense(5, activation='softmax')(flat)

myModel = Model(myInput, out_layer)

import tensorflowjs as tfjs

'''
log_dir = "logs/fit/" + time.strftime('%Y%m%d-%H%M%S')
tensorboard_callback = TensorBoard(log_dir=log_dir,histogram_freq=1)
'''
myModel.summary()

myModel.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=['accuracy'])

myModelHistory = myModel.fit(X_train, y_train, batch_size=20, epochs=10,
                             callbacks=[model_checkPoint, logger, tensorboard])

accuracy = myModelHistory.history['accuracy']
print(accuracy)
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(myModelHistory.history['loss'])
plt.plot(accuracy)

Y_pred = myModel.predict(X_test)
y_test = np.argmax(y_test, axis=1)
Y_pred = np.argmax(Y_pred, axis=1)

plt.figure(figsize=(12, 24))
idx = np.random.choice(len(X_test), 10, replace=False)

p = myModel.predict(X_test[idx])

print(myModel.predict(X_test[idx]))
import datetime

time = datetime.datetime.now()

for i in range(len(idx)):
    plt.subplot(10, 2, 2 * i + 1)
    plt.imshow(X_test[idx[i]], interpolation='spline16')
    plt.title(class_names[y_test[idx[i]]])
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    pred_label = np.argsort(-p[i])[:3]
    pred_prob = [p[i][l] for l in pred_label]
    pred_label = [class_names[l] for l in pred_label]

    plt.subplot(10, 2, 2 * i + 2)
    plt.bar(range(3), pred_prob)
    plt.xticks(range(3), pred_label)

plt.show()

myModel.save(f"./nnmodel/{1}", save_format='tf')
myModel.save_weights('weights.h5')

tfjs.converters.save_keras_model(myModel, './webTest')

