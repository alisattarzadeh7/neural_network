from keras.applications import VGG16
from keras.utils import img_to_array
from keras.utils import load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import numpy as np
import glob

import time

model = VGG16(weights='imagenet')
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 24))
fig = plt.figure(figsize=(12, 24))
for i, photo in enumerate(glob.glob('./dataset/*')):
    plt.subplot(10, 2, 2 * i + 1)
    image = load_img(photo, target_size=(224, 224))
    plt.imshow(image, interpolation='spline16')
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(10, 2, 2 * i + 2)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    preds = model.predict(image)
    preds = decode_predictions(preds)[0]
    plt.bar(range(3), [preds[0][2], preds[1][2], preds[2][2]])
    plt.xticks(range(3), [preds[0][1], preds[1][1], preds[2][1]])
    time.sleep(0.3)
plt.show()
fig.savefig('result.png' ,dpi=fig.dpi)
