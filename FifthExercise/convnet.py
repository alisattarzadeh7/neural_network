from keras.applications import VGG16
from keras.utils import img_to_array
from keras.utils import load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing import image as image_utils
# import opencv as cv2
from keras import optimizers
import matplotlib.pyplot as plt
from keras import models, layers
import os
import numpy as np

model = VGG16(weights='imagenet')

model.summary()
# model = VGG16(weights="imagenet")
# load an image from file
image = load_img('./dataset/IMG_20221203_193308.jpg', target_size=(224, 224))
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
image = preprocess_input(image)
preds = model.predict(preprocess_input(image))
print(decode_predictions(preds))

# loop over the predictions and display the rank-5 predictions +
# probabilities to our terminal
# for (i, (imagenetID, label, prob)) in enumerate(P[0]):
# 	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
# # load the image via OpenCV, draw the top prediction on the image,
# # and display the image to our screen
# orig = cv2.imread(args["image"])
# (imagenetID, label, prob) = P[0][0]
# cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
# 	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# cv2.imshow("Classification", orig)
# cv2.waitKey(0)

# # convert the image pixels to a numpy array
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# print(image)
# # print(image)
# # prepare the image for the VGG model
# image = preprocess_input(image)
#
#
# test_path = "/var/www/html/neural_network/FifthExercise/dataset"
# from keras.preprocessing.image import ImageDataGenerator
#
# #
# # test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(150, 150))
# conv_base = VGG16()
#
# conv_base.summary()
# conv_base.trainable = False
# yhat = conv_base.predict(image)
# label = decode_predictions(yhat)
# # retrieve the most likely result, e.g. highest probability
# label = label[0][0]
# # print the classification
# print('%s (%.2f%%)' % (label[1], label[2]*100))

#
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.summary()
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=30,
#     validation_data=validation_generator,
#     validation_steps=50,
#     verbose=2)
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')
#
# print('This is the number of trainable weights '
#       'before freezing the conv base:', len(model.trainable_weights))
#
#
#
# base_dir = './dataset/'
# test_dir = os.path.join(base_dir, 'test')
#
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=2e-5),
#               metrics=['acc'])
# print('This is the number of trainable weights '
#       'after freezing the conv base:', len(model.trainable_weights))
#


##
# import os
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator
#
# base_dir = 'E:\\software\\cats_and_dogs_small'
#
# train_dir = os.path.join(base_dir, 'train')
# validation_dir = os.path.join(base_dir, 'validation')

#
# datagen = ImageDataGenerator(rescale=1./255)
# batch_size = 20
#
# def extract_features(directory, sample_count):
#     features = np.zeros(shape=(sample_count, 4, 4, 512))
#     labels = np.zeros(shape=(sample_count))
#     generator = datagen.flow_from_directory(
#         directory,
#         target_size=(150, 150),
#         batch_size=batch_size,
#         class_mode='binary')
#     i = 0
#     for inputs_batch, labels_batch in generator:
#         features_batch = conv_base.predict(inputs_batch)
#         features[i * batch_size : (i + 1) * batch_size] = features_batch
#         labels[i * batch_size : (i + 1) * batch_size] = labels_batch
#         i += 1
#         if i * batch_size >= sample_count:
#             # Note that since generators yield data indefinitely in a loop,
#             # we must `break` after every image has been seen once.
#             break
#     return features, labels
#
# train_features, train_labels = extract_features(train_dir, 2000)
# validation_features, validation_labels = extract_features(validation_dir, 1000)
# test_features, test_labels = extract_features(test_dir, 1000)
#
# train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
# validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))
#
#
# from keras import models
# from keras import layers
# from keras import optimizers
#
# model = models.Sequential()
# model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(1, activation='sigmoid'))
#
# model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
# history = model.fit(train_features, train_labels,
#                     epochs=30,
#                     batch_size=20,
#                     validation_data=(validation_features, validation_labels))
#
#
#
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()
#
# exit(0)
# #####################################################################


# %%


# %%


# Note that the validation data should not be augmented!
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#     # This is the target directory
#     train_dir,
#     # All images will be resized to 150x150
#     target_size=(150, 150),
#     batch_size=20,
#     # Since we use binary_crossentropy loss, we need binary labels
#     class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')


#
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()
#
# conv_base.trainable = True
#
# set_trainable = False
# for layer in conv_base.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False
#
#
# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-5),
#               metrics=['acc'])
#
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=50)
#
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(len(acc))
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()
#
#
# def smooth_curve(points, factor=0.8):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points
#
# plt.plot(epochs,
#          smooth_curve(acc), 'bo', label='Smoothed training acc')
# plt.plot(epochs,
#          smooth_curve(val_acc), 'b', label='Smoothed validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs,
#          smooth_curve(loss), 'bo', label='Smoothed training loss')
# plt.plot(epochs,
#          smooth_curve(val_loss), 'b', label='Smoothed validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()
#

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary')
#
# test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
# print('test acc:', test_acc)
