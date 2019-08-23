from __future__ import print_function
import numpy as np
import scipy.io
import cv2

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from matplotlib import pyplot as plt


#-------------- TRAINING PARAMETERS --------------#

batch_size = 32
num_classes = 10
epochs = 20

# input image dimensions
mode = 'char74_fonts' # char74_fonts, google_70k, google_530k, mnist
img_rows, img_cols = 28, 28


#-------------- LOAD IMAGES --------------#

# --> MNIST:
if mode == 'mnist':
  (x_train, y_train), (x_test, y_test) = mnist.load_data() 

# --> Char74:
elif mode == 'char74_fonts':
  x = np.load('../data/char74/fonts/x.npy')
  y = np.load('../data/char74/fonts/y.npy')
  x_train, x_test = np.split(x, [10053]) # 9:1 training:testing split
  y_train, y_test = np.split(y, [10053])

# --> Google:
elif mode == 'google_70k':
  x_train = list()
  y_train = list()
  mat = scipy.io.loadmat('../data/google/train_32x32.mat') # keys: __header__, __version__, __globals__, X, y 
  for img in mat['X'][:,:,:]: # (32, 32, 3, 531131)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    thresh, bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x_train.append(bw)
  for label in mat['y']:
    y_train.append(label)

elif mode == 'google_530k':
  x_train = list()
  y_train = list()
  mat = scipy.io.loadmat('../data/google/extra_32x32.mat') # keys: __header__, __version__, __globals__, X, y 
  for img in mat['X'][:,:,:]: # (32, 32, 3, 531131)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    thresh, bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    x_train.append(bw)
  for label in mat['y']:
    y_train.append(label)

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)



#-------------- AUGMENTATIONS --------------#
# # thresh = 0.5
# # probs = np.random.random(x_train.shape[0])
# augmented = np.zeros(x_train.shape)
# for i in range(x_train.shape[0]):
#     # if probs[i] > thresh:
#     top, bottom, left, right = np.random.randint(5, 15, size=4)
#     temp = cv2.copyMakeBorder(x_train[i], top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
#     augmented[i] = cv2.resize(temp, (28, 28))

# # combine normal data and pre-processed image data
# x_train = np.vstack((x_train, augmented))
# y_train = np.hstack((y_train, y_train))


#-------------- RESHAPE/NORMALIZE -> (num images, 28, 28, 1) --------------#

# reformat total data
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# datagen = ImageDataGenerator(
#     width_shift_range=0.3,
#     height_shift_range=0.3,
#     zoom_range=0.2,
#     shear_range=0.2)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
# datagen.fit(x_train)

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


#-------------- MODEL ARCHITECTURE --------------#

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


#-------------- TRAINING --------------#

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          shuffle=True,
          validation_data=(x_test, y_test))

# model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / batch_size, epochs=epochs)


#-------------- EVALUATION --------------#

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#-------------- SAVE MODEL --------------#

model.save('char74/model_1.h5')
















