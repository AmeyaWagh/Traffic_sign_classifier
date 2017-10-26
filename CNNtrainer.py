#!/usr/bin/python

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

import pickle
import numpy as np

import matplotlib.pyplot as plt
import cv2
import pandas as pd

training_file = './data/train.p'
validation_file= './data/validate.p'
testing_file = './data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_train = len(X_train)

# TODO: Number of validation examples
n_validation = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape


n_classes = len(np.unique(y_train))


print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


def normalize_image(image):
    return -0.5 + (image*1.0)/(255)

def Gray_image(image):
    return np.resize(cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)[:,:,0],(28,28,1))

def preprocess(image):
    img= [] 
    for i in image:
        img.append(normalize_image(Gray_image(i)))
    img = np.array(img)        
    return img

X_train = preprocess(X_train)
X_valid = preprocess(X_valid)
X_test  = preprocess(X_test)

# 6. Preprocess class labels
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
y_valid = np_utils.to_categorical(y_valid, n_classes)

print "imageshape after grayscale",X_train[0].shape
#------------------------------------------------------------------#

# labels = pd.read_csv('signnames.csv').values
# #plotting 50 random images the training dataset images 
# X_t, y_t = train['features'], train['labels']
# count = 1 
# fig = plt.figure(figsize=(50, 50))
# for i in range(0,500,50):
#     ax = fig.add_subplot(10,5 ,count)
#     # plt.plot(X_t[i])
#     plt.imshow(X_t[i])
#     ax.set_title(labels[y_t[i]],fontsize= 30)
#     count += 1
# #------------------------------------------------------------------#
# def plot_classes_distribution(labels):
#     fig ,ax = plt.subplots(figsize = (10,15))
#     unique, counts = np.unique(labels, return_counts=True)
#     ax.barh(unique,counts,align = 'center', color = 'blue')
#     #ax.set_yticks()
#     ax.invert_yaxis()
#     ax.set_xlabel('Frequency of labels')
#     plt.show()

# plot_classes_distribution(y_train)
# plot_classes_distribution(y_valid)
# plot_classes_distribution(y_test)

#------------------------------------------------------------------#

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
image_shape = X_train[0].shape
print "X_train",np.shape(X_train)
print "y_train",np.shape(y_train)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# LeNet

model = Sequential()
model.add(Conv2D(28, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(28, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, 
          batch_size=32, epochs=10, verbose=1)

score = model.evaluate(X_test, y_test, verbose=0)
print 'score',score