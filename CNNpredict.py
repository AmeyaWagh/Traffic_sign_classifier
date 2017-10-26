#!/usr/bin/python


from keras.models import model_from_json
import os
import pickle
import numpy as np
import cv2
from keras.utils import np_utils

modelPath = './model'

validation_file= './data/validate.p'

with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)

X_valid, y_valid = valid['features'], valid['labels']

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

X_valid = preprocess(X_valid)
n_classes = len(np.unique(y_valid))
y_valid = np_utils.to_categorical(y_valid, n_classes)


# load json and create model
json_file = open(os.path.join(modelPath,'model.json'), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(modelPath,"model.h5"))
print("Loaded model from disk")


loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

score = loaded_model.evaluate(X_valid, y_valid, verbose=0)
print 'score',score


