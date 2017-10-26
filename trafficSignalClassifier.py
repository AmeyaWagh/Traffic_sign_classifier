#!/usr/bin/python
from keras.models import model_from_json
import os
import pickle
import numpy as np
import cv2
import csv
from keras.utils import np_utils


class Classifier:

    def __init__(self, imageShape=(28, 28, 1),nClasses=5):
        self.modelPath = './model'
        self.signNamesFile = './signnames.csv'
        self.validation_file = './data/validate.p'
        self.imageShape = imageShape
        self.nClasses=nClasses
        self.signNames = self.readSignNameCSV()
        self.model = self.loadModel()

    def readSignNameCSV(self, nClasses=None):
        with open(self.signNamesFile, 'r') as fp:
            data = list(csv.DictReader(fp))
        # convert dictionary to list and format file names
        data = [
            everyData['SignName'] for everyData in data]

        if nClasses is not None:
            data = data[0:self.nClasses]
        return data

    def Gray_image(self,image):
        return np.resize(cv2.cvtColor(
            image, cv2.COLOR_RGB2YCrCb)[:, :, 0], self.imageShape)

    def normalize_image(self,image):
        return  (image*1.0)/(255.0) - 0.5
    
    def preProcessImage(self, image):
        img = []
        for i in image:
            img.append(self.normalize_image(self.Gray_image(i)))
        img = np.array(img)
        return img

    def loadModel(self):
        # load json and create model
        # json_file = open(os.path.join(modelPath,'model.json'), 'r')
        # loaded_model_json = json_file.read()
        # json_file.close()
        with open(os.path.join(self.modelPath,'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(os.path.join(self.modelPath,"model.h5"))
        loaded_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
        print("Loaded model from disk")
        return loaded_model

    def predict(self,X):
        X = self.preProcessImage(X)
        predictions = self.model.predict(X)
        predictions = [pred.index(max(pred)) for pred in predictions.tolist()]
        return predictions

if __name__ == '__main__':
    classifier = Classifier(imageShape=(28, 28, 1),nClasses=5)
    
    validation_file= './data/validate.p'

    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)

    X_valid, y_valid = valid['features'], valid['labels']
    
    print classifier.predict(X_valid)