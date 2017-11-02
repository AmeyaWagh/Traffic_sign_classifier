#!/usr/bin/python
from keras.models import model_from_json
import os
import pickle
import numpy as np
import cv2
import csv
from keras.utils import np_utils


class Classifier:

    def __init__(self, imageShape=(28, 28, 1), nClasses=5):
        self.modelPath = './model'
        self.signNamesFile = './signnames.csv'
        self.validation_file = './data/validate.p'
        self.imageShape = imageShape
        self.nClasses = nClasses
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

    def convertToGray(self, image):
        '''
            convert image to gray scale
        '''
        return np.resize(cv2.cvtColor(
            image, cv2.COLOR_RGB2YCrCb)[:, :, 0], self.imageShape)

    def normalizeImage(self, image):
        '''
            Normalize image to with zero mean
        '''
        return (image*1.0)/(255.0) - 0.5

    def preProcessImage(self, image):
        img = []
        for i in image:
            img.append(self.normalizeImage(self.convertToGray(i)))
        img = np.array(img)
        return img

    def loadModel(self):
        '''
            load trained model from model.json and weights from model.h5
        '''
        with open(
                os.path.join(self.modelPath, 'model.json'), 'r') as json_file:
            loaded_model_json = json_file.read()

        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(os.path.join(self.modelPath, "model.h5"))
        loaded_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy'])
        print("Loaded model from disk")
        return loaded_model

    def predict(self, X):
        X = self.preProcessImage(X)
        print(np.shape(X))
        predictions = self.model.predict(X)
        predictions = [self.signNames[pred.index(
            max(pred))] for pred in predictions.tolist()]
        return predictions

if __name__ == '__main__':

    #Example 

    classifier = Classifier(imageShape=(28, 28, 1), nClasses=5)

    validation_file = './data/validate.p'

    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)

    X_valid = []
    #-------------------------------------------------------#
    for im_file in os.listdir(os.path.join('.','test')):
        print im_file 
        _im = cv2.imread(os.path.join('.','test',im_file))
        _im = np.array(_im)  
        X_valid.append(_im)  
    #-------------------------------------------------------#    

    # X_valid, y_valid = valid['features'], valid['labels']
    # X_valid = [_im]
    print np.shape(X_valid)
    try:
        for img in X_valid:
            signText = classifier.predict([img])[0]
            # img=np.resize(img,(100,100,1))
            font = cv2.FONT_HERSHEY_SIMPLEX
            print(signText)
            (x,y,c) = np.shape(img)
            # print x,y,c
            cv2.putText(img,signText,(int(0.10*x),int(0.20*y)), font, 0.5,(0,255,0),1,cv2.LINE_AA)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception as e:
        print e
        cv2.destroyAllWindows()
        print("bye\n\n")
        quit()
