#!/usr/bin/python

import csv
import numpy as np
import cv2
import pickle
import os
from random import shuffle
import traceback
import shutil


class Parser:

    def __init__(self, nClasses=None, nTrainSamples=None,
                 nTestSamples=None, nValidateSamples=None,
                 imageSize=[28, 28]):

        self.basePath = './GTSRB/Final_Training/Images/'
        self.destPath = './data'
        self.signNamesFile = './signnames.csv'

        self.nClasses = nClasses
        self.nTrainSamples = nTrainSamples
        self.nTestSamples = nTestSamples
        self.nValidateSamples = nValidateSamples
        self.imageSize = tuple(imageSize)

        self.Xtrain = []
        self.Ytrain = []

        self.Xtest = []
        self.Ytest = []

        self.Xvalidate = []
        self.Yvalidate = []

        self.dirNameFormatter = lambda x: "%05d" % (int(x))
        self.signNameData = self.readSignNameCSV()

        self.imageFileFormatter = lambda x: {'Filename': x[0],
                                             'Width': int(x[1]),
                                             'Height': int(x[2]),
                                             'Roi.X1': int(x[3]),
                                             'Roi.Y1': int(x[4]),
                                             'Roi.X2': int(x[5]),
                                             'Roi.Y2': int(x[6]),
                                             'ClassId': int(x[7])}
        self.dataLen = None
        if self.nTrainSamples is not None:
            self.dataLen = self.nTrainSamples

        if self.nTestSamples is not None:
            self.dataLen += self.nTestSamples

        if self.nValidateSamples is not None:
            self.dataLen += self.nValidateSamples

    def readSignNameCSV(self):
        with open(self.signNamesFile, 'r') as fp:
            data = list(csv.DictReader(fp))
        # convert dictionary to list and format file names
        data = [
            (self.dirNameFormatter(everyData['ClassId']),
             everyData['SignName']) for everyData in data]

        if self.nClasses is not None:
            data = data[0:self.nClasses]
        return data

    def readCSV(self, csvPath):
        with open(csvPath, 'r') as fp:
            data = list(csv.reader(fp))

        # remove the titles of the table
        data.pop(0)

        if self.dataLen is not None:
            data = data[0:self.dataLen]

        data = [
            self.imageFileFormatter(
                everyData[0].split(';')) for everyData in data
        ]
        return data

    def getImage(self, imgData, filePath):
        X_dataset = []
        for descriptor in imgData:
            imgPath = os.path.join(filePath, descriptor['Filename'])
            im = cv2.imread(imgPath)

            # crop Roi
            im = im[descriptor['Roi.X1']:descriptor['Roi.X2'],
                    descriptor['Roi.Y1']:descriptor['Roi.Y2']]

            # resize image
            im = cv2.resize(im,self.imageSize)
            # print np.shape(im)
            X_dataset.append((im,descriptor['ClassId']))
        return X_dataset     

    def createDataSet(self):
        images=[]
        for everySign in self.signNameData:
            # print everySign
            filePath = os.path.join(self.basePath, everySign[0])
            csvPath = os.path.join(filePath, 'GT-'+everySign[0]+'.csv')
            imgData = self.readCSV(csvPath)
            # print imgData
            newimages = self.getImage(imgData, filePath)
            images.extend(newimages)
            # for im in newimages:
            #     cv2.imshow('image', im[0])
            #     print im[1]
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

        shuffle(images)

        print "length of dataset:", len(images)
        # for im in images:
        #     print im[1]


        if self.nTrainSamples is not None:
            self.Xtrain = [image[0] for image in images[0:self.nTrainSamples]]    
            self.Ytrain = [image[1] for image in images[0:self.nTrainSamples]]

            if self.nTestSamples is not None:
                self.Xtest =  [image[0] for image in images[self.nTrainSamples+1:self.nTestSamples]]
                self.Ytest =  [image[1] for image in images[self.nTrainSamples+1:self.nTestSamples]]

            if self.nValidateSamples is not None:
                self.Xtest =  [image[0] for image in images[self.nTestSamples+1:self.nValidateSamples]]
                self.Ytest =  [image[1] for image in images[self.nTestSamples+1:self.nValidateSamples]]
        
        else:
            self.Xtrain = [image[0] for image in images]   
            self.Ytrain = [image[1] for image in images]
        
        print "shape of Xtrain",np.shape(self.Xtrain)
        print "shape of Ytrain",np.shape(self.Ytrain)

        print "shape of Xtest",np.shape(self.Xtest)
        print "shape of Ytest",np.shape(self.Ytest)

        print "shape of Xvalidate",np.shape(self.Xvalidate)
        print "shape of Yvalidate",np.shape(self.Yvalidate)
        self.createFiles()


    def createFiles(self):
        # remove previous files
        if os.path.isdir(self.destPath):
            shutil.rmtree(self.destPath)

        #create new directory
        os.mkdir(self.destPath)

        try:
            trainData = {'features':self.Xtrain,'labels':self.Ytrain}
            pickle.dump( trainData, open( os.path.join(self.destPath,"train.p"), "wb" ) )

            if self.nTestSamples is not None:
                testData = {'features':self.Xtest,'labels':self.Ytest}
                pickle.dump( testData, open( os.path.join(self.destPath,"test.p"), "wb" ) )

            if self.nValidateSamples is not None:
                validateData = {'features':self.Xvalidate,'labels':self.Yvalidate}
                pickle.dump( validateData, open( os.path.join(self.destPath,"validate.p"), "wb" ) )

            print "Files created"
        
        except Exception as e:
            traceback.print_exc(e)
            print "Could not create files"    

if __name__ == '__main__':
    parser = Parser(nClasses=5, nTrainSamples=800, nTestSamples=100,nValidateSamples=100)
    parser.createDataSet()
