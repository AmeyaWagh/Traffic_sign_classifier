#!/usr/bin/python

import csv
import numpy as np
import cv2
import pickle
import os


class Parser:

    def __init__(self, nClasses=None, nTrainSmaples=None,
                 nTestSamples=None, nValidateSamples=None,
                 imageSize=[28, 28]):

        self.basePath = './GTSRB/Final_Training/Images/'
        self.destPath = './data'
        self.signNamesFile = './signnames.csv'

        self.nClasses = nClasses
        self.nTrainSmaples = nTrainSmaples
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
        if self.nTrainSmaples is not None:
            self.dataLen = self.nTrainSmaples

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
        for descriptor in imgData:
            print descriptor['Filename']
            imgPath = os.path.join(filePath, descriptor['Filename'])
            print imgPath
            im = cv2.imread(imgPath)
            print type(im)
            print np.shape(im)

            # crop Roi
            im = im[descriptor['Roi.X1']:descriptor['Roi.X2'],
                    descriptor['Roi.Y1']:descriptor['Roi.Y2']]

            # resize image
            newIm = []
            # for channel in range(3):
            # 	newIm[channel] = np.resize(im[:,:,channel],self.imageSize)
            print np.shape(im[:,:,0])
            print np.shape(im[:,:,1])
            print np.shape(im[:,:,2])

            cv2.imshow('image', im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def createDataSet(self):
        for everySign in self.signNameData:
            print everySign
            filePath = os.path.join(self.basePath, everySign[0])
            csvPath = os.path.join(filePath, 'GT-'+everySign[0]+'.csv')
            print filePath
            print csvPath
            imgData = self.readCSV(csvPath)
            print imgData
            self.getImage(imgData, filePath)


if __name__ == '__main__':
    parser = Parser(nClasses=1, nTrainSmaples=1)
    parser.createDataSet()
