#!/usr/bin/python

import csv
import numpy as np
import cv2
import pickle
import os
from random import shuffle
import traceback
import shutil
import time
import datetime


class datasetGenerator:

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

        self.info = ""

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
            im = cv2.resize(im, self.imageSize)
            # print np.shape(im)
            X_dataset.append((im, descriptor['ClassId']))
        return X_dataset

    def createDataSet(self):
        images = []
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
            print "creating training samples"
            offset = 0
            # print offset
            self.Xtrain = [image[0] for image in images[
                offset:offset+self.nTrainSamples]]
            self.Ytrain = [image[1] for image in images[
                offset:offset+self.nTrainSamples]]

            if self.nTestSamples is not None:
                print "creating test samples"
                offset += self.nTrainSamples
                # print offset
                self.Xtest = [image[0] for image in images[
                    offset+1:offset+self.nTestSamples]]
                self.Ytest = [image[1] for image in images[
                    offset+1:offset+self.nTestSamples]]

            if self.nValidateSamples is not None:
                print "creating validation samples"
                if self.nTestSamples is not None:
                    offset += self.nTestSamples
                else:
                    offset += self.nTrainSamples
                # print offset

                self.Xvalidate = [image[0] for image in images[
                    offset+1:offset+self.nValidateSamples]]
                self.Yvalidate = [image[1] for image in images[
                    offset+1:offset+self.nValidateSamples]]

        else:
            self.Xtrain = [image[0] for image in images]
            self.Ytrain = [image[1] for image in images]

        print "shape of Xtrain", np.shape(self.Xtrain)
        print "shape of Ytrain", np.shape(self.Ytrain)
        print "shape of Xtest", np.shape(self.Xtest)
        print "shape of Ytest", np.shape(self.Ytest)
        print "shape of Xvalidate", np.shape(self.Xvalidate)
        print "shape of Yvalidate", np.shape(self.Yvalidate)
        self.createFiles()

    def generateInfo(self):
        nl = "\n"
        self.info = ""
        self.info += "nClasses = {}".format(self.nClasses)+nl
        self.info += "total samples = {}".format(self.dataLen)+nl
        self.info += "nTrainSamples = {}".format(self.nTrainSamples)+nl
        self.info += "nTestSamples = {}".format(self.nTestSamples)+nl
        self.info += "nValidateSamples = {}".format(
            self.nValidateSamples)+nl+nl

        self.info += "shape of Xtrain = {}".format(np.shape(self.Xtrain))+nl
        self.info += "shape of Ytrain = {}".format(np.shape(self.Ytrain))+nl+nl

        self.info += "shape of Xtest = {}".format(np.shape(self.Xtest))+nl
        self.info += "shape of Ytest = {}".format(np.shape(self.Ytest))+nl+nl

        self.info += "shape of Xvalidate = {}".format(
            np.shape(self.Xvalidate))+nl
        self.info += "shape of Yvalidate = {}".format(
            np.shape(self.Yvalidate))+nl+nl

        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        self.info += "File created on {}".format(st)

        with open(os.path.join(self.destPath, "info.txt"), "w") as fp:
            fp.write(self.info)

    def createFiles(self):
        # remove previous files
        if os.path.isdir(self.destPath):
            shutil.rmtree(self.destPath)

        # create new directory
        os.mkdir(self.destPath)

        try:
            trainData = {'features': self.Xtrain, 'labels': self.Ytrain}
            pickle.dump(trainData, open(
                os.path.join(self.destPath, "train.p"), "wb"))

            if self.nTestSamples is not None:
                testData = {'features': self.Xtest, 'labels': self.Ytest}
                pickle.dump(testData, open(
                    os.path.join(self.destPath, "test.p"), "wb"))

            if self.nValidateSamples is not None:
                validateData = {'features': self.Xvalidate,
                                'labels': self.Yvalidate}
                pickle.dump(validateData, open(
                    os.path.join(self.destPath, "validate.p"), "wb"))

            self.generateInfo()
            print "Files created"

        except Exception as e:
            traceback.print_exc(e)
            print "Could not create files"

if __name__ == '__main__':
    dataGen = datasetGenerator(nClasses=43, nTrainSamples=31367,
                               nTestSamples=3920, nValidateSamples=3920)
    #dataGen = datasetGenerator()
    dataGen.createDataSet()
