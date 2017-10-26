#!/usr/bin/python
import csv
import numpy as np 
import cv2
import pickle

class Parser:

    def __init__(self, nClasses=None,nTrainSmaples=None,nTestSamples=None,nValidateSamples=None):
        self.basePath = './GTSRB/Final_Training/Images/'
        self.destPath = './data'
        self.signNamesFile = './signnames.csv'
        
        self.nClasses = nClasses
        self.nTrainSmaples = nTrainSmaples
        self.nTestSamples = nTestSamples
        self.nValidateSamples = nValidateSamples

        self.dirNameFormatter = lambda x: "%05d" % (int(x))
        self.signNameData = self.readSignNameCSV()

    def readSignNameCSV(self):
        with open(self.signNamesFile, 'r') as fp:
            data = list(csv.DictReader(fp))
        #convert dictionary to list and format file names 
        data = [
            (self.dirNameFormatter(everyData['ClassId']),
             everyData['SignName']) for everyData in data]

        if self.nClasses is not None:
        	data = data[0:self.nClasses]     
        return data


if __name__ == '__main__':
    parser = Parser(nClasses=5)
    print parser.signNameData
