#!/usr/bin/python

import csv
import numpy as np
import cv2
import pickle
import os


class Parser:

	def __init__(self, nClasses=None, nTrainSmaples=None,
				 nTestSamples=None, nValidateSamples=None):

		self.basePath = './GTSRB/Final_Training/Images/'
		self.destPath = './data'
		self.signNamesFile = './signnames.csv'

		self.nClasses = nClasses
		self.nTrainSmaples = nTrainSmaples
		self.nTestSamples = nTestSamples
		self.nValidateSamples = nValidateSamples

		self.Xtrain = []
		self.Ytrain = []

		self.Xtest = []
		self.Ytest = []

		self.Xvalidate = []
		self.Yvalidate = []

		self.dirNameFormatter = lambda x: "%05d" % (int(x))
		self.signNameData = self.readSignNameCSV()

		self.imageFileFormatter = lambda x: {'Filename': x[0],
											 'Width': x[1],
											 'Height': x[2],
											 'Roi.X1': x[3],
											 'Roi.Y1': x[4],
											 'Roi.X2': x[5],
											 'Roi.Y2': x[6],
											 'ClassId': x[7]}
		self.dataLen=None
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

	
	def createDataSet(self):
		for everySign in self.signNameData:
			print everySign
			filePath = os.path.join(self.basePath, everySign[0])
			csvPath = os.path.join(filePath, 'GT-'+everySign[0]+'.csv')
			print filePath
			print csvPath
			imgData = self.readCSV(csvPath)
			print imgData



if __name__ == '__main__':
	parser = Parser(nClasses=1,nTrainSmaples=1)
	parser.createDataSet()
