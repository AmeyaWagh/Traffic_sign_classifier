# Traffic sign classifier for Self driving car

LeNet implementation in Keras is used to build CNN. The CNN is trained on GTSRB dataset.
This classifier pipelined with image segmentation module such as Haar cascade classifier to detect traffic signs can be used for Self driving cars. 


### - ###

![alt text](test/test_50_1.jpg)

### Dependencies(Python 2.7):
* keras - Tensorflow backend
* os
* pickle
* numpy
* cv2
* csv
* pandas
* shutil
* matplotlib



### To download dataset:
run the following command in the project directory
```
$ ./downloadDataset.sh 
```
This will download the zip file and unzip in the current directory as GTSRB. The data formatting of images is given in Readme-Images.txt

### To generate pickle files using GTSRB dataset
change the parameters according to the requirement

dataGen = datasetGenerator(nClasses=5, nTrainSamples=800,
                               nTestSamples=100, nValidateSamples=100,imageSize=[28, 28])
* nClasses - no of classes
* nTrainSamples - no of Training samples
* nTestSamples - no of test samples
* nValidateSamples - no of validation samples
* imageSize - size of 2D image matrix to be resized into.  
                              
```
$ python datasetGenerator.py
```

This will generate the following files
```
├── info.txt
├── test.p
├── train.p
└── validate.p
```

### Training the network:
run CNNtrainer.py
```
$ python CNNtrainer.py
```
This will run for 10 epochs
The model is saved in model directory as 2 files
* model.h5 - which stores weights
* model.json - which stores the architecture of CNN


# To use this classifier:
An example is given in trafficSignalClassifier.py\
Add cropped images to test directory and run the following code
```
$ python trafficSignClassifier.py
```

### Complete directory structure
```
├── citation
├── CNNpredict.py
├── CNNtrainer.py
├── data
│   ├── info.txt
│   ├── test.p
│   ├── train.p
│   └── validate.p
├── datasetGenerator.py
├── downloadDataset.sh
├── GTSRB
│   ├── Final_Training
│   └── Readme-Images.txt
├── GTSRB_dataset.zip
├── model
│   ├── model.h5
│   └── model.json
├── README.md
├── signnames.csv
├── test
│   └── test_50_1.jpg
└── trafficSignClassifier.py

```
