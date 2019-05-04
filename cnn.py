# -*- coding: utf-8 -*-
"""
Created on Fri May  3 22:19:11 2019

@author: gungor2
"""

# USAGE
# python cnn_regression.py --dataset Houses-dataset/Houses\ Dataset/

# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from scipy import signal

# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", type=str, required=True,
#	help="path to input dataset of house images")
#args = vars(ap.parse_args())
#
## construct the path to the input .txt file that contains information
## on each house in the dataset and then load the dataset
#print("[INFO] loading house attributes...")
#inputPath = os.path.sep.join(["D:\Dropbox\kaggle\house price\Houses-dataset\Houses Dataset", "HousesInfo.txt"])
#df = datasets.load_house_attributes(inputPath)
#
## load the house images and then scale the pixel intensities to the
## range [0, 1]
#print("[INFO] loading house images...")
#images = datasets.load_house_images(df, "D:\Dropbox\kaggle\house price\Houses-dataset\Houses Dataset")
#images = images / 255.0


train = pd.read_csv('./LANL-Earthquake-Prediction/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})



rows = 150000
segments = int(np.floor(train.shape[0] / rows))

y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])

fs = 4*10**6
print("[INFO] loading training images...")
images = []
for segment in range(segments):
    seg = train.iloc[segment*rows:segment*rows+rows]
    x = pd.Series(seg['acoustic_data'].values)
    f, t, Sxx = signal.spectrogram(x,fs)
    b = Sxx.reshape((129,669,1))
    images.append(b)

images = np.asarray(images)
maxx_val = np.max(images)

images= images/maxx_val


submission = pd.read_csv('./LANL-Earthquake-Prediction/sample_submission.csv', index_col='seg_id')

images_test = []
print("[INFO] loading testing images...")
for i, seg_id in enumerate((submission.index)):
    seg = pd.read_csv('./LANL-Earthquake-Prediction/test/' + seg_id + '.csv')
    
    x = pd.Series(seg['acoustic_data'].values)
    
    f, t, Sxx = signal.spectrogram(x,fs)
    
    b = Sxx.reshape((129,669,1))
    images_test.append(b)
    
    y = np.mean(seg['time_to_failure'].values)
    
    y_tr.loc[segment, 'time_to_failure'] = y

images_test = np.asarray(images_test)    

images_test= images_test/maxx_val


# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
split = train_test_split(y_tr,images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX,trainImagesX, testImagesX) = split

# find the largest house price in the training set and use it to
# scale our house prices to the range [0, 1] (will lead to better
# training and convergence)
maxPrice = trainAttrX['time_to_failure'].max()
trainY = trainAttrX['time_to_failure'] / maxPrice
testY = testAttrX['time_to_failure'] / maxPrice

# create our Convolutional Neural Network and then compile the model
# using mean absolute percentage error as our loss, implying that we
# seek to minimize the absolute percentage difference between our
# price *predictions* and the *actual prices*
model = models.create_cnn(64, 64, 1, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testImagesX)

# compute the difference between the *predicted* house prices and the
# *actual* house prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)


print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))


prediction = model.predict(images_test)

submission['time_to_failure'] = (prediction)
# submission['time_to_failure'] = prediction_lgb_stack
print(submission.head())
submission.to_csv('submission_cnn.csv')