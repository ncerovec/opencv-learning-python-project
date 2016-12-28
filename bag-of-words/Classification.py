#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import numpy as np
import os
from random import randint
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

# Paths
s = os.path.sep
visualize = True
dataFolder = 'DATA'
testSetFolder = 'test-200'
bowFile = 'BoW-train-50000.pkl'

# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load(bowFile)

# Get the path of the testing image(s) and store them in a list
test_path = dataFolder+s+testSetFolder

image_paths = []
testing_names = os.listdir(test_path)
for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = [os.path.join(dir, f) for f in os.listdir(dir)]
    image_paths += class_path
    
# Create feature extraction and keypoint detector objects
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")

# List where all the descriptors are stored
des_list = []
for image_path in image_paths:
    im = cv2.imread(image_path)
    if im is None:
        print "No such file {}\nCheck if the file exists".format(image_path)
        exit()
    kpts = fea_det.detect(im)
    kpts, des = des_ext.compute(im, kpts)

    if des is not None:
        des_list.append((image_path, des))
    else:
        #des_list.append((image_path, []))
        des_list.append((image_path, np.zeros((1, 128))))
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

# Calculate the histogram of features
test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
test_features = stdSlr.transform(test_features)

# Perform the prediction
numCorrect = 0
predictions = []
for image_path, i in zip(image_paths, clf.predict(test_features)):
    prediction = classes_names[i]
    predictions.append(prediction)
    #prediction = classes_names[randint(0,len(classes_names)-1)]   #Test random prediction accuracy
    #print prediction + " == " + os.path.basename(os.path.dirname(image_path))
    if os.path.basename(os.path.dirname(image_path)) == prediction:
        numCorrect += 1

print "Number of predictions: " + str(len(predictions))
print "Correct predictions: " + str(numCorrect)
print "Accuracy: " + str(float(numCorrect)/len(predictions)*100) + "%"

# Visualize the results, if "visualize" flag set to true by the user
if visualize:
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt ,cv2.FONT_HERSHEY_PLAIN, 0.5, [255, 0, 0], 1)
        cv2.imshow("Image", image)
        print prediction + " == " + os.path.basename(os.path.dirname(image_path))
        cv2.waitKey(1000)