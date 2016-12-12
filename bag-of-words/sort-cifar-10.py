import os
import csv
from shutil import copyfile

s = os.path.sep

dataFolder = 'DATA'
csvClassFile = 'trainLabels.csv'
trainSetFolder = 'train-set-cifar-10'
sortedSetFolder = 'sorted-train-set-cifar-10'
fileType = '.png'

with open(dataFolder+s+csvClassFile, 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for idx, row in enumerate(spamreader):
        #print ', '.join(row)
        #print row
        if idx > 0:
            fileSrcPath = dataFolder+s+trainSetFolder+s+row[0]+fileType
            fileDstFolder = dataFolder+s+sortedSetFolder+s+row[1]
            fileDstPath = fileDstFolder+s+row[0]+fileType
            if not os.path.exists(fileDstFolder): os.makedirs(fileDstFolder)
            copyfile(fileSrcPath, fileDstPath)
            #os.system("start "+fileSrcPath)
