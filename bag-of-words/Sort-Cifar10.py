import os
import csv
from shutil import copyfile

s = os.path.sep

dataFolder = 'DATA'
csvClassFile = 'trainLabels.csv'
trainSetFolder = 'train'
sortedSetFolder = 'train-50000'
fileType = '.png'

classFilePath = dataFolder+s+csvClassFile
if os.path.exists(classFilePath):
	rootDstFolder = dataFolder+s+sortedSetFolder
	rootSrcFolder = dataFolder+s+trainSetFolder
	#if not os.path.exists(fileDstFolder): os.makedirs(fileDstFolder)
	with open(classFilePath, 'rb') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for idx, row in enumerate(spamreader):
			#print ', '.join(row)
			#print row
			if idx > 0:
				fileSrcPath = rootSrcFolder+s+row[0]+fileType
				fileDstFolder = rootDstFolder+s+row[1]
				fileDstPath = fileDstFolder+s+row[0]+fileType
				if not os.path.exists(fileDstFolder): os.makedirs(fileDstFolder)
				if os.path.exists(fileSrcPath): copyfile(fileSrcPath, fileDstPath)
				#os.system("start "+fileSrcPath)
