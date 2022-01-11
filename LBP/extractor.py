import cv2 as cv
import numpy as np
import csv
import glob

def prepareLbpHistogramData():
	print("Preparing LBP data...")
	path = "images"

	with open('lbp.csv', 'w', newline='', encoding='utf-8-sig') as f:
		writer = csv.writer(f)
		for imgPath in glob.glob("./static/corel-1k/*"):
			saveLbpHistogram(imgPath, writer)
		f.close()

def saveLbpHistogram(imgPath, wr):
	img = cv.imread(imgPath, 0)
	img = cv.resize(img, (500, 500))
	histogram = createLbpHistogram(img)
	imgName = imgPath.split("/")[-1]
	histogram.insert(0,imgName)
	wr.writerow(histogram)

def hasLessThanTwoTransitions(no):
	binary = '{0:08b}'.format(no)
	transitions = 0
	for i in range(0, len(binary)-1):
		if binary[i] != binary[i+1]:
			transitions += 1
	if transitions <= 2:
		return True
	else:
		return False

def createLbpHistogram(img):
	row, col = img.shape
	lbpValues = []
	for i in range(1, row-1):
		for j in range(1, col-1):
			pattern = 0
			if (img[i][j] > img[i-1][j-1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i][j-1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i+1][j-1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i+1][j]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i+1][j+1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i][j+1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i-1][j+1]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			if (img[i][j] > img[i-1][j]):
				pattern = pattern << 1
			else:
				pattern = pattern << 1
				pattern += 1
			lbpValues.append(pattern)
	lbpHistogram = np.zeros(256, dtype=int)
	for i in range(0, len(lbpValues)):
		if hasLessThanTwoTransitions(lbpValues[i]):
			lbpHistogram[lbpValues[i]] += 1
	lbpHistogram = normaliseHistogram(lbpHistogram, row*col)
	return lbpHistogram

def normaliseHistogram(histogram, size):
	normalisedHistogram = []
	for i in range(0, len(histogram)):
		normalisedHistogram.append(histogram[i] / float(size))
	return normalisedHistogram

def main():
	prepareLbpHistogramData()
	input()
main()
