import cv2

from searcher import Searcher
from extractor import createLbpHistogram

image = cv2.imread("100000.jpg", 0)
image = cv2.resize(image, (500, 500))
features = createLbpHistogram(image)

searcher = Searcher("index.csv")
results = searcher.search(features)
print(results)