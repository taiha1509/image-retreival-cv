import numpy as np 
import glob
import cv2
import argparse
from gabor import GaborDescriptor
import os
params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}
gd = GaborDescriptor(params)
gaborKernels = gd.kernels()
output = open('gabor.csv',"w")

c = 1
for imagePath in glob.glob(os.path.join('/home/son/Downloads/image_retrieval_platform/static/corel-1k', '*.jpg')):
	imageId = imagePath[imagePath.rfind("/")+1:]
	image = cv2.imread(imagePath)

	features = gd.gaborHistogram(image,gaborKernels)
	features = [str(f) for f in features]
	print(c)
	c += 1 
	output.write("%s,%s\n" % (imageId,",".join(features)))

output.close()