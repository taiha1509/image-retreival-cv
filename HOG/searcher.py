import numpy as np 
import csv
from scipy.spatial import distance
import pickle

class HogSearcher:
	def __init__(self,indexPath):
		self.indexPath = indexPath

	def search(self, queryFeatures, limit=10):
		results = {}
		features = pickle.load(open("/home/son/Downloads/image_retrieval_platform/HOG/hog_features.pkl","rb"))
		paths = pickle.load(open("/home/son/Downloads/image_retrieval_platform/HOG/hog_paths.pkl","rb"))
		i=0
		for row in features:
			features = [float(x) for x in row]
			d = self.distance(features,queryFeatures)
			results[paths[i]] = d
			i+=1
		
		results = sorted([(v,k) for (k,v) in results.items()])

		return results[:limit]

	def _gsearch(self, queryFeatures, limit=10):
		results = {}
		with open(self.indexPath) as i:
			reader = csv.reader(i)

			for row in reader:
				features = [float(x) for x in row[1:]]
				error = np.sum((queryFeatures - features)**2)
				results[row[0]] = error

		i.close()

		results = sorted([(v,k) for (k,v) in results.items()])
		return results[:limit]

	def distance(self, histA, histB):
		d = distance.correlation(histA, histB)
		return d
