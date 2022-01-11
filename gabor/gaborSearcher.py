import numpy as np 
import csv
from scipy import spatial
from sklearn.metrics import mean_squared_error
class GaborSearcher:
	def __init__(self,indexPath):
		self.indexPath = indexPath

	def search(self, queryFeatures, limit=10):
		results = {}

		with open(self.indexPath) as i:
			reader = csv.reader(i)

			for row in reader:
				features = [float(x) for x in row[1:]]
				d = self.distance(features,queryFeatures)
				results[row[0]] = d

		i.close()

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

	def distance(self, vecA, vecB):
		# result = spatial.distance.cosine(vecA, vecB)
		result = mean_squared_error(vecA, vecB)
		return result