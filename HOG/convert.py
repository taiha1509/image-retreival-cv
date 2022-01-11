import pickle
import csv
# with open('/home/son/Downloads/image_retrieval_platform/HOG/hog.csv') as i:
#     features = []
#     paths = []
#     reader = csv.reader(i)
#     for row in reader:
#         paths.append(str(row[0]))
#         features.append([float(x) for x in row[1:]])
# vector_file = "hog_features.pkl"
# path_file = "hog_paths.pkl"
        
# pickle.dump(features, open(vector_file, "wb"))
# pickle.dump(paths, open(path_file, "wb"))
# i.close()

with open('/home/son/Downloads/image_retrieval_platform/HOG/hog.csv') as i:
    features = []
    paths = []
    reader = csv.reader(i)
    for row in reader:
        paths.append(str(row[0]))
        features.append([float(x) for x in row[1:]])
vector_file = "hog_features.pkl"
path_file = "hog_paths.pkl"
        
pickle.dump(features, open(vector_file, "wb"))
pickle.dump(paths, open(path_file, "wb"))
i.close()