
from detector import FeatureDetector
from searcher import Searcher 
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

cd = FeatureDetector((8, 12, 3))

fig = plt.figure(figsize=(10, 10))
axes = []

#read and display image query
query = cv2.imread("/media/son/01D7DCAD7B2DEFE0/20211/CV/prj/ImageRetrieval/CorelDB/woman/181006.jpg")
axes.append(fig.add_subplot(3,4, 1))
axes[-1].set_title("query")
plt.imshow(cv2.cvtColor(query, cv2.COLOR_BGR2RGB))
plt.axis('off')

#tinh dac trung anh query
queryFeatures = cd.describe(query)
print(len(queryFeatures))

#tim anh
data = Searcher('index.csv')
results = data.search(queryFeatures,10)

#hien thi anh tim kiem duoc
i=1
for score, images in results:

        print(images)
        img = mpimg.imread('corel-1k/'+images)
        
        axes.append(fig.add_subplot(3,4, i+1))

        axes[-1].set_title(round(score,2))
        plt.imshow(img)
        plt.axis('off')
        i=i+1
print(i)
plt.show()

        