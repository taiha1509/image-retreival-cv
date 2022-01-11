from skimage.feature import hog
import cv2

from searcher import HogSearcher

width = 64
height = 128
dim = (width, height)
image = cv2.imread("/media/son/01D7DCAD7B2DEFE0/20211/CV/prj/final/181006.jpg")
resizeImage = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
features, hog_image = hog(resizeImage, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis = -1)
features = features.tolist()

searcher = HogSearcher("/media/son/01D7DCAD7B2DEFE0/20211/CV/prj/final/image_retrieval_platform/HOG/hog.csv")
results = searcher.search(features)
similarity = []
index = []
for result in results[:6]:
    sim, img = result
    similarity.append(sim)
    index.append('./static/corel-1k/'+ img)
tmb_images = index
print(tmb_images)