from skimage.feature import hog
import cv2
import glob
import csv

with open('10k_hog.csv', 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    for imagePath in glob.glob("/home/son/Downloads/image_retrieval_platform/static/corel-10k/*"):
        imageId = imagePath.split("/")[-1]
        width = 64
        height = 128
        dim = (width, height)
        image = cv2.imread(imagePath)
        resizeImage = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        df, hog_image = hog(resizeImage, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis = -1)
        df = df.tolist()
        df.insert(0,imageId)
        writer.writerow(df)
    f.close()