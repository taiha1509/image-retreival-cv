from detector import FeatureDetector
import glob
import cv2
from annoy import AnnoyIndex
cd = FeatureDetector((8, 12, 3))
ann_file = '10k_hsv.ann'
hsv = AnnoyIndex(1440,'angular')
i=0
for imgPath in glob.glob("/home/son/Downloads/image_retrieval_platform/static/dataset/*"):
    imgID = imgPath[imgPath.rfind("/")+1:]
    image = cv2.imread(imgPath)
    feature = cd.describe(image)
    hsv.add_item(i, feature)
    with open('hsv_mapping.txt', 'a') as f:
        f.writelines(imgPath.split('/')[-1] + '\n')
    i+=1
hsv.build(20)
hsv.save(ann_file)