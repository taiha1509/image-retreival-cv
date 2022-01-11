import os
import cv2
import time
import csv
from skimage.feature import hog
from retrieval.create_thumb_images import create_thumb_images
from flask import Flask, render_template, request, redirect, url_for, make_response,jsonify, flash
from retrieval.retrieval import load_model, load_data, extract_feature, load_query_image, sort_img, extract_feature_query
from hsv_detector import FeatureDetector
from hsv_searcher import Searcher 
from HOG.searcher import HogSearcher
from annoy import AnnoyIndex
from vgg16.feature_extractor import FeatureExtractor
import torch
import torch.nn as nn
import pickle
from torchvision import datasets, models, transforms
from pq.pq_computation import *
from numpy import array, empty, float32
from gabor.gabor import GaborDescriptor
from gabor.gaborSearcher import GaborSearcher

work_dir = '/home/son/Downloads/image_retrieval_platform/hsv'
f_csv= 'vgg16/mapping.txt'
hsv_mapping_file = 'hsv/hsv_mapping.txt'
vgg_ann = AnnoyIndex(4096, metric='angular')
vgg_ann.load('/home/son/Downloads/image_retrieval_platform/vgg16/20k_vgg.ann')
hsv_ann = AnnoyIndex(1440, metric='angular')
hsv_ann.load('/home/son/Downloads/image_retrieval_platform/hsv/10k_hsv.ann')
#i = 0
vectors = pickle.load(open("vgg16/son_vectors.pkl","rb"))
paths = pickle.load(open("vgg16/son_paths.pkl","rb"))
codeword = pickle.load(open('pq/codeword.pkl','rb'))
pqcode = pickle.load(open('pq/pqcode.pkl','rb'))
# Initialize the model
model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)

# Change the device to GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)
# print(new_model)
from tqdm import tqdm
import numpy as np

# Transform the image, so it becomes readable with the model
transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.CenterCrop(512),
  transforms.Resize(448),
  transforms.ToTensor()                              
])


with open(f_csv) as f:
    lines = f.readlines()
    vgg_image_mapping_path = []
    for line in lines:
        vgg_image_mapping_path.append(line[:-1])
with open(hsv_mapping_file) as f:
    lines = f.readlines()
    hsv_image_mapping_path = []
    for line in lines:
        hsv_image_mapping_path.append(line[:-1])
##########

##
# Create thumb images.
create_thumb_images(full_folder='./static/image_database/',
                    thumb_folder='./static/thumb_images/',
                    suffix='',
                    height=200,
                    del_former_thumb=True,
                    )

# Prepare data set.
data_loader = load_data(data_path='./static/image_database/',
                        batch_size=2,
                        shuffle=False,
                        transform='default',
                        )

# Prepare model.
model = load_model(pretrained_model='./retrieval/models/net_best.pth', use_gpu=True)

# Extract database features.
gallery_feature, image_paths = extract_feature(model=model, dataloaders=data_loader)

# Picture extension supported.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp', 'jpeg', 'JPEG'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# Set static file cache expiration time
# app.send_file_max_age_default = timedelta(seconds=1)




        
@app.route('/', methods=['POST', 'GET'])  # add route
def image_retrieval():
    basepath = os.path.dirname(__file__)    # current path
    upload_path = os.path.join(basepath, 'static/upload_image','query.jpg')

    if request.method == 'POST':
        if request.form['submit'] == 'upload':
            if len(request.files) == 0:
                return render_template('upload_finish.html', message='Please select a picture file!')
            else:
                f = request.files['picture']
         
                if not (f and allowed_file(f.filename)):
                    # return jsonify({"error": 1001, "msg": "Examine picture extension, only png, PNG, jpg, JPG, or bmp supported."})
                    return render_template('upload_finish.html', message='Examine picture extension, png、PNG、jpg、JPG、bmp support.')
                else:
                    f.save(upload_path)
                    # transform image format and name with opencv.
                    img = cv2.imread(upload_path)
                    cv2.imwrite(os.path.join(basepath, 'static/upload_image', 'query.jpg'), img)
             
                    return render_template('upload_finish.html', message='Upload successfully!')

        elif request.form['submit'] == 'retrieval':
            select = request.form.get('features_types')
            print(select)
            if (select == 'vgg_annoy'):
                start_time = time.time()
                img = cv2.imread('./static/upload_image/query.jpg')
                # Transform the image
                img = transform(img)
                # Reshape the image. PyTorch model reads 4-dimensional tensor
                # [batch_size, channels, width, height]
                img = img.reshape(1, 3, 448, 448)
                img = img.to(device)
                # We only extract features, so we don't need gradient
                K = 6
                with torch.no_grad():
                    # Extract the feature from the image
                    start_time = time.time()
                    feature = new_model(img)
                    
                    feature = np.array(feature.cpu().detach().numpy().reshape(-1))
                    feature = feature / np.linalg.norm(feature)
                    
                    matches = vgg_ann.get_nns_by_vector(feature, 6, include_distances=True)
                    print("Retrieval finished, cost {:3f} seconds.".format(time.time() - start_time))
                    tmb_images = []
                    for i in matches[0]:
                        path = './static/dataset/'+ str(vgg_image_mapping_path[i].split('/')[-1])
                        tmb_images.append(path)
                    similarity = matches[1]
                
            if (select == 'vgg'):
                start_time = time.time()
                img = cv2.imread('./static/upload_image/query.jpg')
                # Transform the image
                img = transform(img)
                # Reshape the image. PyTorch model reads 4-dimensional tensor
                # [batch_size, channels, width, height]
                img = img.reshape(1, 3, 448, 448)
                img = img.to(device)
                # We only extract features, so we don't need gradient
                K = 6
                with torch.no_grad():
                    # Extract the feature from the image
                    start_time = time.time()
                    feature = new_model(img)
                    
                    feature = np.array(feature.cpu().detach().numpy().reshape(-1))
                    feature = feature / np.linalg.norm(feature)
                    
                    distance = np.linalg.norm(vectors - feature, axis=1)
                    ids = np.argsort(distance)[:K]
                    similarity = np.sort(distance)[:K]
                    print("Retrieval finished, cost {:3f} seconds.".format(time.time() - start_time))
                tmb_images = [( './static/dataset/'+str(paths[i].split('/')[-1])) for i in ids]
            
            if (select == 'vgg_pq'):
                start_time = time.time()
                img = cv2.imread('./static/upload_image/query.jpg')
                # Transform the image
                img = transform(img)
                # Reshape the image. PyTorch model reads 4-dimensional tensor
                # [batch_size, channels, width, height]
                img = img.reshape(1, 3, 448, 448)
                img = img.to(device)
                # We only extract features, so we don't need gradient
                K = 6
                with torch.no_grad():
                    # Extract the feature from the image
                    
                    feature = new_model(img)
                    feature = np.array(feature.cpu().detach().numpy().reshape(-1))
                    query = feature / np.linalg.norm(feature)
                    N = 10790
                    
                    dist = search(codeword, pqcode, query)
                    dist_with_name = empty((N, 2), dtype='object')
                    for i in range(0, N, 1):
                        dist_with_name[i][0] = paths[i]
                        dist_with_name[i][1] = dist[i]
                    list_ids = array(dist_with_name[:, 1].argsort())
                    tmb_images = []
                    similarity = []
                    for index, id_ in enumerate(list_ids[:6]):
                    # get 10 images that most like the query image
                        tmb_images.append('./static/dataset/'+str(dist_with_name[id_][0].split('/')[-1]))
                        similarity.append(dist_with_name[id_][1])
                    
                    print(start_time)
                
            if (select == 'hsv'):
                start_time = time.time()
                # Query.
                #query_image = load_query_image('./static/upload_image/query.jpg')
                query_image = cv2.imread('./static/upload_image/query.jpg')
                # extract query features
                cd = FeatureDetector((8, 12, 3))
                queryFeatures = cd.describe(query_image)
                print(len(queryFeatures))
                #tim anh
                data = Searcher('hsv/hsv.csv')
                results = data.search(queryFeatures,8)
                #[(1.011323588790338e-15, '181006.jpg'), (7.1364918746922195, '275009.jpg'), (8.331189192505065, '280002.jpg')]
                print(results)
                similarity = []
                index = []
                for result in results[:6]:
                    sim, img = result
                    similarity.append(sim)
                    index.append('./static/corel-1k/'+ img)
                tmb_images = index

            if (select == 'gabor'):
                start_time = time.time()
                # Query.
                #query_image = load_query_image('./static/upload_image/query.jpg')
                query_image = cv2.imread('./static/upload_image/query.jpg')
                # extract query features
                params = {"theta":4, "frequency":(0,1,0.5,0.8), "sigma":(1,3),"n_slice":2}
                gd = GaborDescriptor(params)
                gaborKernels = gd.kernels()
                queryFeatures = gd.gaborHistogram(query_image,gaborKernels)
                print(len(queryFeatures))
                #tim anh
                data = GaborSearcher('gabor/gabor.csv')
                results = data.search(queryFeatures,8)
                #[(1.011323588790338e-15, '181006.jpg'), (7.1364918746922195, '275009.jpg'), (8.331189192505065, '280002.jpg')]
                print(results)
                similarity = []
                index = []
                for result in results[:6]:
                    sim, img = result
                    similarity.append(sim)
                    index.append('./static/corel-1k/'+ img)
                tmb_images = index
            
            if (select == 'rgb'):
                start_time = time.time()
                # Query.
                #query_image = load_query_image('./static/upload_image/query.jpg')
                query_image = cv2.imread('./static/upload_image/query.jpg')
                # extract query features
                cd = FeatureDetector((8, 12, 3))
                queryFeatures = cd.describe(query_image)
                print(len(queryFeatures))
                #tim anh
                data = Searcher('hsv/rgb.csv')
                results = data.search(queryFeatures,8)
                #[(1.011323588790338e-15, '181006.jpg'), (7.1364918746922195, '275009.jpg'), (8.331189192505065, '280002.jpg')]
                print(results)
                similarity = []
                index = []
                for result in results[:6]:
                    sim, img = result
                    similarity.append(sim)
                    index.append('./static/corel-1k/'+ img)
                tmb_images = index

            if (select == 'hog'):
                start_time = time.time()
                query_image = cv2.imread('./static/upload_image/query.jpg')
                width = 64
                height = 128
                dim = (width, height)
                resizeImage = cv2.resize(query_image, dim, interpolation = cv2.INTER_AREA)
                features, hog_image = hog(resizeImage, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis = -1)
                features = features

                searcher = HogSearcher("HOG/hog.csv")
                results = searcher.search(features)
                similarity = []
                index = []
                for result in results[:6]:
                    sim, img = result
                    similarity.append(sim)
                    index.append('./static/dataset/'+ img)
                tmb_images = index
                print(similarity, tmb_images)
            
            if (select == 'lbp'):
                start_time = time.time()
                query_image = cv2.imread('./static/upload_image/query.jpg')
                width = 64
                height = 128
                dim = (width, height)
                resizeImage = cv2.resize(query_image, dim, interpolation = cv2.INTER_AREA)
                features, hog_image = hog(resizeImage, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, channel_axis = -1)
                features = features.tolist()

                searcher = HogSearcher("HOG/hog.csv")
                results = searcher.search(features)
                similarity = []
                index = []
                for result in results[:6]:
                    sim, img = result
                    similarity.append(sim)
                    index.append('./static/corel-1k/'+ img)
                tmb_images = index
                print(similarity, tmb_images)
        
            if (select == 'hsv_annoy'):
                start_time = time.time()
                query_image = cv2.imread('./static/upload_image/query.jpg')
                cd = FeatureDetector((8, 12, 3))
                query_features = cd.describe(query_image)
                matches = hsv_ann.get_nns_by_vector(query_features, 6, include_distances=True)
                print(matches)
                tmb_images = []
                for i in matches[0]:
                    path = './static/dataset/'+ str(hsv_image_mapping_path[i])
                    tmb_images.append(path)
                similarity = matches[1]
                print(tmb_images, similarity)
                print('-----------')
            return render_template('retrieval.html', message="Retrieval finished, cost {:3f} seconds.".format(time.time() - start_time),
                sml1=similarity[0], sml2=similarity[1], sml3=similarity[2], sml4=similarity[3], sml5=similarity[4], sml6=similarity[5],
                img1_tmb=tmb_images[0], img2_tmb=tmb_images[1],img3_tmb=tmb_images[2],img4_tmb=tmb_images[3],img5_tmb=tmb_images[4],img6_tmb=tmb_images[5])
            
            

    return render_template('upload.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='127.0.0.1', port=8080, debug=True)