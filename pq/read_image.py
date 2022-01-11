import csv
from pq_computation import train, search, encode
from numpy import array, empty, float32
import numpy as np
import time
import pickle

vector_file = "codeword.pkl"
path_file = "pqcode.pkl"
        
if __name__ == '__main__':
    # N tổng số lương ảnh
    # Nt số lượng ảnh để phân cụm
    # D số chiều của 1 vector
    #data, image_name = read_image()
    image_features = pickle.load(open("/home/son/Downloads/image_retrieval_platform/experiences/CV_prj/son_vectors.pkl","rb"))
    image_name = pickle.load(open("/home/son/Downloads/image_retrieval_platform/experiences/CV_prj/son_paths.pkl","rb"))
    data = np.array(image_features)
    N = data.shape[0] - 1
    Nt = data.shape[0] - 1
    D = data.shape[1] 
    vec = data
    vec_train = vec
    
    # hàng cuối cùng
    # a 128-dim query vector
    # số lương các sub vector được chia
    M = 8
    codeword = train(vec_train, M)
    #print(f'codeword: {codeword}')
    # short-code (pq-code) của toàn bộ tập dữ liệu, mỗi phần tử gồm 8 phần tử đại diện cho subvector tương ứng thuộc cụm nào
    pqcode = encode(codeword, vec)
    #print(f'pqcode: {pqcode}')

        
    pickle.dump(codeword, open(vector_file, "wb"))
    pickle.dump(pqcode, open(path_file, "wb"))
    start = time.time() * 1000
    # dist = search(codeword, pqcode, query)
    # end = time.time() * 1000
    # dist_with_name = empty((N, 2), dtype='object')
    # for i in range(0, N, 1):
    #     dist_with_name[i][0] = image_name[i]
    #     dist_with_name[i][1] = dist[i]
    # list_ids = array(dist_with_name[:, 1].argsort())
    # end = time.time() * 1000
    # # for index, id_ in enumerate(list_ids):
    # #     # get 10 images that most like the query image
    # #     if(index < 10):
    # #         print("Image name: {} -> Dist: {}".format(dist_with_name[id_][0], dist_with_name[id_][1]))
    # print(dist_with_name)
    # print(list_ids)