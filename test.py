import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage import feature

def load_dataset(path, classes):
    class_images = []
    for cls in classes :
        i=1
        cls_imgs = []
        img_names = os.listdir(path + cls + "/")
        for img_name in img_names :
            if(i==0):
                break
            i-=1
            cls_imgs.append(cv2.resize(cv2.imread(path + cls + "/" + img_name), (300, 200)))
        class_images.append(np.array(cls_imgs))
    return class_images

data = load_dataset("./Data/train/", ["Parasitized", "Uninfected"])
test_data = load_dataset("./Data/test/", ["Parasitized", "Uninfected"])

y_train = np.concatenate([np.ones(len(data[0])), np.zeros(len(data[1]))])
y_test = np.concatenate([np.ones(len(test_data[0])), np.zeros(len(test_data[1]))])

x_train = np.concatenate(data)
x_test = np.concatenate(test_data)

#shuffle
p = np.random.permutation(len(x_train))
x_train = x_train[p]
y_train = y_train[p]

sift_keypoints = []
i=0
print('Start'+'-'*80)
for image in x_train :
    print(str(i)+'-'*80)
    i+=1
    image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    if(i==1):
        for j in image:
            print(j,end=' ')
        print('='*90)
        print(len(image))
        print(len(image[0]))
        print(image.shape)
        break
    sift = cv2.xfeatures2d.SIFT_create()
    kp, descriptors = sift.detectAndCompute(image,None)

    sift_keypoints.append(descriptors)

print('End'+'-'*80)
sift_keypoints = np.concatenate(sift_keypoints, axis=0)
kmeans = KMeans(n_clusters = 10).fit(sift_keypoints)
#print(sift_keypoints[0])
# print(sift_keypoints.shape)
def calculate_histogram(images, model):

    feature_vectors=[]

    for image in images :

        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #SIFT extraction
        sift = cv2.xfeatures2d.SURF_create()
        kp, descriptors = sift.detectAndCompute(image,None)
        #classification of all descriptors in the model
        predict_kmeans = model.predict(descriptors)
        #calculates the histogram
        hist, bin_edges = np.histogram(predict_kmeans, bins = 10)
        #histogram is the feature vector
        feature_vectors.append(hist)

    feature_vectors=np.asarray(feature_vectors)

    return np.array(feature_vectors)




# x_feat_train = calculate_histogram(x_train, kmeans)
# x_feat_test = calculate_histogram(x_test, kmeans)

# clf = svm.LinearSVC(multi_class='ovr')
# clf.fit(x_feat_train, y_train)

# y_pred = clf.predict(x_feat_test)
# print(y_pred)
# print(accuracy_score(y_pred, y_test))

# image = cv2.cvtColor(data[0],cv2.COLOR_BGR2GRAY)
# #SIFT extraction
# sift = cv2.xfeatures2d.SURF_create()
# kp, descriptors = sift.detectAndCompute(image,None)

# print(x_feat_train.shape)
# print(x_feat_train)
