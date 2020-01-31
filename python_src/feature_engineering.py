import cv2
import numpy as np
from sklearn.cluster import KMeans

from preprocessing import PreProcessing
class FeatureEngineering:

    def __init__(self):
        self.PP = PreProcessing()

    def find_contours_and_centers(self,img_input):
        try:
            img, contours_raw, hierarchy = cv2.findContours(img_input, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours_raw, hierarchy = cv2.findContours(img_input, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = [i for i in contours_raw]
        contour_centers = []
        
        for idx, c in enumerate(contours):
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except:
                cX = int(M["m10"] / (M["m00"] + 0.0001))
                cY = int(M["m01"] / (M["m00"] + 0.0001))
            samp_bounds = cv2.boundingRect(c)
            contour_centers.append(((cX,cY), samp_bounds))
        contour_centers = sorted(contour_centers, key=lambda x: x[0])

        return (contours, contour_centers)

    def sortresponse1(self,kp1):
        return kp1[0].response 

    def calculate_histogram(self,images, model,n,num_of_kp):

        feature_vectors=[]

        for image in images :

            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #SIFT extraction
            sift = cv2.xfeatures2d.SURF_create()
            kp, descriptors = sift.detectAndCompute(image,None)
            #classification of all descriptors in the model
            z = zip(kp,descriptors)
            z=list(z)
            z.sort(key=self.sortresponse1,reverse=True)
            descriptors = [x[1] for x in z]
            descriptors = descriptors[0:num_of_kp]
            predict_kmeans = model.predict(descriptors)
            #calculates the histogram
            hist, bin_edges = np.histogram(predict_kmeans, bins = n)
            #histogram is the feature vector
            feature_vectors.append(hist)

        feature_vectors=np.asarray(feature_vectors)

        return np.array(feature_vectors)

    def kmeans(self,x_train,num_of_clusters=10,num_of_kp=40):
        sift_keypoints = []
        for image in x_train :
            image = self.PP.change_contrast(image)
            image= self.PP.image_cleanup(image)
            image = self.PP.remove_noisy_regions(image)
    #         image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            sift = cv2.xfeatures2d.SURF_create()
            kp, descriptors = sift.detectAndCompute(image,None)
            z = zip(kp,descriptors)
            z=list(z)
            z.sort(key=self.sortresponse1,reverse=True)
            descriptors = [x[1] for x in z]
            descriptors = descriptors[0:num_of_kp]
            sift_keypoints.append(descriptors)


        sift_keypoints = np.concatenate(sift_keypoints, axis=0)
        kmeans = KMeans(n_clusters = num_of_clusters).fit(sift_keypoints)
        print(sift_keypoints.shape)
        return kmeans

    def surf_features(self,kmeans,x_train,x_test,num_of_clusters=10,num_of_kp=40):
        x_feat_train = self.calculate_histogram(x_train, kmeans,num_of_clusters,num_of_kp)
        x_feat_test = self.calculate_histogram(x_test, kmeans,num_of_clusters,num_of_kp)
        return (x_feat_train,x_feat_test)