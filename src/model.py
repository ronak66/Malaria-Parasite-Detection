import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_handling import DataHandling
from preprocessing import PreProcessing
from feature_engineering import FeatureEngineering


class BlobDetectionModel:

    def __init__(self,image_list,image_labels):
        self.DH = DataHandling()
        self.PP = PreProcessing()
        self.FE = FeatureEngineering()
        self.image_list = image_list
        self.image_labels = image_labels

    def predict(self,image):
        im_processed = self.PP.preprocess(image)
        try:
            conts, cents = self.FE.find_contours_and_centers(im_processed)
        except:
            print('corrupted image')
            return
        if(len(cents) >= 1):
            return 1
        return 0

    def accuracy(self):
        count = 0
        total = 0
        for i in range(len(self.image_list)):
            im_processed = self.PP.preprocess(self.image_list[i])
            y = self.predict(self.image_list[i])
            if(y == self.image_labels[i]):
                count+=1
            total += 1
        return(count/total)

class MachineLearningModel:

    def __init__(self,x_train,y_train,x_test,y_test):
        self.DH = DataHandling()
        self.PP = PreProcessing()
        self.FE = FeatureEngineering()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def get_all_features(self):
        kmeans = self.FE.kmeans(self.x_train)
        (x_feat_train,x_feat_test) = self.FE.surf_features(kmeans,self.x_train,self.x_test)
        x=[]
        y=[]
        for i in range(len(self.x_train)):
            im_processed = self.PP.preprocess(self.x_train[i])
            conts, cents = self.FE.find_contours_and_centers(im_processed)
            mm = list(x_feat_train[i])
            mm.append(len(cents))
            x.append(mm)
            
        for i in range(len(self.x_test)):
            im_processed = self.PP.preprocess(self.x_test[i])
            conts, cents = self.FE.find_contours_and_centers(im_processed)
            mm = list(x_feat_test[i])
            mm.append(len(cents))
            y.append(mm)
        self.x_train = x
        self.x_test = y

        return

    def train(self):
        self.get_all_features()
        self.rf = RandomForestClassifier(n_estimators = 1000)
        self.rf.fit(np.asarray(self.x_train), self.y_train)
    
    def accuracy(self):
        y_pred = self.rf.predict(np.asarray(self.x_test))
        return accuracy_score(y_pred, self.y_test)