import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pickle

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

    def train(self):
        tp = 0
        tn = 0
        total = 0
        fp = 0
        fn = 0
        for i in range(len(self.image_list)):
            im_processed = self.PP.preprocess(self.image_list[i])
            y = self.predict(self.image_list[i])
            if(y == 1 and self.image_labels[i] == 1):
                tp+=1
            if(y == 0 and self.image_labels[i] == 0):
                tn+=1
            elif(y == 1 and self.image_labels[i] == 0):
                fp+=1
            elif(y == 0 and self.image_labels[i] == 1):
                fn+=1
            total += 1
        con_mat = [[tp,fn],[fp,tn]]
        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        self.cm = con_mat
        self.total = total
        print("model trained")
    
    def accuracy(self):
        return ((self.tn + self.tp)/self.total)

    def confusion_matrix(self):
        return self.cm



class MachineLearningModel:

    def __init__(self,x_train,y_train,x_test,y_test):
        self.DH = DataHandling()
        self.PP = PreProcessing()
        self.FE = FeatureEngineering()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.get_all_features()
        print('-'*80)
        print(len(self.x_train))
        print(len(self.x_test))

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

    def train_random_forrest(self):
        self.rf = RandomForestClassifier(n_estimators = 1000)
        self.rf.fit(np.asarray(self.x_train), self.y_train)
        pickle.dump(self.rf, open('RandomForestPickle', 'wb'))

    def train_logistic_regression(self):
        self.lr = LogisticRegression()
        self.lr.fit(np.asarray(self.x_train), self.y_train)

    def train_svm(self):
        self.clf = svm.LinearSVC(multi_class='ovr',max_iter=100000)
        self.clf.fit(np.asarray(self.x_train), self.y_train)
    

    def accuracy(self,model_name):
        if(model_name == 'RandomForrest'):
            model = self.rf
        elif(model_name == 'LogisticRegression'):
            model = self.lr
        elif(model_name == 'SVM'):
            model = self.clf
        else:
            print('No Such model')
            return
        y_pred = model.predict(np.asarray(self.x_test))
        print(model_name)
        print("Length",len(y_pred),len(self.y_test))
        print(accuracy_score(y_pred, self.y_test))
        print(confusion_matrix(self.y_test,y_pred))
        return accuracy_score(y_pred, self.y_test)