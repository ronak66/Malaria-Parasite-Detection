from model import *
from data_handling import *
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    images = 10000
    (x_train, y_train) = DataHandling.load_dataset("../Data/train/", ["Parasitized", "Uninfected"],amount=images)
    (x_test, y_test) = DataHandling.load_dataset("../Data/test/", ["Parasitized", "Uninfected"],amount=images)


    bd = BlobDetectionModel(x_test,y_test)
    bd.train()
    print(bd.accuracy())
    cm = bd.confusion_matrix()
    print(cm)

    ml = MachineLearningModel(x_train,y_train,x_test,y_test)
    ml.train_random_forrest()
    ml.accuracy(model_name='RandomForrest')
    ml.train_logistic_regression()
    ml.accuracy(model_name='LogisticRegression')
    ml.train_svm()
    ml.accuracy(model_name='SVM')
    
    