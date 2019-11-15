from model import *
from data_handling import *

if __name__ == '__main__':
    (x_train, y_train) = DataHandling.load_dataset("../Data/train/", ["Parasitized", "Uninfected"])
    (x_test, y_test) = DataHandling.load_dataset("../Data/test/", ["Parasitized", "Uninfected"])


    # bd = BlobDetectionModel(x_train,y_train)
    # print(bd.accuracy())

    ml = MachineLearningModel(x_train,y_train,x_test,y_test)
    ml.train()
    print(ml.accuracy())