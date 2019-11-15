from model import *
from data_handling import *

if __name__ == '__main__':
    (x_train, y_train) = DataHandling.load_dataset("../Data/train/", ["Parasitized", "Uninfected"])
    bd = BlobDetectionModel(x_train,y_train)
    print(bd.accuracy())