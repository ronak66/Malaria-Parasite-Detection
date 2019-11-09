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
            im_processed = PP.preprocess(self.dataset[i])
            y = self.predict(self.dateset[i])
            if(y == self.image_labels[i]):
                count+=1
            total += 1
        print(count/total)
