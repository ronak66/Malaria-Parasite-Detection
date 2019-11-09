import pandas as pd
import numpy as np
import os, glob
import cv2

class DataHandling:

    def generate_imagepath_test(self,base_path):
        df = pd.DataFrame()
        files = glob.glob(base_path+'/*.png')
        df["FILE"] = files
        return df.reset_index(drop=True)

    
    def read_image(img_path):
        image = cv2.resize(cv2.imread(img_path),(100,100))
        return image

    def load_dataset_test(df):
        x = []
        for index, row in df.iterrows():
            x.append(read_image(row["FILE"]))
        return x