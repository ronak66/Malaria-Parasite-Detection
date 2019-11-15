import pandas as pd
import numpy as np
import os, glob
import cv2

class DataHandling:

    @staticmethod
    def load_dataset(path, classes,start=0, amount=500):
        count=0
        class_images = []
        for cls in classes :
            cls_imgs = []
            amounts=amount
            img_names = os.listdir(path + cls + "/")
            for i in range(start,len(img_names)) :
                if(amounts==0):
                    break
                amounts-=1
                try:
                    cls_imgs.append(cv2.resize(cv2.imread(path + cls + "/" + img_names[i]), (125, 125)))
                except Exception as e:
                    count+=1
            class_images.append(np.array(cls_imgs))
        y_train = np.concatenate([np.ones(len(class_images[0])), np.zeros(len(class_images[1]))])
        x_train = np.concatenate(class_images)
        print(str(count)+' images encountered error')
        return (x_train,y_train)