import cv2

class PreProcessing:

    def change_contrast(image,alpha = 1.5,beta=25):
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    def image_cleanup(image):
        blurred = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)
        thresh = cv2.threshold(blurred, 186, 250, cv2.THRESH_BINARY)[1]
        return thresh

    def remove_noisy_regions(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        mask = cv2.erode(gray, element, iterations = 50)
        mask = cv2.dilate(mask, element, iterations = 50)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
        mask = cv2.erode(gray, element, iterations = 1)
        mask = cv2.dilate(mask, element, iterations = 1)
        mask = cv2.erode(mask, element)
        gray = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        return gray

    def preprocess(image):
        i1 = change_contrast(image,1.97,17)
        i2 = image_cleanup(i1)
        img = remove_noisy_regions(i2)
        h,s,v = cv2.split(img)
        ab = cv2.subtract(h,s)
        return ab