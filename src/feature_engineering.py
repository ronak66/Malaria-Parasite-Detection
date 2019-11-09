import cv2

class FeatureEngineering:

    def find_contours_and_centers(img_input):
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