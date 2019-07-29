import matplotlib.pyplot
import cv2
import numpy as np
def CropByFCNLabel(Gt,rawimage_path):
    raw = matplotlib.pyplot.imread(rawimage_path)
    (rawy, rawx, c) = raw.shape
    reshaped_Gt = cv2.resize(Gt, (rawx, rawy))
    if len(Gt.shape)==3:
        reshaped_Gt = cv2.cvtColor(reshaped_Gt, cv2.COLOR_BGR2GRAY)
    #image, contours, hierarchy = cv2.findContours(reshaped_Gt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(reshaped_Gt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key = cv2.contourArea)
    
    x,y,w,h = cv2.boundingRect(c)
    cropedRaw = raw[y:y+h, x:x+w]
    return cropedRaw

def CropByDaytimetbFCNLabel(Gt, raw):
    (rawy, rawx, c) = raw.shape
    reshaped_Gt = Gt
    if len(Gt.shape)==3:
        reshaped_Gt = cv2.cvtColor(reshaped_Gt, cv2.COLOR_BGR2GRAY)
    #image, contours, hierarchy = cv2.findContours(reshaped_Gt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cropedimgs = []
    for i in range(1,4):
        typed_Gt = (reshaped_Gt==i).astype(np.uint8)
        typed_Gt = cv2.resize(typed_Gt, (rawx, rawy))
        #ret, thresh = cv2.threshold(typed_Gt, 127, 255, 0)
        #if (i==1):
        #    cv2.imwrite('1.jpg', thresh)
        contours, hierarchy = cv2.findContours(typed_Gt,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            cropedimgs.append(None)
        else:
            c = max(contours, key = cv2.contourArea)
        
            x,y,w,h = cv2.boundingRect(c)
            cropedRaw = raw[y:y+h, x:x+w]
            cropedimgs.append(cropedRaw)
        
    return tuple(cropedimgs)
