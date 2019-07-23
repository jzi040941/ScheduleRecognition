import matplotlib.pyplot
import cv2
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
