import matplotlib.pyplot
import cv2
import os
def ConvertToFcninput(dirpath):
    onlyfiles = [os.path.join(dirpath,f) for f in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, f))]
    print(onlyfiles[0])
    if len(onlyfiles) == 0:
        print("There's no Input Images")
        exit()

    Img = matplotlib.pyplot.imread(onlyfiles[0])
    Img = cv2.resize(Img, dsize=(250,380))
    cv2.imwrite(os.path.join("zoo/data_zoo/fcn_timetableseg",onlyfiles[0].split('/')[-1]),cv2.cvtColor(Img,cv2.COLOR_RGB2BGR))
    return onlyfiles[0]
