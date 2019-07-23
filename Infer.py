from model.fcn_tensorflow import fcn_timetableseg
from utils import Input_Images
from utils import CropImage
import cv2

InputImageDir_Path = "zoo/data_zoo/Input_Images"
def main():
    rawimage_path = Input_Images.ConvertToFcninput(InputImageDir_Path)
    fcn_timetableseg_outputlabel = fcn_timetableseg.Inference()
    croped_timetable = CropImage.CropByFCNLabel(fcn_timetableseg_outputlabel,rawimage_path)
main()
