from model.fcn_tensorflow import fcn_timetableseg,fcn_daytimetbseg
from utils import Input_Images,CropImage,finddaytime
import cv2
from model.keras_retinanet import detectScheduled
from model.Tesseract_With_attention import DeepPos_V2_xywh
InputImageDir_Path = "zoo/data_zoo/Input_Images"
def main():
    rawimage_path = Input_Images.ConvertToFcninput(InputImageDir_Path)
    fcn_timetableseg_outputlabel = fcn_timetableseg.Inference()
    croped_timetable = CropImage.CropByFCNLabel(fcn_timetableseg_outputlabel,rawimage_path)
    ScheduleList = detectScheduled.Inference(croped_timetable)
    fcn_daytimetbseg_outputlabel = fcn_daytimetbseg.Inference(croped_timetable)
    table,day,time = CropImage.CropByDaytimetbFCNLabel(fcn_daytimetbseg_outputlabel, croped_timetable)
    words, pos = DeepPos_V2_xywh.process(time.Img)
    time.setratioboxesList(pos)
    timetable = finddaytime.TimeTable(ScheduleList, words, time)
    scheduleInfos = timetable.getScheduleInfos()
    print(scheduleInfos)
main()
