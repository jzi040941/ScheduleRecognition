import numpy as np

class cropedImg:
    def __init__(self, img, pos):
        self.setImg(img)
        self.setImgpos(pos)
        
    def setImg(self, img):
        '''
        arg : img=()
        '''
        self.Img = img
    def setImgpos(self, pos):
        '''
        Arg : pos=(x,y,w,h)'
        '''
        self.x, self.y, self.w, self.h = pos
    def getUpperPos(self, x,y,w,h):
        return self.x+x,self.y+y,w,h

    def Pos(self, x,y):
        return x,y 
    
    def setboxesList(self, boxeslist):
        self.boxesList = boxeslist
        
    def setratioboxesList(self, ratioboxeslist):
        for item in ratioboxeslist:
            item[0] = item[0] * self.w + self.x
            item[1] = item[1] * self.h + self.y
            item[2] = item[2] * self.w
            item[3] = item[3] * self.h
        print(ratioboxeslist)
        self.boxesList = ratioboxeslist
        
class TimeTable:
    def __init__(self, schedulelist, words, time):
        self.scheduleInfos = []
        self.setSchedulePosList(schedulelist)
        self.setTesseractResult(words, time)
        
    def setshape(self, img):
        self.shape = img.shape

    def setSchedulePosList(self, schedulelist):
        self.widthperday = int(np.mean(schedulelist[:,2]-schedulelist[:,0]))
        self.schedulePosList = schedulelist

    def setTesseractResult(self, words, time):
        '''
        args:
        time type cropedImg
        '''
        self.splitedTime = time
        self.period = self.checkIsPeriod(words)
        if not self.period:
            self.settimezone(words, time)
    def getScheduleInfos(self):
        if self.period:
            print("isPeroid")
        else:
            for schedulePos in self.schedulePosList:
                self.scheduleInfos.append(self.getScheduleInfoWithTime(schedulePos));
        return self.scheduleInfos
    
    def getScheduleInfoWithTime(self, schedulePos):
        '''
        arg:
        schedulePos : [x1,y1,x2,y2] list
        
        output:
        day: 0~6 0->Monday 6->Sunday
        starttime: 0~24 float ex)12:30->12.5
        endtime: same as starttime
        '''
        day = self.calculateDay(schedulePos[0], schedulePos[2])
        starttime= self.calculateTime(schedulePos[1])
        endtime = self.calculateTime(schedulePos[3])
        return dict({'day':day, 'starttime':starttime, 'endtime':endtime}) 

    def calculateTime(self, y):
        return self.timezonestart + self.timeperpixel*(y - self.splitedTime.y)
    
    def calculateDay(self, x1, x2):
        return int(((x1+x2)/2)/self.widthperday)
    
    def settimezone(self, words, time):
        self.timezonestart = int(words[0])
        self.timeperpixel = 1/self.calculateAvgDistanceToNext(time.boxesList)

    def calculateAvgDistanceToNext(self, time):
        '''
        arg:
        time : 2d list [ [ x, y , w, h]...]
        
        !! caution !!
        output is not average for height in time list
        !!         !!
        
        output is average of distance for each center.y between center.y and next center.y'
        '''
        centerlist = convertxywhListToCenterList(time)
        res = 0
        count = len(centerlist)-1
        for i in range(count):
            res = res + (centerlist[i+1][1]-centerlist[i][1])
        return res/count
    def checkIsPeriod(self, words):
        return int(words[0])<4

def calculateCenter(x,y,w,h):
    return int(x+w/2),int(y+h/2)

def convertxywhListToCenterList(xywh_list):
    output = []
    for item in xywh_list:
        x,y,w,h = tuple(item)
        output.append(list(calculateCenter(x,y,w,h)))
    return output