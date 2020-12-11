
# image recognition and object detection
# on nvidia jetson nano
import jetson.inference
# let us interact with a screen, camera
import jetson.utils
import time
# opencv
import cv2
import numpy as np

# grabbing the time when we start program
# since fps will be calculated
timeStamp=time.time()
fpsFilt=0

# ssd - mobilenet network for object detection
# 50% is our threshold here, says found something after 50%
net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)
dispW=1280
dispH=720
flip=2 # depends on your cam
font=cv2.FONT_HERSHEY_SIMPLEX

# Gstreamer code for improvded Raspberry Pi Camera Quality
#camSet='nvarguscamerasrc wbmode=3 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=1 ! video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! videobalance contrast=1.5 brightness=-.2 saturation=1.2 ! appsink'
#cam=cv2.VideoCapture(camSet)
#cam=jetson.utils.gstCamera(dispW,dispH,'0')

# for USB webcam usage
cam=cv2.VideoCapture('/dev/video1')
cam.set(cv2.CAP_PROP_FRAME_WIDTH, dispW)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, dispH)

#cam=jetson.utils.gstCamera(dispW,dispH,'/dev/video1')
#display=jetson.utils.glDisplay()
#while display.IsOpen():

while True:
    #img, width, height= cam.CaptureRGBA() # using the jetson utils
    _,img = cam.read() # read cam with cv2
    height=img.shape[0]
    width=img.shape[1]

    # unsigned int to float 32
    frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGBA).astype(np.float32)
    # convert np to cuda to use the network
    frame=jetson.utils.cudaFromNumpy(frame)

    # the detection part
    detections=net.Detect(frame, width, height)
    for detect in detections:
        #print(detect) # gives you a list of information
        # ID of detected object
        ID=detect.ClassID
        # positions
        top=int(detect.Top)
        left=int(detect.Left)
        bottom=int(detect.Bottom)
        right=int(detect.Right)
        # actual the name of the class
        item=net.GetClassDesc(ID)
        #print(item,top,left,bottom,right)
        # create a box around it
        # (0,255,0) is the color; 1 is the thickness of the line
        cv2.rectangle(img,(left,top),(right,bottom),(0,255,0),1)
        cv2.putText(img,item,(left,top+20),font,.75,(0,0,255),2)

    #display.RenderOnce(img,width,height)

    # calculate the changing time (derivative of time) and fps
    dt=time.time()-timeStamp
    timeStamp=time.time()
    fps=1/dt
    # Low Pass Filer to smooth fps
    fpsFilt=.9*fpsFilt + .1*fps
    #print(str(round(fps,1))+' fps')
    cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),font,1,(0,0,255),2)
    cv2.imshow('detCam',img)
    cv2.moveWindow('detCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

# references:

# https://toptechboy.com/ai-on-the-jetson-nano-lesson-53-object-detection-and-recognition-in-opencv/
# https://www.youtube.com/watch?v=1cX0uxd--qo&list=PLGs0VKk2DiYxP-ElZ7-QXIERFFPkOuP4_&index=54
# https://www.youtube.com/watch?v=3mo6vlz0qGo&list=PLGs0VKk2DiYxP-ElZ7-QXIERFFPkOuP4_&index=53
# https://github.com/dusty-nv/jetson-inference



