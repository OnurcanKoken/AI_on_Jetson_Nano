# https://toptechboy.com/ai-on-the-jetson-nano-lesson-50-introduction-to-deep-learning-and-deep-neural-networks/

# image recognition and object detection
# on nvidia jetson nano
import jetson.inference
# let us interact with a screen, camera
import jetson.utils
import time
# opencv
import cv2
import numpy as np

# suitable width and height for raspi cam and usb cam
width=1280
height=720
# for USB cam, uncomment the line below
cam=jetson.utils.gstCamera(width,height,'/dev/video1') # might need to try video0 or 2
# for raspberry pi cam, uncomment the line below
#cam=jetson.utils.gstCamera(width,height,'0') # might need to try video0 or 2

# create neural network - imageNet - googlenet
# check the github repo for more info about the networks instead of using the googlenet
# https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md
net=jetson.inference.imageNet('googlenet')

# grabbing what the clock time is
timeMark=time.time()
fpsFilter=0
timeMark=time.time()
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    frame, width, height = cam.CaptureRGBA(zeroCopy=1)
    # classID is not a word, it is a number that identifies the class
    classID, confidence = net.Classify(frame, width, height)
    # get the actual item by classID
    item = net.GetClassDesc(classID)
    # calculate the changing time (derivative of time) and fps
    dt=time.time()-timeMark
    fps=1/dt
    # Low Pass Filer to smooth fps
    fpsFilter=.95*fpsFilter+.05*fps
    timeMark=time.time()
    # convert cuda to np to use opencv; RGBA -> 4
    frame=jetson.utils.cudaToNumpy(frame,width,height,4)
    # float 32 to unsigned int
    frame=cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR).astype(np.uint8)
    # type the text of item on screen
    cv2.putText(frame,str(round(fpsFilter,1))+'      '+item,(0,30),font,1,(0,0,255),2)
    cv2.imshow('webCam',frame)
    cv2.moveWindow('webCam',0,0)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()