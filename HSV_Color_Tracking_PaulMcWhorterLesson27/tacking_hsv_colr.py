import cv2
import numpy as np
print(cv2.__version__)

def nothing(x):
    pass

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars',1320,0)
cv2.createTrackbar('hueLower','Trackbars',50,179,nothing)
cv2.createTrackbar('hueHigher','Trackbars',100,179,nothing)
cv2.createTrackbar('satLow','Trackbars',100,255,nothing)
cv2.createTrackbar('satHigh','Trackbars',255,255,nothing)
cv2.createTrackbar('valLow','Trackbars',100,255,nothing)
cv2.createTrackbar('valHigh','Trackbars',255,255,nothing)

cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    #cv2.imshow('nanoCam',frame)
    #frame=cv2.imread('smarties.png')
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
    hueLow=cv2.getTrackbarPos('hueLower', 'Trackbars')
    hueUp=cv2.getTrackbarPos('hueHigher', 'Trackbars')
    
    Ls=cv2.getTrackbarPos('satLow', 'Trackbars')
    Us=cv2.getTrackbarPos('satHigh', 'Trackbars')
    
    Lv=cv2.getTrackbarPos('valLow', 'Trackbars')
    Uv=cv2.getTrackbarPos('valHigh','Trackbars')
    
    l_b=np.array([hueLow,Ls,Lv])
    u_b=np.array([hueUp,Us,Uv])

    FGmask=cv2.inRange(hsv,l_b,u_b)
    #cv2.imshow('FGmask',FGmask)

    FG=cv2.bitwise_and(frame, frame ,mask=FGmask)
    #cv2.imshow('sekilsukul',FG)

    bigmask=cv2.bitwise_not(FGmask)
    #cv2.imshow('not',bigmask)

    BG=cv2.cvtColor(bigmask,cv2.COLOR_GRAY2BGR)
    final=cv2.add(FG,BG)
    cv2.imshow('BG',final)


    #    cv2.imshow('Smarties',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
