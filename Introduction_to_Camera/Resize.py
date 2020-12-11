import cv2
print(cv2.__version__)

cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame2=cv2.resize(frame,(320,240))
    cv2.imshow('nanoCam',frame2)
    cv2.moveWindow('nanoCam',0,0)

    small=cv2.resize(frame,(320,240))
    gray=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    cv2.moveWindow('BW',320,0)
    cv2.imshow('BW',gray)


    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
