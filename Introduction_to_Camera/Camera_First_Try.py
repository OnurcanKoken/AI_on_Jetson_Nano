import cv2
print(cv2.__version__)

cam=cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    cv2.imshow('nanoCam',frame)
    cv2.moveWindow('nanoCam',0,200)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    cv2.imshow('graywindow',gray)
    cv2.moveWindow('graywindow',500,200)

    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
