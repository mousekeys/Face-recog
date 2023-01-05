from tkinter.ttk import Frame
import cv2
from random import randrange 

#load datas previously learnt
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam= cv2.VideoCapture('face.mp4')

while True:
    succesful_read,frame= webcam.read()

    scalegray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    face_coor = face.detectMultiScale(scalegray)

    for (x,y,w,h) in face_coor:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)) ,10)
    
    cv2.imshow('in',frame)
    key= cv2.waitKey(1)

    if key==81 or key ==113:
        break

webcam.release()