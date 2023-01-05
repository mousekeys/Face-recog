import cv2
from random import randrange 

#load datas previously learnt
face=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose photo to read
img= cv2.imread('rdj.png')

#convert the code to grayscale so that the ai can more easily detect the faces more properly
grayscale_img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#shows the images but for a certain instance only
#cv2.imshow('main',grayscale_img)

#detects faces
face_coordinates= face.detectMultiScale(grayscale_img)

#shows face coordinates
print(face_coordinates)

#lets us draw an rectangle
#the 0,22,0 gives color this gibs green color to the rectangle
#the end part 2 is the thicknmess of the rectangle
for (x,y,w,h) in face_coordinates:
  img2=cv2.rectangle(img, (x,y), (x+w,y+h), (randrange(256),randrange(256),randrange(256)) ,10)

cv2.imshow('face',img)

#makes the loaded code wait so that the user can see the output
cv2.waitKey()

print("hello world")