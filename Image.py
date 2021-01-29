
# ----------------- Face Detection Code for an Image -----------------

# Steps
    # 1. Load of Faces
    # 2. Make them black and white
    # 3. Create the Algorithms

import cv2
from random import randrange

# Load pretrained dataset of opencv in xml file  
face_data = cv2.CascadeClassifier('Data_Face.xml')
# eye_data = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_data = cv2.CascadeClassifier('Data_Smile.xml')

img = cv2.imread('smile1.jpg')

# Convert the image to a grayscale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coo = face_data.detectMultiScale(gray_img)
# eye_coo = eye_data.detectMultiScale(gray_img)

smile_coo = smile_data.detectMultiScale(gray_img)
# detectMultiScale it is going to detect the faces in multiscale 

# prints the face Coordinates 
print(face_coo)

# Draw Rectangle surrounding the face coordinates

# Two Ways 

# [x,y,w,h] = face_coo[0]
# cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)

# For Face 
for x,y,w,h in face_coo:
    cv2.rectangle(img, (x,y), (x+w, y+h), (randrange(156,256),randrange(156,256),randrange(156,256)), 2)



# For Smile 
for x,y,w,h in smile_coo:
    cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0), 1)

# cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
# cv2.rectangle(img, (30,38), (30 + 118,38 + 118), (255,0,0), 4)

#It will create a popup with the image and a name
cv2.imshow("Smile Detector", img)


#It will make the popup wait for a while
cv2.waitKey()

print("Code Completed") 



