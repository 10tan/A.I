# Steps
    # 1. Load of Faces
    # 2. Make them black and white
    # 3. Create the Algorithms

import cv2
from random import randrange

# Load pretrained dataset of opencv in xml file  
face_data = cv2.CascadeClassifier('Data_Face.xml')
smile_data = cv2.CascadeClassifier('Data_Smile.xml')


# Choose a video to read using opencv

# For Prerecorded video stored in computer
# video = cv2.VideoCapture('me.mp4')
video = cv2.VideoCapture(0)

# Iterate forever through the frames in video 
while True:
    # Read the Current Frame 
    successfull_frame_read, frame = video.read()
    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not successfull_frame_read:
        break
    #Detect Faces
    face_coo = face_data.detectMultiScale(gray_video)
    smile_coo = smile_data.detectMultiScale(gray_video)

    # # Two Ways To draw Rectangle over the face

    # # [x,y,w,h] = face_coo[0]
    # # cv2.rectangle(video, (x,y), (x+w, y+h), (255,255,0), 2)

    for (x,y,w,h) in face_coo:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(randrange(156,256),randrange(156,256),randrange(156,256)), 4)
    
    for (x,y,w,h) in smile_coo:
        cv2.rectangle(frame, (x,y), (x+w, y+h),(randrange(156,256),randrange(156,256),randrange(156,256)), 4)

    cv2.imshow("Face Detector ", frame)

    # if key==81 or key==118:
    #     break
    cv2.waitKey(1)

video.release()
cv2.destroyAllWindows()
# # It uses the webcam connected to the computer
# # video = cv2.VideoCapture(0)
