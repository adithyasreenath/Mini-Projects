import cv2
import numpy as np
import sys
def nothing(x):
    pass
    
def faceDraw():
#   if(x != 0):
    facePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(facePath)
    faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor= sF,
       minNeighbors=8,
       minSize=(55, 55),
       flags=0)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,"Face Detected",(x,y-5),font,0.65,(0,255,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

def smileDraw():
#   if(x != 0):
    facePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(facePath)
    smilePath = "haarcascade_smile.xml"
    smileCascade = cv2.CascadeClassifier(smilePath)
    faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor= sF,
       minNeighbors=8,
       minSize=(55, 55),
       flags=0)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=0)

        # Set region of interest for smiles
        if len(smile) > 0.75:
            for (x, y, w, h) in smile:
                cv2.rectangle(roi_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(roi_color,"Smile Found",(x,y+h+20),font,0.50,(255,0,0),2)

                
def eyeDraw():
#   if(x != 0):
    facePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(facePath)
    smilePath = "haarcascade_smile.xml"
    smileCascade = cv2.CascadeClassifier(smilePath)
    eyePath = "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(eyePath)
    faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor= sF,
       minNeighbors=8,
       minSize=(55, 55),
       flags=0)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex-10,ey-10),(ex+10+ew,ey+eh-30),(0,0,255),2)
            
                  
def frownDraw():
#   if(x != 0):
    facePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(facePath)
    smilePath = "haarcascade_smile.xml"
    smileCascade = cv2.CascadeClassifier(smilePath)
    eyePath = "haarcascade_eye.xml"
    eye_cascade = cv2.CascadeClassifier(eyePath)
    faces = faceCascade.detectMultiScale(
       gray,
       scaleFactor= sF,
       minNeighbors=8,
       minSize=(55, 55),
       flags=0)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=0)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            if len(smile) > 0.1:
              break
            else:
               cv2.rectangle(roi_color,(ex-10,ey-10),(ex+10+ew,ey+eh-30),(0,0,255),2)
                    
                    
                    
facePath = "haarcascade_frontalface_default.xml"
smilePath = "haarcascade_smile.xml"
faceCascade = cv2.CascadeClassifier(facePath)
smileCascade = cv2.CascadeClassifier(smilePath)

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

sF = 1.05
print("PRESS D FOR FACE DETECTION\nPRESS S FOR SMILE DETECTION\nPRESS E FOR EYEBROW DETECTION\nPRESS F FOR FROWN DETECTION\nPRESS A FOR FACE PLUS SMILE PLUS EYEBROW DETECTION\nPRESS Q TO QUIT")
x = str(raw_input())
while True:
   while True:
    ret, frame = cap.read() # Capture frame-by-frame
    img = frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #   facetog = cv2.getTrackbarPos('Face','Smile/Frown Detector')
    #   faceDraw(facetog)
    #   smiletog = cv2.getTrackbarPos('Smile','Smile Detection')
    #   smileDraw(smiletog)
    if(x == 'A' or x == 'a'):
        faceDraw()
        smileDraw()
        frownDraw()
    elif(x == 'E' or x == 'e'):
        eyeDraw()
    elif(x == 'D' or x == 'd'):
        faceDraw()
    elif(x == 'S' or x == 's'):
        smileDraw()
    elif(x == 'F' or x == 'f'):
        frownDraw()
    else:
        print("INVALID CHOICE")
        break    
    cv2.imshow('Smile/Frown Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   cv2.destroyAllWindows()
#   print("PRESS D FOR FACE DETECTION\nPRESS S FOR SMILE DETECTION\nPRESS E FOR EYEBROW DETECTION\nPRESS F FOR FROWN DETECTION\nPRESS A FOR FACE PLUS SMILE PLUS EYEBROW DETECTION\nPRESS Q TO QUIT")
   x = str(raw_input())
   if(x == 'Q' or x == 'q'):
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()