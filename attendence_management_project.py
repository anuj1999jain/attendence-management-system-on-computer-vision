import face_recognition
import numpy as np
import cv2
import os
from datetime import datetime

path="student_images"
mylist=os.listdir(path)
images=[]
student_name=[]
#print(mylist)

for cl in mylist:                          
    curImg=cv2.imread(f'{path}/{cl}')                     #to read the imageds into the program for computation
    images.append(curImg)
    student_name.append(os.path.splitext(cl)[0])

def findEncodings(images):
    ''' 
        accept images as the arguments and the return the encodings of those images
    '''
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)           # the images are originally in BGR so we have to convert in RGB
        encode=face_recognition.face_encodings(img)[0]    #this will find the encodings of the image
        encodeList.append(encode)                         
    return encodeList

def mark_attendance(name):
    '''
        this function accept the name as the argument and mark the attendance
    '''
    with open('attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeings_known=findEncodings(images)

cap=cv2.VideoCapture(0)                          # to start the webcamp 

while True:
    success,img=cap.read()                       #images are taken from each frame
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    face_Cur_frame=face_recognition.face_locations(imgS)
    encodes_cur_frame=face_recognition.face_encodings(imgS,face_Cur_frame)

    for encodeFace,faceLoc in zip(encodes_cur_frame,face_Cur_frame):
        matches=face_recognition.compare_faces(encodeings_known,encodeFace)
        faceDist=face_recognition.face_distance(encodeings_known,encodeFace)
        #print(faceDist)

        #it will return the index of the images with least face distance
        matchIndex=np.argmin(faceDist)

        if matches[matchIndex]:  
            name=student_name[matchIndex].upper()
            mark_attendance(name)
            y1,x2,y2,x1=faceLoc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)


