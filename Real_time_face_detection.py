# import library
import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime

# path of the known images folder
path = "C:\\Users\\Vipul Singh\\Desktop\\Img\\Known"

# load encoding for known images
def load_encoding(path):
    encodings = []
    names = []
    for i in os.listdir(path):
        img = cv2.imread(path+'\\'+i)
        encoding = face_recognition.face_encodings(img)
        encodings.append(encoding[0])
        names.append(os.path.splitext(i)[0])
    return encodings, np.array(names)


# mark attendence of detected images
def MarkAttendence(name):
    with open('attendence.csv', 'r+') as f:
        nameList = []
        myDataList = f.readlines()
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name  not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')
        
        
encodings, names = load_encoding(path)

# initialising webcam    
cam = cv2.VideoCapture(0)

# real-time detection
while 1:
    _, img = cam.read()
    imgs = cv2.resize(img, (0, 0), None, 1, 1)
    encode_img = face_recognition.face_encodings(imgs)
    face_loc = face_recognition.face_locations(imgs)
    for i in range(len(face_loc)):
        face = face_loc[i]
#         face = (face[0]*2, face[1]*2, face[2]*2, face[3]*2)
        mask = face_recognition.compare_faces(encodings, encode_img[i])
        cv2.rectangle(img, (face[3], face[0]), (face[1], face[2]), (255, 0, 0))
        cv2.rectangle(img, (face[3], face[2]-20), (face[1], face[2]), (0, 0, 255), cv2.FILLED)
        if(names[mask].size):
            cv2.putText(img, names[mask][0], (face[3], face[2]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 70, 255), 2)
            MarkAttendence(names[mask][0])
        else:
            cv2.putText(img, 'Unknown', (face[3], face[2]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 70, 255), 2)
    cv2.imshow('img',img)
    if cv2.waitKey(30) & 0xff == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
