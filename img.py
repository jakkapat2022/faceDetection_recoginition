import numpy as np 
import cv2 
import os 
import dlib 
import pickle 

face_dector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
FACE_DESC , FACE_NAME = pickle.load(open('trainset.pk','rb'))
cap = cv2.VideoCapture('1.mp4')

while True:
    _,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_dector.detectMultiScale(gray,1.2, 5)
    for (x, y, w, h) in faces:
        img = frame[y-10:y+h+10, x-10:x+w+10][:,:,::-1]
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img,d)
            face_desc0 = model.compute_face_descriptor(img,shape,1)
            d = []
            for face_desc in FACE_DESC:
                d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
            d = np.array(d)
            idx = np.argmin(d)
            if d[idx] < 0.5:
                name = FACE_NAME[idx]
                cv2.putText(frame,name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,.7,(0,255,0),1)
                cv2.rectangle(frame,(x,y), (x+w,y+h), (0,255,0),1)
            else:
                cv2.putText(frame, 'unknow', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
