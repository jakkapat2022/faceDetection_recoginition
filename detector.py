# -*- coding: utf-8 -*-
import json
from multiprocessing.sharedctypes import Value
import numpy as np , cv2 , os ,dlib ,pickle,shutil
import firebase_admin
from datetime import date
from time import strftime
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('')
firebase_admin.initialize_app(cred,{
    'databaseURL': ''
})


#opendata.set({
#   'start':'0'
#})

src = r'/home/jakkapat/Desktop/project/backlist/list.pk'
dst = r'/home/jakkapat/Desktop/project/list.pk'
shutil.copyfile(src,dst)

face_dector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC , FACE_NAME , FACE_LNAME ,FACE_ID = pickle.load(open('trainset.pk','rb'))
cap = cv2.VideoCapture('1.mp4')

DC = 0
state = False


while True:
    current_time = strftime('%H:%M:%S')
    open_class = int(strftime('%H'))
    time_reset_blacklist = strftime('%H:%M:%S')
    opendata = db.reference('active/')
    json_str = json.dumps(opendata.get())
    resp = json.loads(json_str)
    Blcklist = pickle.load(open('list.pk', 'rb'))
    now = str(date.today())
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_dector.detectMultiScale(gray,1.2, 5)
    cv2.putText(frame, current_time, (500, 30), cv2.FONT_HERSHEY_COMPLEX, .6, (0, 255, 0), 2)
    cv2.putText(frame, now, (50, 30), cv2.FONT_HERSHEY_COMPLEX, .6, (0, 255, 0), 2)
    for (x, y, w, h) in faces:
        img = frame[y-10:y+h+10, x-10:x+w+10][:,:,::-1]
        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_desc0 = model.compute_face_descriptor(img, shape, 1)
            d = []
            for face_desc in FACE_DESC:
                d.append(np.linalg.norm(np.array(face_desc) - np.array(face_desc0)))
            d = np.array(d)
            idx = np.argmin(d)
            percent = round(d[idx] * 100,2)
            if d[idx] < 0.4:
                id = FACE_ID[idx]
                name = FACE_NAME[idx]
                lname = FACE_LNAME[idx]
                refdata = db.reference(now + '/' + name + ' ' + lname)
                recogName = Blcklist
                if recogName != name :#and resp(str['name']) != name:  # and recogName != str(''):
                    DC = 0
                    cv2.rectangle(frame, (x, y + h), (x + w, y + (h + 22)), (0, 150, 0), cv2.FILLED)
                    cv2.putText(frame, str(percent) + ' %', (x, y + (h + 20)), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255), 1)
                else:
                    DC = 1
                    cv2.rectangle(frame, (x, y + h), (x + w, y + (h + 42)), (0, 150, 0), cv2.FILLED)
                    cv2.putText(frame, 'Active', (x , y + (h + 20)), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255), 1)
                    cv2.putText(frame, str(percent) + ' %', (x, y + (h + 40)), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255), 1)
                if name == name and name != recogName:
                    if DC == 0 :#and open_class == 8:
                        if str(resp) == '1':
                            refdata.set({
                                'date': str(now),
                                'id': id,
                                'name': name,
                                'lname':lname,
                                'time': current_time
                            })
                            BLACK_LIST = name
                            pickle.dump((BLACK_LIST), open('list.pk', 'wb'))
                #cv2.rectangle(frame, (x + 1, y), (x + w, y - 30), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x - 30, y - 10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y + h), (x + w, y + (h + 42)), (0, 0, 150), cv2.FILLED)
                cv2.putText(frame, 'UNKNOW', (x , y + (h + 20)), cv2.FONT_HERSHEY_COMPLEX, .5,(255, 255, 255), 1)
                cv2.putText(frame, str(percent) + ' %', (x, y + (h + 40)), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
    if time_reset_blacklist == '11:30:00':
        src = r'/home/jakkapat/Desktop/project/backlist/deful.pk'
        dst = r'/home/jakkapat/Desktop/project/list.pk'
        shutil.copyfile(src,dst)

    if str(resp) == '1':
        cv2.putText(frame, 'Active Class', (50, 350), cv2.FONT_HERSHEY_COMPLEX, .6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'No Active Class', (50, 350), cv2.FONT_HERSHEY_COMPLEX, .6, (0, 0, 255), 2)

    cv2.imshow('System of Facerecognition',frame)
    if cv2.waitKey(1) == 27:
        #os.remove("list.pk")
        src = r'/home/jakkapat/Desktop/project/list.pk'
        dst = r'/home/jakkapat/Desktop/project/backlist/list.pk'
        shutil.copyfile(src,dst)
        break

cap.release()
cv2.destroyAllWindows()





