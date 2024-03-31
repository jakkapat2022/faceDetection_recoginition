from random import random
import numpy as np , cv2 , os , pickle ,dlib
path = './facedata/'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC = []
FACE_NAME = []
FACE_LNAME = []
FACE_ID = []
for fn in os.listdir(path):
    if fn.endswith('.jpg'):
        img = cv2.imread(path + fn)[:,:,::-1]
        dets = detector(img, 1)
        for k ,d in enumerate(dets):
            shape = sp(img,d)
            face_desc = model.compute_face_descriptor(img,shape,100)
            FACE_DESC.append(face_desc)
            print('loading...', fn)
            FACE_NAME.append(fn[fn.index(' ') + 1:fn.index('=')])
            FACE_LNAME.append(fn[fn.index('=') + 1:fn.index('_')])
            FACE_ID.append(fn[:fn.index('-')])
            print(FACE_ID,FACE_NAME,FACE_LNAME) 
pickle.dump((FACE_DESC,FACE_NAME , FACE_LNAME ,FACE_ID),open('trainset.pk', 'wb'))