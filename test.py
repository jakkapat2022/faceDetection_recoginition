import cv2 ,dlib
face_dector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('1.mp4')
detector = dlib.get_frontal_face_detector()

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_dector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
      cv2.putText(img, 'face', (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 0, 255), 1)
      cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("img", img)
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()
