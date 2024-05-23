import cv2
import tensorflow as tf
import numpy as np
from pygame import mixer
import os
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = tf.keras.models.load_model(r'Models/model_v2.h5')

thicc = 2

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
path = os.getcwd()
mixer.init()
sound = mixer.Sound(r'alarm.wav')
cap = cv2.VideoCapture(0)
Score = 0
frame_counter = 0
skip_frames = 5  # Skip processing every 5 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[0:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame_counter % skip_frames == 0:
        faces = face_cascade.detectMultiScale(gray, 1.2, 3)
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 1)
    frame_counter += 1

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    for (ex, ey, ew, eh) in eyes:
        eye = frame[ey:ey + eh, ex:ex + ew]
        eye = cv2.resize(eye, (80, 80))
        eye = eye / 255.0
        eye = eye.reshape(1, 80, 80, 3)
        prediction = model.predict(eye)

        if prediction[0][0] > 0.50:
            cv2.putText(frame, "Closed", (10, height - 20), font, 1, (5, 30, 252), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score:" + str(Score), (100, height - 20), font, 1, (252, 5, 5), 1, cv2.LINE_AA)
            Score += 1
            if Score > 5:
                cv2.imwrite(os.path.join(path, 'Images/status.jpg'), frame)
                try:
                    sound.play()
                except:
                    pass
                thicc = thicc + 2 if thicc < 16 else thicc - 2
                thicc = max(thicc, 2)
                cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
        elif prediction[0][1] > 0.50:
            cv2.putText(frame, "Open", (10, height - 20), font, 1, (5, 30, 252), 1, cv2.LINE_AA)
            cv2.putText(frame, "Score:" + str(Score), (100, height - 20), font, 1, (252, 5, 5), 1, cv2.LINE_AA)
            Score = max(Score - 1, 0)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
