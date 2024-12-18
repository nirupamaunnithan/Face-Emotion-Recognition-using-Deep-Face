import cv2
import time
from predict import analyze_emotion

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

capture_interval = 3
last_capture_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    cv2.imwrite('photo.jpg', frame)
    current_time = time.time()

    if len(faces) > 0 and (current_time - last_capture_time >= capture_interval):
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 0), 2) # will box around your face when capturing
            cv2.imwrite('photo.jpg', frame) # To capture the initial photo to make sure the face is there
            last_capture_time = current_time

    #print(analyze_emotion('photo.jpg')) # this will print the current emotion in terminal
    predicted_emotion,predicted_score = analyze_emotion('photo.jpg') # will save the present emotion every 3 second
    predicted_score_str = str(predicted_score)
    cv2.putText(frame,predicted_emotion,(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame,predicted_score_str,(100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('Amvi', frame)
    if cv2.waitKey(10) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
