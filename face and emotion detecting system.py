import cv2
from deepface import DeepFace

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    break

    # Create green background
    green_background = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_background[:, :, 0] = 60  # Set hue to green
    green_background = cv2.cvtColor(green_background, cv2.COLOR_HSV2BGR)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        try:
            # Emotion detection
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        cv2.rectangle(green_background, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(green_background, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face & Emotion Detection - Green Background", green_background)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()