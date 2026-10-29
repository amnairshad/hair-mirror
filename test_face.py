import cv2
import mediapipe as mp

face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if result.multi_face_landmarks:
        h, w, _ = frame.shape
        for p in result.multi_face_landmarks[0].landmark:
            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)
    cv2.imshow("Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()