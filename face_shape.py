import cv2
import math
import mediapipe as mp
import csv, os
from collections import deque, Counter

face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
cap = cv2.VideoCapture(0)

# keeps the last 30 results so the answer does not flicker
recent = deque(maxlen=30)
last = None
person = 1
ratio_hist = deque(maxlen=15)

# Landmark numbers on the face mesh
TOP = 10            # top of forehead
CHIN = 152          # bottom of chin
CHEEK_L, CHEEK_R = 234, 454   # widest part (cheekbones)
JAW_L, JAW_R = 172, 397       # jaw corners
FORE_L, FORE_R = 54, 284      # sides of forehead


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def decide_shape(len_r, jaw_r, fore_r):
    # len_r  = face length / cheek width
    # jaw_r  = jaw width / cheek width
    # fore_r = forehead width / cheek width
    # These numbers are starting values. You will tune them in Step 7.
    if len_r > 1.45:
        return "Oblong"
    if fore_r - jaw_r > 0.12:
        return "Heart"
    if len_r < 1.30:
        if jaw_r > 0.78:
            return "Square"
        return "Round"
    return "Oval"


while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        pts = [(p.x * w, p.y * h) for p in lm]

        length = dist(pts[TOP], pts[CHIN])
        cheek = dist(pts[CHEEK_L], pts[CHEEK_R])
        jaw = dist(pts[JAW_L], pts[JAW_R])
        fore = dist(pts[FORE_L], pts[FORE_R])

        len_r = length / cheek
        jaw_r = jaw / cheek
        fore_r = fore / cheek
        ratio_hist.append((len_r, jaw_r, fore_r))
        last = tuple(sum(v) / len(ratio_hist) for v in zip(*ratio_hist))

        recent.append(decide_shape(*last))
        shape = Counter(recent).most_common(1)[0][0]

        # draw the key points and lines
        for idx in (TOP, CHIN, CHEEK_L, CHEEK_R, JAW_L, JAW_R, FORE_L, FORE_R):
            x, y = int(pts[idx][0]), int(pts[idx][1])
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        cv2.line(frame, tuple(map(int, pts[CHEEK_L])), tuple(map(int, pts[CHEEK_R])), (255, 200, 0), 1)
        cv2.line(frame, tuple(map(int, pts[JAW_L])), tuple(map(int, pts[JAW_R])), (0, 200, 255), 1)
        cv2.line(frame, tuple(map(int, pts[FORE_L])), tuple(map(int, pts[FORE_R])), (255, 0, 200), 1)
        cv2.line(frame, tuple(map(int, pts[TOP])), tuple(map(int, pts[CHIN])), (0, 255, 255), 1)

        # show the result and the ratios
        cv2.putText(frame, "Face shape: " + shape, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"length/cheek={len_r:.2f}  jaw/cheek={jaw_r:.2f}  forehead/cheek={fore_r:.2f}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    else:
        last = None
        ratio_hist.clear()
        recent.clear()
        cv2.putText(frame, "No face found - please stand in front of the camera",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Face Shape", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    labels = {ord("o"): "Oval", ord("r"): "Round", ord("s"): "Square",
              ord("h"): "Heart", ord("l"): "Oblong"}
    if key in labels and last is not None:
        is_new = not os.path.exists("data.csv")
        with open("data.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["person", "label", "length_cheek", "jaw_cheek", "forehead_cheek"])
            writer.writerow([person, labels[key], round(last[0], 3), round(last[1], 3), round(last[2], 3)])
        print("Saved: person", person, labels[key])
    if key == ord("n"):
        person += 1
        print("Now person", person)
 
cap.release()
cv2.destroyAllWindows()