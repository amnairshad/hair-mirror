import cv2, math, os
import numpy as np
import joblib
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = "dataset"
SHAPES = ["Heart", "Oblong", "Oval", "Round", "Square"]
TOP, CHIN = 10, 152
CHEEK_L, CHEEK_R = 234, 454
NOSE = 1
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
        379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
        234, 127, 162, 21, 54, 103, 67, 109]

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def features(pts):
    a = np.array(pts)
    l, r = a[CHEEK_L], a[CHEEK_R]
    center = (l + r) / 2
    ang = math.atan2(r[1] - l[1], r[0] - l[0])
    c, s = math.cos(-ang), math.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    b = (a - center) @ rot.T / np.linalg.norm(r - l)
    ratio = [np.linalg.norm(b[TOP] - b[CHIN])]
    return np.concatenate([b[OVAL].flatten(), ratio])


X = {"train": [], "test": []}
y = {"train": [], "test": []}
count = 0

for folder, _, files in os.walk(ROOT):
    label = os.path.basename(folder).capitalize()
    if label not in SHAPES:
        continue
    split = "test" if "test" in os.path.relpath(folder, ROOT).lower() else "train"
    for name in files:
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        count += 1
        if count % 200 == 0:
            print("Processed", count, "images...")
        img = cv2.imread(os.path.join(folder, name))
        if img is None:
            continue
        h, w, _ = img.shape
        res = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        pts = [(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark]
        left = math.dist(pts[NOSE], pts[CHEEK_L])
        right = math.dist(pts[NOSE], pts[CHEEK_R])
        if right == 0 or not (0.7 < left / right < 1.45):
            continue
        X[split].append(features(pts))
        y[split].append(label)

print("Train images:", len(y["train"]), " Test images:", len(y["test"]))

model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
model.fit(np.array(X["train"]), y["train"])
pred = model.predict(np.array(X["test"]))

print("\nTest accuracy:", round(accuracy_score(y["test"], pred) * 100, 1), "%")
print("\nRows = real shape, columns = predicted:", SHAPES)
print(confusion_matrix(y["test"], pred, labels=SHAPES))

joblib.dump(model, "face_shape_model.joblib")
print("\nModel saved as face_shape_model.joblib")