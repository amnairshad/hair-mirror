import cv2
import csv
import math
import os
import statistics
from collections import defaultdict
import mediapipe as mp

ROOT = "dataset"
SHAPES = {"Heart", "Oblong", "Oval", "Round", "Square"}

TOP, CHIN = 10, 152
CHEEK_L, CHEEK_R = 234, 454
JAW_L, JAW_R = 172, 397
FORE_L, FORE_R = 54, 284
NOSE = 1

face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


rows = []
skipped = 0
count = 0

for folder, _, files in os.walk(ROOT):
    label = os.path.basename(folder).capitalize()
    if label not in SHAPES:
        continue
    rel = os.path.relpath(folder, ROOT).lower()
    split = "test" if "test" in rel else "train"

    for name in files:
        if not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        count += 1
        if count % 200 == 0:
            print("Processed", count, "images...")

        img = cv2.imread(os.path.join(folder, name))
        if img is None:
            skipped += 1
            continue

        h, w, _ = img.shape
        result = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            skipped += 1
            continue

        pts = [(p.x * w, p.y * h) for p in result.multi_face_landmarks[0].landmark]

        # skip faces that are turned too much (not looking straight)
        left = dist(pts[NOSE], pts[CHEEK_L])
        right = dist(pts[NOSE], pts[CHEEK_R])
        if right == 0 or not (0.8 < left / right < 1.25):
            skipped += 1
            continue

        cheek = dist(pts[CHEEK_L], pts[CHEEK_R])
        rows.append([
            split, label,
            round(dist(pts[TOP], pts[CHIN]) / cheek, 3),
            round(dist(pts[JAW_L], pts[JAW_R]) / cheek, 3),
            round(dist(pts[FORE_L], pts[FORE_R]) / cheek, 3),
        ])

with open("dataset_ratios.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "label", "length_cheek", "jaw_cheek", "forehead_cheek"])
    writer.writerows(rows)

print("\nDone. Used", len(rows), "images, skipped", skipped)

# summary of the training images, per shape
groups = defaultdict(list)
for r in rows:
    if r[0] == "train":
        groups[r[1]].append(r[2:])

print("\nlabel | count | mean length | mean jaw | mean forehead")
for label in sorted(groups):
    vals = groups[label]
    means = [round(statistics.mean(v[i] for v in vals), 3) for i in range(3)]
    print(label, "|", len(vals), "|", means[0], "|", means[1], "|", means[2])