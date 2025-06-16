import torch
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans

# Configuration
YOLO_MODEL = 'yolov8x-seg.pt'
FRAME_PATH = 'test.png'
NUM_DOMINANT = 1
NUM_CLUSTERS = 3
OUTPUT_PATH = 'result.png'

def get_dominant_color(image, mask_, k=NUM_DOMINANT):
    masked_pixels = image[mask_ > 0]
    if len(masked_pixels) < 10:
        return np.array([0, 0, 0])

    kmeans_ = KMeans(n_clusters=k, n_init='auto')
    kmeans_.fit(masked_pixels)
    return kmeans_.cluster_centers_[0].astype(int)

# Load model
model = YOLO(YOLO_MODEL)
frame = cv2.imread(FRAME_PATH)
results = model(frame)[0]

# extract masks and dominant colors
dominant_colors = []
valid_indices = []
bounded_boxes = []
resized_mask = []

for i, (mask, cls, conf) in enumerate(zip(results.masks.data, results.boxes.cls, results.boxes.conf)):
    if int(cls.item()) != 0:
        continue
    if conf.item() < 0:
        continue

    mask_np = mask.cpu().numpy().astype(np.uint8)
    mask_np_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = get_dominant_color(frame, mask_np_resized)
    dominant_colors.append(color)
    valid_indices.append(i)
    bounded_boxes.append(results.boxes.xyxy[i].cpu().numpy())
    resized_mask.append(mask_np_resized)

if not dominant_colors:
    print("No player detected")
    exit()

# colors cluster
dominant_colors_np = np.array(dominant_colors)
kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init='auto')
labels = kmeans.fit_predict(dominant_colors_np)

# colors mapping
visual_colors = [
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255)
]

# draw results
for index, label, box, mask in zip(valid_indices, labels, bounded_boxes, resized_mask):
    x1, y1, x2, y2 = map(int, box)
    color = visual_colors[label]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    colored_mask = np.zeros_like(frame, dtype=np.uint8)
    colored_mask[mask > 0] = color
    frame = cv2.addWeighted(frame, 1.0, colored_mask, 0.4, 0)

legend_labels = ['Team A (Red)', 'Team B (Blue)', 'Ref (Yellow)']
visual_colors = kmeans.cluster_centers_.astype(int).tolist()
print(visual_colors)
x_start, y_start = 10, 10
legend_height = 30 * NUM_CLUSTERS + 10
legend_width = 200
cv2.rectangle(frame, (x_start - 5, y_start - 5), (x_start + legend_width, y_start + legend_height), (255, 255, 255), -1)

for i, (label, color) in enumerate(zip(legend_labels, visual_colors)):
    top_left = (x_start, y_start + i * 30)
    bottom_right = (x_start + 20, y_start + 20 + i * 30)
    cv2.rectangle(frame, top_left, bottom_right, color, -1)
    cv2.putText(frame, label, (x_start + 30, y_start + 15 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

# save and show
cv2.imwrite(OUTPUT_PATH, frame)
print(f"result save in {OUTPUT_PATH}")
