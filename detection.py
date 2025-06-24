import torch
from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO
from teamDetection import TeamClassifier

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

BALL_ID = 0
PLAYER_ID = 1
REFEREE_ID = 2
RIM_ID = 3

SOURCE_VIDEO_PATH = 'tmp/video/video_3.mp4'
TARGET_VIDEO_PATH = 'output_2.mp4'
model = YOLO('best.pt')
model.conf = 0.5

def extract_crops(source_video_path: str):
    frame_generator = sv.get_video_frames_generator(source_video_path, stride=15)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = model(frame, device=device, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]
        crops += [ sv.crop_image(frame, xyxy) for xyxy in detections.xyxy ]

    return crops

crops = extract_crops(SOURCE_VIDEO_PATH)
team_classifier = TeamClassifier(device = device)
team_classifier.fit(crops)

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(["#0000FF", "#00FF00", "#FFFF00"]),
    thickness=2
)

triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#F88158"),
    base=20, height=17
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#0000FF", "#00FF00"]),
    text_position=sv.Position.BOTTOM_CENTER
)

tracker = sv.ByteTrack()
tracker.reset()

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
video_sink = sv.VideoSink(TARGET_VIDEO_PATH, video_info=video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

with video_sink:
    for frame in tqdm(frame_generator, total=video_info.total_frames):

        result = model(frame, device=device, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(result)

        detections = detections[detections.class_id != RIM_ID]

        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        person_detections = detections[detections.class_id != BALL_ID]
        person_detections = person_detections.with_nms(threshold=0.5, class_agnostic=True)
        person_detections = tracker.update_with_detections(person_detections)

        players_detections = person_detections[person_detections.class_id == PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = team_classifier.predict(players_crops)

        referees_detections = person_detections[person_detections.class_id == REFEREE_ID]

        person_detections = sv.Detections.merge([players_detections, referees_detections])

        labels = [
            f"#{tracker_id}"
            for tracker_id
            in players_detections.tracker_id
        ]


        annotated_frame = frame.copy()
        annotated_frame = ellipse_annotator.annotate(annotated_frame, person_detections)
        annotated_frame = label_annotator.annotate(annotated_frame, players_detections, labels=labels)
        annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)
        video_sink.write_frame(annotated_frame)