import torch
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from players_detection.teamDetection import TeamClassifier

class PlayerDetector:
    BALL_ID = 0
    PLAYER_ID = 1
    REFEREE_ID = 2
    RIM_ID = 3

    def __init__(
            self,
            model: str,
            source_video_path:str,
            device:str
    ):
        self.model = YOLO(model)
        self.model.conf = 0.5
        self.device = device

        ### Initialize teamClassifier
        frame_generator = sv.get_video_frames_generator(source_video_path, stride=15)

        crops = []
        for frame in tqdm(frame_generator, desc='collecting crops'):
            result = self.model(frame, device=self.device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections.with_nms(threshold=0.5, class_agnostic=True)
            detections = detections[detections.class_id == self.PLAYER_ID]
            crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

        self.team_classifier = TeamClassifier(device=self.device)
        self.team_classifier.fit(crops)

        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(["#0000FF", "#00FF00", "#FFFF00"]),
            thickness=2
        )

        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex("#F88158"),
            base=20, height=17
        )

        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(["#0000FF", "#00FF00"]),
            text_position=sv.Position.BOTTOM_CENTER
        )

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

    def detect_players(self, frame):
        result = self.model(frame, device=self.device, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(result)

        detections = detections[detections.class_id != self.RIM_ID]

        ball_detections = detections[detections.class_id == self.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        person_detections = detections[detections.class_id != self.BALL_ID]
        person_detections = person_detections.with_nms(threshold=0.5, class_agnostic=True)
        person_detections = self.tracker.update_with_detections(person_detections)

        players_detections = person_detections[person_detections.class_id == self.PLAYER_ID]
        players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        players_detections.class_id = self.team_classifier.predict(players_crops)

        referees_detections = person_detections[person_detections.class_id == self.REFEREE_ID]

        person_detections = sv.Detections.merge([players_detections, referees_detections])

        labels = [
            f"#{tracker_id}"
            for tracker_id
            in players_detections.tracker_id
        ]

        annotated_frame = frame.copy()
        annotated_frame = self.ellipse_annotator.annotate(annotated_frame, person_detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, players_detections, labels=labels)
        annotated_frame = self.triangle_annotator.annotate(annotated_frame, ball_detections)

        return annotated_frame, players_detections, referees_detections


