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

        return players_detections, referees_detections, ball_detections


