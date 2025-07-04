import torch
import warnings
from tqdm import tqdm
import supervision as sv
from players_detection.PlayersDetector import PlayerDetector
from keypoints_detection.keypoints_detector import KeypointsDetector
from player_projection.player_projection.player_projection import PlayerProjection
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

SOURCE_VIDEO_PATH = 'video_4.mp4'
PLAYERS_DETCTION_VIDEO_PATH = 'players_detection.mp4'
KEYPOINTS_DETECION_VIDEO_PATH = 'keypoints_detection.mp4'
RADAR_VIDEO_PATH = 'radar.mp4'
COURT_PATH = 'nba_cup_court_16x9.jpg'

players_detector = PlayerDetector('models/YOLOv11-players.pt', SOURCE_VIDEO_PATH, device)

video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
radar_video_info = sv.VideoInfo(1920, 1080, video_info.fps, video_info.total_frames)
players_video_sink = sv.VideoSink(PLAYERS_DETCTION_VIDEO_PATH, video_info=video_info)
keypoints_video_sink = sv.VideoSink(KEYPOINTS_DETECION_VIDEO_PATH, video_info=video_info)
radar_video_sink = sv.VideoSink(RADAR_VIDEO_PATH, video_info=radar_video_info)
frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
keypoints_detector = KeypointsDetector('models/YOLOv8-keypoints.pt')
player_projector = PlayerProjection(COURT_PATH)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    with players_video_sink, keypoints_video_sink, radar_video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):
            players_annotated_frame, players_detections, referees_detections = players_detector.detect_players(frame)
            keypoints, filter, keypoints_annotated_frame = keypoints_detector.get_keypoints(frame)
            radar_frame = player_projector.project_players(keypoints, filter, players_detections, referees_detections)
            players_video_sink.write_frame(players_annotated_frame)
            keypoints_video_sink.write_frame(keypoints_annotated_frame)
            radar_video_sink.write_frame(radar_frame)