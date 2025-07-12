import os
import shutil
import argparse
import torch
import warnings
import supervision as sv
from tqdm import tqdm
from moviepy import VideoFileClip

from players_detection.PlayersDetector import PlayerDetector
from keypoints_detection.keypoints_detector import KeypointsDetector
from player_projection.player_projection import PlayerProjection
from video_annotation.annotation import annotate_detections
from ocr.ocr_model import OcrModel

warnings.simplefilter("ignore", category=FutureWarning)


def process_video(source_video_path):

    if not os.path.exists("output/"):
        os.mkdir("output/")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    PLAYERS_DETECTION_VIDEO_PATH = "output/players_detection.mp4"
    KEYPOINTS_DETECION_VIDEO_PATH = "output/keypoints_detection.mp4"
    RADAR_VIDEO_PATH = "output/radar.mp4"
    COURT_PATH = "data/nba_cup_court_16x9.jpg"

    players_detector = PlayerDetector(
        "models/YOLOv11-players.pt", source_video_path, device
    )

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    radar_video_info = sv.VideoInfo(1920, 1080, video_info.fps, video_info.total_frames)
    players_video_sink = sv.VideoSink(PLAYERS_DETECTION_VIDEO_PATH, video_info=video_info)
    keypoints_video_sink = sv.VideoSink(KEYPOINTS_DETECION_VIDEO_PATH, video_info=video_info)
    radar_video_sink = sv.VideoSink(RADAR_VIDEO_PATH, video_info=radar_video_info)
    frame_generator = sv.get_video_frames_generator(source_video_path)
    keypoints_detector = KeypointsDetector("models/YOLOv8-keypoints.pt")
    player_projector = PlayerProjection(COURT_PATH)
    ocr_model = OcrModel()

    with players_video_sink, keypoints_video_sink, radar_video_sink:
        for frame in tqdm(frame_generator, total=video_info.total_frames):

            players_detections, referees_detections, ball_detections = players_detector.detect_players(frame)
            jersey_numbers = ocr_model.predict_boxes(frame, players_detections)
            players_annotated_frame = annotate_detections(
                frame,
                players_detections,
                referees_detections,
                ball_detections,
                jersey_numbers,
            )

            try:
                keypoints, filter, keypoints_annotated_frame = keypoints_detector.get_keypoints(frame)
                radar_frame = player_projector.project_players(
                    keypoints,
                    filter,
                    players_detections,
                    referees_detections,
                    jersey_numbers,
                )
            except:  # less than 4 keypoints were found
                keypoints_annotated_frame = frame
                radar_frame = player_projector.court_image

            players_video_sink.write_frame(players_annotated_frame)
            keypoints_video_sink.write_frame(keypoints_annotated_frame)
            radar_video_sink.write_frame(radar_frame)


def convert_video_codec():
    for video in ["output/players_detection.mp4", "output/radar.mp4", "output/keypoints_detection.mp4"]:
        clip = VideoFileClip(video)

        # Re-encode to browser-compatible H.264 format
        clip.write_videofile(
            "output/tmp.mp4",
            codec="libx264",
            audio=False,
            logger=None,
        )

        clip.close()

        shutil.move("output/tmp.mp4", video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_video_path", type=str, help="Path to the input video")

    args = parser.parse_args()
    process_video(args.source_video_path)
    convert_video_codec()
