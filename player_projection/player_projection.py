from keypoints_detection.view_transformer.view import ViewTransformer
from keypoints_detection.configs.reference_court import BasketCourtConfiguration
from video_annotation.annotation import annotate_radar
import supervision as sv
import numpy as np
import cv2


class PlayerProjection:

    def __init__(self, court_image):
        self.config = BasketCourtConfiguration()
        self.reference_keypoints = self.config.vertices
        self.court_image = cv2.imread(court_image)


    def project_players(
        self, frame_reference_points, filter, players_detections, referees_detections, jersey_numbers
    ):
        court_reference_points = np.array(self.reference_keypoints)[filter]
        transformer = ViewTransformer(source=frame_reference_points, target=court_reference_points)

        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        court_players_xy = transformer.transform_points(points=players_xy)

        referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        court_referees_xy = transformer.transform_points(points=referees_xy)

        annotated_frame = annotate_radar(self.court_image, court_players_xy, court_referees_xy, players_detections, jersey_numbers)

        return annotated_frame
