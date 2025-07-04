from keypoints_detection.view_transformer.view import ViewTransformer
from keypoints_detection.configs.reference_court import BasketCourtConfiguration
import supervision as sv
import numpy as np
import cv2

class PlayerProjection:

    def __init__(self, court_image):
        self.config = BasketCourtConfiguration()
        self.reference_keypoints = self.config.vertices
        self.court_image = cv2.imread(court_image)


    def draw_points(self, image, points, face_color, edge_color, scale=1, padding=50, radius=20, thickness=2):
        for point in points:
            scaled_point = (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            cv2.circle(
                img=image,
                center=scaled_point,
                radius=radius,
                color=face_color.as_bgr(),
                thickness=-1
            )
            cv2.circle(
                img=image,
                center=scaled_point,
                radius=radius,
                color=edge_color.as_bgr(),
                thickness=thickness
            )
        return image

    def project_players(self, frame_reference_points, filter, players_detections, referees_detections):
        court_reference_points = np.array(self.reference_keypoints)[filter]
        transformer = ViewTransformer(
            source=frame_reference_points,
            target=court_reference_points
        )

        players_xy = players_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        court_players_xy = transformer.transform_points(points=players_xy)

        referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        court_referees_xy = transformer.transform_points(points=referees_xy)

        annotated_frame = self.court_image.copy()
        annotated_frame = self.draw_points(annotated_frame, court_players_xy[players_detections.class_id == 0], sv.Color.from_hex("#0000FF"), sv.Color.BLACK)
        annotated_frame = self.draw_points(annotated_frame, court_players_xy[players_detections.class_id == 1], sv.Color.from_hex("#00FF00"), sv.Color.BLACK)
        annotated_frame = self.draw_points(annotated_frame, court_referees_xy, sv.Color.from_hex("#FFFF00"), sv.Color.BLACK)

        return annotated_frame