import supervision as sv
import numpy as np
import cv2


def annotate_detections(
    frame: np.ndarray,
    players_detections: sv.Detections,
    referees_detections: sv.Detections,
    ball_detections: sv.Detections,
    jersey_numbers: np.ndarray,
):
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(["#0000FF", "#00FF00", "#FFFF00"]), thickness=2
    )

    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#F88158"), base=20, height=17
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(["#0000FF", "#00FF00"]),
        text_position=sv.Position.BOTTOM_CENTER,
    )

    person_detections = sv.Detections.merge([players_detections, referees_detections])

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(annotated_frame, person_detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, players_detections, labels=jersey_numbers
    )
    annotated_frame = triangle_annotator.annotate(annotated_frame, ball_detections)

    return annotated_frame


def annotate_keypoints(
    frame: np.ndarray, edges, frame_all_key_points, frame_reference_key_points
):
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex("#00BFFF"), thickness=2, edges=edges
    )
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.from_hex("#FF1493"), radius=8)
    vertex_annotator_2 = sv.VertexAnnotator(
        color=sv.Color.from_hex("#00BFFF"), radius=8
    )

    annotated_frame = frame.copy()
    annotated_frame = edge_annotator.annotate(
        scene=annotated_frame, key_points=frame_all_key_points
    )
    annotated_frame = vertex_annotator_2.annotate(
        scene=annotated_frame, key_points=frame_all_key_points
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame, key_points=frame_reference_key_points
    )


def annotate_radar(
    frame: np.ndarray,
    court_players_xy: np.ndarray,
    court_referees_xy: np.ndarray,
    players_detections: sv.Detections,
    jersey_numbers: np.ndarray,
):
    annotated_frame = frame.copy()
    annotated_frame = draw_points(
        annotated_frame,
        court_players_xy[players_detections.class_id == 0],
        sv.Color.from_hex("#0000FF"),
        sv.Color.BLACK,
        jersey_numbers=jersey_numbers[players_detections.class_id == 0],
    )
    annotated_frame = draw_points(
        annotated_frame,
        court_players_xy[players_detections.class_id == 1],
        sv.Color.from_hex("#00FF00"),
        sv.Color.BLACK,
        jersey_numbers=jersey_numbers[players_detections.class_id == 1],
    )
    annotated_frame = draw_points(
        annotated_frame,
        court_referees_xy,
        sv.Color.from_hex("#FFFF00"),
        sv.Color.BLACK,
    )

    return annotated_frame


def draw_points(
    image,
    points,
    face_color,
    edge_color,
    scale=1,
    padding=50,
    radius=30,
    thickness=2,
    jersey_numbers=None,
):
    for idx, point in enumerate(points):
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding,
        )
        cv2.circle(
            img=image,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1,
        )
        cv2.circle(
            img=image,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness,
        )

        if jersey_numbers is not None:
            number = jersey_numbers[idx]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                number, font, font_scale, font_thickness
            )
            text_org = (
                scaled_point[0] - text_width // 2,
                scaled_point[1] + text_height // 2,
            )
            cv2.putText(
                img=image,
                text=number,
                org=text_org,
                fontFace=font,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )
    return image
