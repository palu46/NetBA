from ultralytics import YOLO
import supervision as sv
import numpy as np
from keypoints_detection.view_transformer.view import ViewTransformer
from keypoints_detection.configs.reference_court import BasketCourtConfiguration
from video_annotation.annotation import annotate_keypoints

class KeypointsDetector:

    def __init__(self,model: str):
        self.model = YOLO(model)
        self.config = BasketCourtConfiguration()
        self.reference_keypoints = self.config.vertices
        self.edges = self.config.edges

    def get_keypoints(self, input_frame: np.ndarray) -> tuple[sv.KeyPoints, np.ndarray, np.ndarray]:
        ultralytics_results = self.model.predict(source=input_frame, conf=0.3, verbose=False)
        ultralytics_result = ultralytics_results[0]

        key_points = sv.KeyPoints.from_ultralytics(ultralytics_result)
        
        filter = key_points.confidence[0] > 0.85
        frame_reference_points = key_points.xy[0][filter]
        frame_reference_key_points = sv.KeyPoints(
            xy=frame_reference_points[np.newaxis, ...])

        court_reference_points = np.array(self.reference_keypoints)[filter]
        if len(court_reference_points) < 4:
            return None, None, input_frame

        transformer = ViewTransformer(
            source=court_reference_points,
            target=frame_reference_points
        )

        court_all_points = np.array(self.reference_keypoints)
        frame_all_points = transformer.transform_points(points=court_all_points)

        frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

        annotated_frame = annotate_keypoints(input_frame, self.edges, frame_all_key_points, frame_reference_key_points)

        return frame_reference_points, filter, annotated_frame

"""
    reference_keypoints = CONFIG.vertices
    edges = CONFIG.edges

    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        thickness=2, edges=edges)
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.from_hex('#FF1493'),
        radius=8)
    vertex_annotator_2 = sv.VertexAnnotator(
        color=sv.Color.from_hex('#00BFFF'),
        radius=8)

    frame_count = 0
    frame_generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    cap = cv2.VideoCapture(SOURCE_VIDEO_PATH)
    if not cap.isOpened():
        print("Error while opening the video.")
        exit()

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = "output_annotated_video.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for frame in frame_generator:
        ultralytics_results = model.predict(source=frame, conf=0.3, verbose=False)
        ultralytics_result = ultralytics_results[0]
        key_points = sv.KeyPoints.from_ultralytics(ultralytics_result)

        filter = key_points.confidence[0] > 0.85
        frame_reference_points = key_points.xy[0][filter]
        frame_reference_key_points = sv.KeyPoints(
            xy=frame_reference_points[np.newaxis, ...])

        court_reference_points = np.array(reference_keypoints)[filter]

        transformer = ViewTransformer(
            source=court_reference_points,
            target=frame_reference_points
        )

        court_all_points = np.array(reference_keypoints)
        frame_all_points = transformer.transform_points(points=court_all_points)

        frame_all_key_points = sv.KeyPoints(xy=frame_all_points[np.newaxis, ...])

        annotated_frame = frame.copy()
        annotated_frame = edge_annotator.annotate(
            scene=annotated_frame,
            key_points=frame_all_key_points)
        annotated_frame = vertex_annotator_2.annotate(
            scene=annotated_frame,
            key_points=frame_all_key_points)
        annotated_frame = vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=frame_reference_key_points)

        out.write(annotated_frame)

        frame_count += 1
        print(f"Elaborated Frame {frame_count}")

    out.release()
    print(f"Video saved in: {output_video_path}")
"""