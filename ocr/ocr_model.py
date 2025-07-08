from paddleocr import PaddleOCR
import numpy as np
import supervision as sv


class OcrModel:
    def __init__(self):
        self.model = PaddleOCR(
            text_detection_model_dir="models/PP-OCRv5_server_det",
            text_recognition_model_dir="models/PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        self.memory = {}

    def predict_boxes(self, frame: np.ndarray, players_detections: sv.Detections):
        boxes = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
        jersey_numbers = []
        for box, id in zip(boxes, players_detections.tracker_id):
            result = self.model.predict(box)[0]
            if id in self.memory:
                number = self.memory[id]
            else:
                number = ""
            for text in result["rec_texts"]:
                if text.isnumeric():
                    number = text
                    self.memory[id] = number
                    break
            jersey_numbers.append(number)
        return np.array(jersey_numbers, dtype=str)
