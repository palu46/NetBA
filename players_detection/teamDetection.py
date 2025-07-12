from typing import Generator, Iterable, List, TypeVar

import numpy as np
import supervision as sv
import torch
import umap
import cv2
from sklearn.cluster import KMeans
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH, use_fast=True)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in batches:
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)
    
    def extract_feature_histogram(self, crops: List[np.ndarray]) -> np.ndarray:
      """
      Extracts HSV Hue histograms from a list of player crops.
      
      Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
      """
      features = []

      for crop in crops:
          hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

          h_crop, w_crop = hsv.shape[:2]

          # trying to mask jersey area
          jersey_mask = np.zeros((h_crop, w_crop), np.uint8)
          jersey_mask[int(0.1 * h_crop):int(0.5 * h_crop), int(0.2 * w_crop):int(0.8 * w_crop)] = 255

          # Compute histogram of Hue channel using the jersey mask
          hist = cv2.calcHist([hsv], [0], jersey_mask, [16], [0, 180])
          hist = cv2.normalize(hist, None).flatten()

          features.append(hist)

      return np.vstack(features)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        # data = self.extract_features(crops)
        # projections = self.reducer.fit_transform(data)
        data = self.extract_feature_histogram(crops)
        self.cluster_model.fit(data)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        # data = self.extract_features(crops)
        # projections = self.reducer.transform(data)
        data = self.extract_feature_histogram(crops)
        return self.cluster_model.predict(data)