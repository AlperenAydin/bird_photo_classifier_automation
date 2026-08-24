from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import birder
import numpy as np
import torch
from birder.inference.classification import infer_image
from PIL import Image
from ultralytics import YOLO


@dataclass
class BirdDetectionResult:
    bbox: List[int]  # [xmin, ymin, xmax, ymax]
    detection_confidence: float
    top_species: List[Tuple[str, float]]  # List of (species_name, confidence_percent)


class BirdIdentifierPipeline:
    """End-to-end pipeline that detects birds with YOLO and classifies

    the exact species using the Birder library.
    """

    def __init__(
        self,
        yolo_model_name: str = "yolov8n.pt",
        birder_model_name: str = "mvit_v2_t_il-all",
        yolo_conf_threshold: float = 0.35,
        device: Optional[str] = None,
    ):
        self.yolo_model_name = yolo_model_name
        self.birder_model_name = birder_model_name
        self.yolo_conf_threshold = yolo_conf_threshold

        # Determine target device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy/Instance loading
        self._yolo_model: Optional[YOLO] = None
        self._birder_net: Optional[torch.nn.Module] = None
        self._birder_transform = None
        self._idx_to_class: Dict[int, str] = {}
        self._bird_class_id: Optional[int] = None

        self._initialize_models()

    # ==========================================
    # Private Setup & Utility Methods
    # ==========================================

    def _initialize_models(self) -> None:
        """Loads both YOLO and Birder models into memory."""
        self._load_yolo()
        self._load_birder()

    def _load_yolo(self) -> None:
        """Initializes the YOLO detector and finds the 'bird' class ID."""
        self._yolo_model = YOLO(self.yolo_model_name)

        # Locate the COCO class ID for 'bird'
        for class_id, name in self._yolo_model.names.items():
            if name.lower() == "bird":
                self._bird_class_id = int(class_id)
                break

        if self._bird_class_id is None:
            raise ValueError("Could not find 'bird' class in YOLO model taxonomy.")

    def _load_birder(self) -> None:
        """Initializes the Birder classification network and its transforms."""
        self._birder_net, model_info = birder.load_pretrained_model(
            self.birder_model_name,
            inference=True,
            device=self.device,
        )

        # Build transform pipeline required by the specific model backbone
        img_size = birder.get_size_from_signature(model_info.signature)
        self._birder_transform = birder.classification_transform(
            img_size, model_info.rgb_stats
        )

        # Construct index-to-label mapping
        if isinstance(model_info.class_to_idx, list):
            self._idx_to_class = {i: name for i, name in enumerate(model_info.class_to_idx)}
        else:
            self._idx_to_class = {idx: label for label, idx in model_info.class_to_idx.items()}

    def _load_image(self, image_source: Union[str, Path, Image.Image]) -> Image.Image:
        """Ensures the incoming input is normalized to an RGB PIL Image."""
        if isinstance(image_source, (str, Path)):
            return Image.open(image_source).convert("RGB")
        elif isinstance(image_source, Image.Image):
            return image_source.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image_source)}")

    def _detect_bird_boxes(self, image: Image.Image) -> List[Tuple[List[int], float]]:
        """Runs YOLO detection and filters specifically for bird bounding boxes."""
        results = self._yolo_model(
            image,
            conf=self.yolo_conf_threshold,
            verbose=False,
            device=self.device,
        )[0]

        detections = []
        for box in results.boxes:
            if int(box.cls[0]) == self._bird_class_id:
                coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(box.conf[0])
                detections.append((coords, conf))

        return detections

    def _classify_crop(self, crop: Image.Image, top_k: int) -> List[Tuple[str, float]]:
        """Runs the Birder model on a cropped bird image and returns top predictions."""
        probabilities, _ = infer_image(
            self._birder_net,
            crop,
            self._birder_transform,
            device=self.device,
        )
        probs = probabilities[0]

        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(self._idx_to_class[idx], float(probs[idx] * 100)) for idx in top_indices]

    # ==========================================
    # Public API Methods
    # ==========================================

    def process_image(
        self,
        image_source: Union[str, Path, Image.Image],
        top_k: int = 3,
        fallback_to_full_image: bool = True,
    ) -> List[BirdDetectionResult]:
        """Detects and classifies all birds found in the given image.

        Args:
            image_source: File path (str/Path) or PIL Image instance.
            top_k: Number of highest-confidence species to return per detection.
            fallback_to_full_image: If True, runs classification on the full
              image if YOLO fails to detect any bounding box.

        Returns:
            List of BirdDetectionResult objects.
        """
        image = self._load_image(image_source)
        detections = self._detect_bird_boxes(image)

        # Handle case where no birds were detected by YOLO
        if not detections:
            if not fallback_to_full_image:
                return []
            detections = [([0, 0, image.width, image.height], 0.0)]

        results: List[BirdDetectionResult] = []
        for bbox, det_conf in detections:
            xmin, ymin, xmax, ymax = bbox
            crop = image.crop((xmin, ymin, xmax, ymax))

            species_predictions = self._classify_crop(crop, top_k=top_k)

            results.append(
                BirdDetectionResult(
                    bbox=bbox,
                    detection_confidence=det_conf,
                    top_species=species_predictions,
                )
            )

        return results

    def print_results(self, results: List[BirdDetectionResult]) -> None:
        """Pretty-prints the detection and classification results."""
        if not results:
            print("No birds found.")
            return

        for idx, res in enumerate(results, start=1):
            det_label = (
                f"{res.detection_confidence * 100:.1f}%"
                if res.detection_confidence > 0
                else "Full-image fallback"
            )
            print(f"\n[Bird #{idx}] BBox: {res.bbox} | Detection Confidence: {det_label}")
            for rank, (species, prob) in enumerate(res.top_species, start=1):
                print(f"  {rank}. {species}: {prob:.2f}%")


# ==========================================
# Execution Example
# ==========================================
if __name__ == "__main__":
    # Initialize once (loads weights into VRAM/RAM)
    pipeline = BirdIdentifierPipeline(
        yolo_model_name="yolov8n.pt",
        birder_model_name="mvit_v2_t_il-all",
        yolo_conf_threshold=0.30,
    )

    # Run inference on target image
    image_path = "bird_photo.jpg"

    # Replace with an actual file to test
    try:
        results = pipeline.process_image(image_path, top_k=3)
        pipeline.print_results(results)
    except FileNotFoundError:
        print(f"File not found: '{image_path}'. Place a test photo to run.")