from typing import List, Optional, Sequence

import cv2
import numpy as np

from .air2s_types import Detection


class ColorTargetDetector:
    """High-contrast orange target detector used by the closed-loop simulator."""

    def __init__(self, label: str = 'target', min_area: int = 120):
        self.label = label
        self.min_area = int(min_area)
        self.lower_hsv = np.array([5, 120, 100], dtype=np.uint8)
        self.upper_hsv = np.array([25, 255, 255], dtype=np.uint8)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(0, self.label, 0.99, (x, y, x + w, y + h)))
        return sorted(detections, key=lambda item: item.area, reverse=True)


class RepositoryYoloDetector:
    """Adapter from the repository's FFCA-YOLO/YOLOv5 runtime to Detection."""

    def __init__(
        self,
        weights: str,
        data: str,
        target_labels: Optional[Sequence[str]] = None,
        imgsz: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
        device: str = '0',
        half: bool = False,
    ):
        from .ffca_detector import FFCAYoloDetector

        self._detector = FFCAYoloDetector(
            weights=weights,
            imgsz=(imgsz, imgsz),
            conf_thres=confidence,
            iou_thres=iou,
            device=device,
            half=half,
            data=data,
        )
        self._target_labels = {label.lower() for label in target_labels or ()}
        self.last_inference_ms = 0.0
        self.last_nms_ms = 0.0

    @property
    def names(self):
        return self._detector.names

    def load(self) -> None:
        self._detector.load_model()

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raw, input_shape, inference_ms, nms_ms = self._detector.infer(frame)
        self.last_inference_ms = float(inference_ms)
        self.last_nms_ms = float(nms_ms)
        if len(raw) == 0:
            return []
        scaled = self._detector.scale_detections(raw, input_shape, frame.shape)
        detections = []
        for x1, y1, x2, y2, confidence, cls in scaled.detach().cpu().tolist():
            class_id = int(cls)
            label = self._name_for_class(class_id)
            if self._target_labels and label.lower() not in self._target_labels:
                continue
            detections.append(
                Detection(class_id, label, float(confidence), (x1, y1, x2, y2))
            )
        return detections

    def _name_for_class(self, class_id: int) -> str:
        names = self._detector.names
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if names is not None and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)
