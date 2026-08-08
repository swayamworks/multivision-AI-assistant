"""Local drowsiness and age-estimation inference pipeline."""

import os

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
)
from ultralytics import YOLO
from mtcnn import MTCNN


def _build_drowsiness_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3), nn.Linear(model.classifier[1].in_features, 1)
    )
    return model, weights.transforms()


def _build_age_model():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
    return model, weights.transforms()


def _load_weights(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=True)


class DrowsinessDetector:
    """Runs the local YOLOv8, EfficientNet-B0, and custom MobileNetV3 models."""

    def __init__(self):
        module_dir = os.path.dirname(__file__)
        project_dir = os.path.dirname(os.path.dirname(module_dir))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo = YOLO(os.path.join(project_dir, "yolov8n.pt"))

        drowsiness_path = os.path.join(module_dir, "best_model.pt")
        if not os.path.exists(drowsiness_path):
            drowsiness_path = os.path.join(module_dir, "final_weights.pt")
        age_path = os.path.join(module_dir, "models", "age_estimator_best.pt")
        if not os.path.exists(drowsiness_path):
            raise FileNotFoundError("Drowsiness weights are missing.")
        if not os.path.exists(age_path):
            raise FileNotFoundError("Custom age weights are missing: models/age_estimator_best.pt")

        self.drowsiness_model, self.drowsiness_transform = _build_drowsiness_model()
        _load_weights(self.drowsiness_model, drowsiness_path, self.device)
        self.drowsiness_model.to(self.device).eval()

        self.age_model, self.age_transform = _build_age_model()
        _load_weights(self.age_model, age_path, self.device)
        self.age_model.to(self.device).eval()

        self.face_detector = MTCNN()
        print(f"[DrowsinessDetector] Local models ready on {self.device}")

    def _classify_drowsiness(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return False, 0.0
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        tensor = self.drowsiness_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            awake_probability = torch.sigmoid(self.drowsiness_model(tensor)).item()
        sleeping = awake_probability < 0.5
        return sleeping, (1 - awake_probability if sleeping else awake_probability)

    def _extract_face(self, person_crop):
        if person_crop is None or person_crop.size == 0:
            return None
        rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        faces = self.face_detector.detect_faces(rgb)
        if not faces:
            return None
        # Find the largest face by bounding box area
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = largest_face['box']
        # MTCNN can return negative coordinates if face is near edge
        x, y = max(0, x), max(0, y)
        padding = int(max(w, h) * 0.16)
        y1, y2 = max(0, y - padding), min(person_crop.shape[0], y + h + padding)
        x1, x2 = max(0, x - padding), min(person_crop.shape[1], x + w + padding)
        return person_crop[y1:y2, x1:x2]

    def estimate_age(self, person_crop):
        """Return an approximate age from the locally trained UTKFace regressor."""
        face = self._extract_face(person_crop)
        if face is None or face.size == 0:
            return None
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = self.age_transform(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            age = self.age_model(tensor).squeeze().item()
        return int(round(max(0, min(100, age))))

    def process_image(self, image):
        annotated = image.copy()
        height, width = image.shape[:2]
        boxes = []
        for result in self.yolo(image, verbose=False):
            for box in result.boxes:
                if int(box.cls[0]) == 0 and float(box.conf[0]) >= 0.25:
                    boxes.append(tuple(map(int, box.xyxy[0])))
        if not boxes:
            boxes = [(0, 0, width, height)]

        details, sleeping_count, awake_count = [], 0, 0
        for person_id, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
            crop = image[y1:y2, x1:x2]
            sleeping, drowsiness_confidence = self._classify_drowsiness(crop)
            age = self.estimate_age(crop)
            status = "Sleeping" if sleeping else "Awake"
            color = (0, 0, 255) if sleeping else (0, 255, 0)
            sleeping_count += int(sleeping)
            awake_count += int(not sleeping)
            details.append({
                "person_id": person_id,
                "status": status,
                "age": age,
                "age_confidence": None,
                "drowsiness_confidence": drowsiness_confidence,
                "bbox": (x1, y1, x2, y2),
            })
            age_text = f"Age ~{age}" if age is not None else "Age unavailable"
            label = f"#{person_id} {status.upper()} | {age_text} ({drowsiness_confidence * 100:.0f}%)"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3 if sleeping else 2)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            text_y = max(0, y1 - text_height - 10)
            cv2.rectangle(annotated, (x1, text_y), (x1 + text_width + 8, text_y + text_height + 8), color, -1)
            cv2.putText(annotated, label, (x1 + 4, text_y + text_height + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2)

        sleeping_details = [detail for detail in details if detail["status"] == "Sleeping"]
        return annotated, {
            "total_people": len(details),
            "sleeping_count": sleeping_count,
            "awake_count": awake_count,
            "sleeping_details": sleeping_details,
            "occupant_details": details,
        }
