"""
infer.py
========
Unified AI Inference Engine for Village Street-Survey & Infrastructure Inspection.

Provides modular and independent classes for:
  1. Road Surface Classification  (ConvNeXt-Tiny with automatic Noise Fallback logic)
  2. Solid Waste Detection        (YOLO bounding boxes for trash, dump)
  3. Infrastructure Segmentation  (YOLO-seg polygon masks for road damage, open/closed/broken drainage)
  4. 2D Perspective Visualization (Alpha-blended mask overlays, crisp contours & label badges)

Default Weights:
  - Road Classifier: convnext_road_model_v2.pth
  - YOLO Detector:   detector.pt
  - YOLO Segmentor:  segmentor.pt
"""

import argparse
import json
import math
import os
import re
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

# Albumentations is used if legacy MobileNet checkpoints are loaded
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

from ultralytics import YOLO

# ════════════════════════════════════════════════════════════
# 1. GLOBAL CONFIGURATION & TAXONOMY
# ════════════════════════════════════════════════════════════

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default Production Weights
DEFAULT_ROAD_MODEL = "convnext_road_model_v2.pth"
DEFAULT_DET_MODEL  = "detector.pt"
DEFAULT_SEG_MODEL  = "segmentor.pt"

# Taxonomies
ROAD_CLASSES = ["concrete", "dirt", "noise", "tar"]
VALID_ROAD_CLASSES = {"tar", "cement", "concrete", "gravel", "dirt"}
DET_CLASSES = ["trash", "dump", "open_drainage", "road_damage", "closed_drainage"]

# Inference resolution MUST match training. Every run under runs/ records
# "imgsz: 640" in its args.yaml -- the previous hardcoded 1024 was a 60%
# scale shift against what the models actually learned.
TRAIN_IMGSZ = 640
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Hex / BGR Color palette for rendering annotations
COLOR_PALETTE_BGR = {
    "road_damage": (60, 76, 231),       # Coral / Red in BGR
    "trash": (173, 68, 142),            # Magenta / Purple in BGR
    "dump": (0, 165, 255),              # Orange in BGR
    "garbage": (173, 68, 142),          # Fallback for trash/dump
    "drainage": (133, 160, 22),         # Teal in BGR
    "open_drainage": (255, 215, 60),    # Bright Cyan / Aqua
    "closed_drainage": (240, 140, 70),  # Soft Azure
    "covered_drainage": (240, 140, 70), # Soft Azure
    "broken_drainage": (60, 76, 231),   # Coral
}

# Mapping table: Raw YOLO prediction name -> (spec_label, spec_subtype)
CLASS_MAP = {
    "road_damage": ("road_damage", None),
    "pothole": ("road_damage", None),
    "trash": ("trash", None),
    "dump": ("dump", None),
    "garbage": ("trash", None),
    "open_drainage": ("drainage", "open"),
    "closed_drainage": ("drainage", "covered"),
    "covered_drainage": ("drainage", "covered"),
    "broken_drainage": ("drainage", "broken"),
    "drainage_broken": ("drainage", "broken"),
}


# ════════════════════════════════════════════════════════════
# 2. ROAD SURFACE CLASSIFIERS
# ════════════════════════════════════════════════════════════

class RoadPrediction(dict):
    """Encapsulates road prediction result, supporting dict access and 3-tuple unpacking."""
    def __init__(self, class_name: str, confidence: float, probabilities: dict, fallback_used: bool = False, primary_class: str = ""):
        super().__init__({
            "class": class_name,
            "confidence": confidence,
            "primary_class": primary_class or class_name,
            "fallback_used": fallback_used,
            "probabilities": probabilities
        })
        self.class_name = class_name
        self.confidence = confidence
        self.probabilities = probabilities
        self.fallback_used = fallback_used
        self.primary_class = primary_class or class_name

    def __iter__(self):
        """Allows unpacking as (road_type, confidence, probabilities_dict) for backward compatibility."""
        return iter((self.class_name, self.confidence, self.probabilities))


class ConvNeXtRoadClassifier:
    """
    ConvNeXt-Tiny Road Surface Classifier with automated Noise Fallback Logic.
    
    If the primary prediction is 'noise' (e.g., severe glare, heavy shadow, or occlusion),
    it automatically falls back to the 2nd highest predicted road class.
    """

    def __init__(self, ckpt_path: str = DEFAULT_ROAD_MODEL,
                 class_names: Optional[List[str]] = None,
                 input_res: Optional[int] = None,
                 noise_fallback: bool = True):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"ConvNeXt classifier weights not found: {ckpt_path}")

        # Preprocessing resolution MUST match training. best_convnext_384.pth
        # was trained at 384; running it through a 224 resize is a silent
        # domain shift that costs accuracy with no error. If not given
        # explicitly, infer it from the filename, else fall back to 224.
        if input_res is None:
            m = re.search(r"(\d{3,4})", os.path.basename(ckpt_path))
            input_res = int(m.group(1)) if m and 128 <= int(m.group(1)) <= 1024 else 224
        self.input_res = input_res

        # When the top class is "noise" the fallback reports the RUNNER-UP
        # instead. Useful for a survey where every frame must be assigned a
        # surface, but it means "noise" never appears in the output -- and a
        # frame the model was 90% sure was noise gets reported as, say, tar
        # at 6%. Disable it to match a plain argmax reference implementation.
        self.noise_fallback = noise_fallback

        print(f"[Load] ConvNeXt-Tiny Road Classifier: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        state_dict = raw
        ckpt_classes = None
        if isinstance(raw, dict):
            for key in ("model_state", "state_dict", "model"):
                if key in raw and isinstance(raw[key], dict):
                    state_dict = raw[key]
                    break
            if raw.get("class_names"):
                ckpt_classes = list(raw["class_names"])

        # A checkpoint that records its own class order is authoritative --
        # a wrong order mislabels every frame confidently and silently.
        self.class_names = ckpt_classes or class_names or ROAD_CLASSES
        if ckpt_classes and class_names and ckpt_classes != class_names:
            print(f"       ! checkpoint class order {ckpt_classes} overrides "
                  f"the supplied {class_names}")
        num_classes = len(self.class_names)

        self.model = models.convnext_tiny(weights=None)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f"       ! state_dict mismatch: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys -- check architecture "
                  f"and class count ({num_classes})")

        self.model.to(DEVICE)
        self.model.eval()

        print(f"       Classes: {self.class_names}  |  input {self.input_res}px"
              f"  |  noise fallback {'ON' if self.noise_fallback else 'OFF'}")

        self.transform = transforms.Compose([
            transforms.Resize((self.input_res, self.input_res)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _finalize(self, probs) -> "RoadPrediction":
        """Shared top-1 + noise-fallback logic for single and batch paths."""
        prob_dict = {self.class_names[i]: float(probs[i].item())
                     for i in range(len(self.class_names))}
        top_probs, top_indices = torch.topk(probs, k=len(self.class_names))

        best_label = self.class_names[top_indices[0].item()]
        best_conf = float(top_probs[0].item())

        fallback_used = False
        if self.noise_fallback and best_label == "noise" and len(self.class_names) > 1:
            final_label = self.class_names[top_indices[1].item()]
            final_conf = float(top_probs[1].item())
            fallback_used = True
        else:
            final_label, final_conf = best_label, best_conf

        return RoadPrediction(
            class_name=final_label,
            confidence=final_conf,
            probabilities=prob_dict,
            fallback_used=fallback_used,
            primary_class=best_label,
        )

    def predict_batch(self, bgr_images: List[np.ndarray]) -> List["RoadPrediction"]:
        """Classify several views in one forward pass (e.g. all 4 at once)."""
        if not bgr_images:
            return []
        batch = torch.stack([
            self.transform(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
            for im in bgr_images
        ]).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)
        return [self._finalize(probs[i]) for i in range(probs.shape[0])]

    def predict(self, bgr_img: np.ndarray) -> RoadPrediction:
        """Runs inference on a single BGR OpenCV frame."""
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(Image.fromarray(rgb_img)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(self.model(input_tensor), dim=1)[0]
        return self._finalize(probs)

class MobileNetRoadClassifier:
    """
    MobileNet Road Surface Classifier with automated Noise Fallback Logic.
    Supports both MobileNetV3-Large (e.g. mobilenet_two_stage_final.pth) and legacy MobileNetV2 (.pt).
    """

    def __init__(self, checkpoint_path: str, class_names: Optional[List[str]] = None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Classifier weights not found: {checkpoint_path}")

        print(f"[Load] MobileNet Road Classifier: {checkpoint_path}")
        self.ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        # Extract state dict
        if isinstance(self.ckpt, dict) and "model_state" in self.ckpt:
            state_dict = self.ckpt["model_state"]
        elif isinstance(self.ckpt, dict) and "state_dict" in self.ckpt:
            state_dict = self.ckpt["state_dict"]
        else:
            state_dict = self.ckpt

        # Extract class names
        if class_names:
            self.class_names = class_names
        elif isinstance(self.ckpt, dict) and "class_names" in self.ckpt:
            self.class_names = self.ckpt["class_names"]
        else:
            self.class_names = ROAD_CLASSES  # ["concrete", "dirt", "noise", "tar"]

        num_classes = len(self.class_names)

        # Detect whether it is MobileNetV3 or MobileNetV2
        is_v3 = (
            any("classifier.3" in k for k in state_dict.keys())
            or (
                "features.0.0.weight" in state_dict
                and state_dict["features.0.0.weight"].shape[0] == 16
            )
        )

        if is_v3:
            self.model = models.mobilenet_v3_large(weights=None)
            in_features = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(in_features, num_classes)
        else:
            self.model = models.mobilenet_v2(weights=None)
            in_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_features, num_classes),
            )

        self.model.load_state_dict(state_dict)
        self.model.to(DEVICE)
        self.model.eval()

        arch_name = "MobileNetV3-Large" if is_v3 else "MobileNetV2"
        print(f"       Architecture: {arch_name} | Classes: {self.class_names}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, bgr_img: np.ndarray) -> RoadPrediction:
        """
        Runs inference on a single BGR OpenCV frame.
        Returns RoadPrediction(class_name, confidence, probabilities_dict).
        """
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        input_tensor = self.transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0]

        prob_dict = {self.class_names[i]: float(probs[i].item()) for i in range(len(self.class_names))}
        top_probs, top_indices = torch.topk(probs, k=len(self.class_names))

        best_idx = top_indices[0].item()
        best_label = self.class_names[best_idx]
        best_conf = float(top_probs[0].item())

        # Noise Fallback Logic
        fallback_used = False
        primary_label = best_label
        if best_label == "noise" and len(self.class_names) > 1:
            second_idx = top_indices[1].item()
            final_label = self.class_names[second_idx]
            final_conf = float(top_probs[1].item())
            fallback_used = True
        else:
            final_label = best_label
            final_conf = best_conf

        return RoadPrediction(
            class_name=final_label,
            confidence=final_conf,
            probabilities=prob_dict,
            fallback_used=fallback_used,
            primary_class=primary_label
        )


# ════════════════════════════════════════════════════════════
# 3. YOLO DETECTION & SEGMENTATION ENGINE
# ════════════════════════════════════════════════════════════

class YoloInferenceEngine:
    """
    Batched YOLO Inference Engine for Object Detection and Instance Segmentation.
    
    Matches exact model_detections.py architecture:
      - imgsz=TRAIN_IMGSZ for high-resolution defect detection
      - Direct BGR image processing for Ultralytics
      - Extraction of bounding boxes and polygon contour masks
      - Frontend box_1000 coordinate normalization ([ymin, xmin, ymax, xmax] scaled 0-1000)
      - Keyword fallback mapping and subtype filtering
    """

    def __init__(self, model_path: str = DEFAULT_DET_MODEL, conf_defaults: Optional[Dict[str, float]] = None, iou: float = 0.45, augment: bool = False):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model weights not found: {model_path}")

        print(f"[Load] YOLO Engine: {model_path}")
        self.model = YOLO(model_path)
        self.iou = iou
        # Test-time augmentation: runs each image at several scales and flips
        # and merges the results. Costs roughly 2-3x inference time for a few
        # points of recall. Does not change the weights.
        self.augment = augment
        self.conf_defaults = conf_defaults or {
            "road_damage": 0.15,
            "trash": 0.25,
            "dump": 0.25,
            "garbage": 0.25,
            "drainage": 0.15,
        }
        self.names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.is_seg = hasattr(self.model, "task") and self.model.task == "segment"

    def predict_batch(
        self,
        images: List[np.ndarray],
        conf_thresholds: Optional[Dict[str, float]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Runs YOLO detection/segmentation on a batch of perspective view numpy images.
        Returns a list of detection records per view (Spec §4 requirement).
        """
        if not images:
            return []

        thresholds = self.conf_defaults.copy()
        if conf_thresholds:
            thresholds.update(conf_thresholds)

        # CLASS_MAP maps raw "trash" to label "trash", but main.py passes the
        # threshold under the key "garbage" -- so --conf-garbage never reached
        # trash detections. Mirror whichever key was supplied onto the other.
        if "garbage" in (conf_thresholds or {}) and "trash" not in (conf_thresholds or {}):
            thresholds["trash"] = thresholds["garbage"]
        elif "trash" in (conf_thresholds or {}) and "garbage" not in (conf_thresholds or {}):
            thresholds["garbage"] = thresholds["trash"]

        min_conf = min(thresholds.values())

        # Batched inference call with exact 1024 imgsz resolution
        results = self.model.predict(
            images,
            conf=min_conf,
            iou=self.iou,
            imgsz=TRAIN_IMGSZ,
            augment=self.augment,
            verbose=False
        )

        batch_detections = []

        for result in results:
            h, w = result.orig_shape
            view_dets = []

            has_masks = result.masks is not None
            polygons_n = result.masks.xyn if (has_masks and result.masks.xyn is not None) else []
            polygons_px = result.masks.xy if (has_masks and result.masks.xy is not None) else []

            if result.boxes is not None:
                for idx, box in enumerate(result.boxes):
                    cid = int(box.cls[0])
                    raw_name = self.names.get(cid, f"class_{cid}").lower()
                    raw_conf = float(box.conf[0])

                    # Map raw model class -> (spec_label, spec_subtype)
                    mapping = CLASS_MAP.get(raw_name)
                    if not mapping:
                        # Fallback keyword matching
                        if "damage" in raw_name or "pothole" in raw_name:
                            mapping = ("road_damage", None)
                        elif "trash" in raw_name or "garbage" in raw_name:
                            mapping = ("trash", None)
                        elif "dump" in raw_name:
                            mapping = ("dump", None)
                        elif "open" in raw_name and "drain" in raw_name:
                            mapping = ("drainage", "open")
                        elif "close" in raw_name and "drain" in raw_name:
                            mapping = ("drainage", "covered")
                        elif "broken" in raw_name and "drain" in raw_name:
                            mapping = ("drainage", "broken")
                        elif "drain" in raw_name:
                            mapping = ("drainage", "open")
                        else:
                            continue

                    label, subtype = mapping

                    # Drop drainage detections missing a valid subtype
                    if label == "drainage" and subtype not in {"open", "broken", "covered"}:
                        continue

                    # Filter by per-class confidence threshold
                    class_conf_thresh = thresholds.get(label, 0.25)
                    if raw_conf < class_conf_thresh:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                    # Convert pixel box to normalized box_1000 integer array: [ymin, xmin, ymax, xmax]
                    ymin = int(round(max(0.0, min(1000.0, (y1 / h) * 1000.0))))
                    xmin = int(round(max(0.0, min(1000.0, (x1 / w) * 1000.0))))
                    ymax = int(round(max(0.0, min(1000.0, (y2 / h) * 1000.0))))
                    xmax = int(round(max(0.0, min(1000.0, (x2 / w) * 1000.0))))

                    det_record = {
                        "label": label,
                        "box_1000": [ymin, xmin, ymax, xmax],
                        "confidence": round(raw_conf, 2),
                    }
                    if subtype:
                        det_record["subtype"] = subtype

                    # Internal geometric data for 360 reprojection annotation
                    det_record["_raw_bbox_xyxy"] = [x1, y1, x2, y2]
                    det_record["_view_w"] = w
                    det_record["_view_h"] = h

                    if has_masks and idx < len(polygons_n):
                        poly_norm = [[float(pt[0]), float(pt[1])] for pt in polygons_n[idx]]
                        det_record["_polygon_norm"] = poly_norm
                    if has_masks and idx < len(polygons_px):
                        poly_pixel = [[float(pt[0]), float(pt[1])] for pt in polygons_px[idx]]
                        det_record["_polygon_px"] = poly_pixel

                    view_dets.append(det_record)

            batch_detections.append(view_dets)

        return batch_detections


# ════════════════════════════════════════════════════════════
# 4. 2D PERSPECTIVE VIEW VISUALIZER
# ════════════════════════════════════════════════════════════

def annotate_perspective_view(
    bgr_img: np.ndarray,
    detections: List[Dict[str, Any]],
    road_type: Optional[str] = None
) -> np.ndarray:
    """
    Renders bounding boxes, polygon segmentation masks, and label pills on a 2D perspective crop.
    """
    annotated = bgr_img.copy()
    h, w = annotated.shape[:2]

    # Draw segmentation masks & bounding boxes
    for det in detections:
        label = det["label"]
        subtype = det.get("subtype")
        conf = det.get("confidence", 0.0)

        if label == "drainage" and subtype:
            color_key = f"{subtype}_drainage"
        else:
            color_key = label
        color = COLOR_PALETTE_BGR.get(color_key, COLOR_PALETTE_BGR.get(label, (0, 255, 0)))

        # Draw polygon mask if present
        if "_polygon_norm" in det and len(det["_polygon_norm"]) >= 3:
            pts_norm = np.array(det["_polygon_norm"], dtype=np.float32)
            pts_px = np.zeros_like(pts_norm, dtype=np.int32)
            pts_px[:, 0] = np.int32(np.clip(pts_norm[:, 0] * w, 0, w - 1))
            pts_px[:, 1] = np.int32(np.clip(pts_norm[:, 1] * h, 0, h - 1))

            # Translucent mask overlay
            mask_layer = annotated.copy()
            cv2.fillPoly(mask_layer, [pts_px], color)
            annotated = cv2.addWeighted(mask_layer, 0.30, annotated, 0.70, 0)
            cv2.polylines(annotated, [pts_px], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            min_y_idx = np.argmin(pts_px[:, 1])
            bx, by = pts_px[min_y_idx, 0], pts_px[min_y_idx, 1]
        else:
            # Fallback to bounding box
            if "_raw_bbox_xyxy" in det:
                x1, y1, x2, y2 = [int(v) for v in det["_raw_bbox_xyxy"]]
            else:
                ymin, xmin, ymax, xmax = det["box_1000"]
                x1, y1 = int((xmin / 1000) * w), int((ymin / 1000) * h)
                x2, y2 = int((xmax / 1000) * w), int((ymax / 1000) * h)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            bx, by = x1, y1

        # Draw dark badge with colored accent
        lbl_str = f"{subtype}_drainage" if (label == "drainage" and subtype) else label
        badge_text = f"{lbl_str} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, 1)

        px1 = max(5, min(w - tw - 15, bx))
        py1 = max(5, by - th - 10)
        px2 = px1 + tw + 10
        py2 = py1 + th + 8

        badge_overlay = annotated.copy()
        cv2.rectangle(badge_overlay, (px1, py1), (px2, py2), (20, 20, 20), -1)
        annotated = cv2.addWeighted(badge_overlay, 0.70, annotated, 0.30, 0)
        cv2.rectangle(annotated, (px1, py1), (px1 + 3, py2), color, -1)
        cv2.putText(annotated, badge_text, (px1 + 6, py2 - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw Road Type pill if provided
    if road_type:
        road_text = f"Road: {road_type.capitalize()}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (rtw, rth), _ = cv2.getTextSize(road_text, font, 0.6, 2)
        rx1, ry1 = 15, 15
        rx2, ry2 = rx1 + rtw + 18, ry1 + rth + 14

        road_bg = annotated.copy()
        cv2.rectangle(road_bg, (rx1, ry1), (rx2, ry2), (15, 15, 15), -1)
        annotated = cv2.addWeighted(annotated, 0.25, road_bg, 0.75, 0)
        cv2.rectangle(annotated, (rx1, ry1), (rx1 + 4, ry2), (0, 215, 255), -1)
        cv2.putText(annotated, road_text, (rx1 + 10, ry2 - 6), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return annotated


# ════════════════════════════════════════════════════════════
# 5. STANDALONE 2D INFERENCE CLI RUNNER
# ════════════════════════════════════════════════════════════

def run_standalone_inference():
    """Runs 3-model combined inference on a folder of standard 2D images."""
    parser = argparse.ArgumentParser(description="Run 3-model Combined AI Inference on 2D Images")
    parser.add_argument("--folder", default="test_images", help="Folder of input images")
    parser.add_argument("--output", default="output_annotated", help="Folder to save annotated outputs")
    parser.add_argument("--classifier", default=DEFAULT_ROAD_MODEL, help="Road classifier weights (.pth or .pt)")
    parser.add_argument("--yolo-det", default=DEFAULT_DET_MODEL, help="YOLO detection weights (trash/dump)")
    parser.add_argument("--yolo-seg", default=DEFAULT_SEG_MODEL, help="YOLO segmentation weights (damage/drainage)")
    parser.add_argument("--det-conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--seg-conf", type=float, default=0.25, help="Segmentation confidence threshold")
    parser.add_argument("--json", action="store_true", help="Save detection JSON records")
    args = parser.parse_args()

    in_dir = Path(args.folder)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        print(f"Error: Input folder not found: {in_dir}")
        return

    # Load Models
    if str(args.classifier).endswith(".pth") or "convnext" in str(args.classifier):
        classifier = ConvNeXtRoadClassifier(args.classifier)
    else:
        classifier = MobileNetRoadClassifier(args.classifier)

    det_engine = YoloInferenceEngine(args.yolo_det)
    seg_engine = YoloInferenceEngine(args.yolo_seg)

    image_files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"Found {len(image_files)} images in {in_dir}. Running inference...")

    for img_path in image_files:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        # 1. Road classification
        road_pred = classifier.predict(bgr)

        # 2. Batched YOLO detection & segmentation
        det_dets = det_engine.predict_batch([bgr], conf_thresholds={"garbage": args.det_conf})[0]
        seg_dets = seg_engine.predict_batch([bgr], conf_thresholds={"road_damage": args.seg_conf, "drainage": args.seg_conf})[0]

        # Prioritize segmentation masks over duplicate boxes
        combined_dets = list(seg_dets)
        for d in det_dets:
            if d["label"] not in {"road_damage", "drainage"}:
                combined_dets.append(d)

        # Annotate image
        annotated = annotate_perspective_view(bgr, combined_dets, road_type=road_pred.class_name)
        out_path = out_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)

        if args.json:
            json_path = out_dir / f"{img_path.stem}.json"
            record = {
                "file": img_path.name,
                "road_type": road_pred,
                "detections": combined_dets
            }
            json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        print(f"  [Done] {img_path.name} | Road: {road_pred.class_name} ({road_pred.confidence:.2f}) | Objects: {len(combined_dets)}")

    print(f"\nAll images processed. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    run_standalone_inference()