"""
main.py
=======
Village Street-Survey Detection Pipeline CLI (Spec Compliant)

Features:
  1. CLI inputs: --input, --output, --gpx, --fps, --views, --village, etc.
  2. 360° Equirectangular pinhole perspective view extractor (penta, tri, forward).
  3. Batched multi-model detection & classification using model_detections.py.
  4. Perspective-to-equirectangular inverse reprojection & annotation rendering with
     back-view 180° seam unwrapping.
  5. GPX track linear interpolation & haversine distance attribution with noise clamping.
  6. Frontend JSON contract export ({village}_detections.json, {village}_kpi.json, {village}_kpi.txt).
  7. Rotating file logging & resumability checkpointing (.processed.jsonl).
"""

import argparse
import json
import math
import os
import re
import sys
import logging
from collections import Counter
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta
from fractions import Fraction
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image, ExifTags
import torch

try:
    from temporal_gating import gate_sequence, summarize as gating_summary
except ImportError:
    gate_sequence, gating_summary = None, None
from inference_engine import (
    ConvNeXtRoadClassifier,
    MobileNetRoadClassifier,
    YoloInferenceEngine,
    annotate_perspective_view
)
from track_interpolator import TemporalTrackInterpolator, annotate_interpolated_on_image

# ── VIEW CONFIGURATIONS (§2 Spec Table) ──
# -- GEOMETRY MUST MATCH split_views.py (the training-crop generator) --
# Confirmed against the training data: square 1024x1024 crops,
# FOV 100 deg, pitch -20 deg, yaw 0/90/180/270 (front/right/back/left).
#
# Why SQUARE matters: vertical FOV is not set directly, it falls out of the
# aspect ratio via f = 0.5*w/tan(fov/2). A square crop at 100 deg gives
# 100 deg vertically. The previous 1280x960 at 90 deg gave only 73.7 deg --
# a far tighter view of the road surface than the models ever saw.
#
# The ~10 deg overlap between adjacent views (100 deg FOV, 90 deg spacing)
# is deliberate: an object near a boundary is captured whole by at least one
# view. The panoramic NMS below removes the resulting duplicates.

# ── EXCLUSIVE CLASS OWNERSHIP ────────────────────────────────
# Each class is drawn by exactly ONE model. The detection model is 5-class
# and also predicts road_damage and open_drainage, but the segmentation
# model owns those -- otherwise, every time segmentation misses an instance
# (recall ~0.53) the detector's BOX survives and gets drawn instead of a
# mask. Identified as (label, subtype) because CLASS_MAP collapses both
# drainage classes to label "drainage".
SEG_OWNED = {
    ("road_damage", None),
    ("drainage", "open"),
}

TRAIN_FOV = 100.0
TRAIN_PITCH = -20.0
TRAIN_VIEW_SIZE = 1024          # square, as in training

VIEW_CONFIGS = {
    "front": {"yaw": 0.0,   "pitch": TRAIN_PITCH, "fov": TRAIN_FOV},
    "right": {"yaw": 90.0,  "pitch": TRAIN_PITCH, "fov": TRAIN_FOV},
    "back":  {"yaw": 180.0, "pitch": TRAIN_PITCH, "fov": TRAIN_FOV},
    "left":  {"yaw": 270.0, "pitch": TRAIN_PITCH, "fov": TRAIN_FOV},
}

VIEW_SETS = {
    "quad": ["front", "right", "back", "left"],   # matches training -- default
    "tri": ["front", "right", "left"],
    "forward": ["front"],
}

# Luminous modern color palette in BGR format (avoiding yellow/green, high clarity & light glow)
COLOR_PALETTE = {
    "open_drainage": (255, 215, 60),     # Electric Aqua / Bright Cyan (Light luminous cyan)
    "closed_drainage": (240, 140, 70),   # Soft Sky Blue / Azure
    "covered_drainage": (240, 140, 70),  # Soft Sky Blue / Azure
    "road_damage": (235, 90, 220),       # Bright Orchid / Magenta
    "dump": (250, 140, 180),             # Light Lavender / Violet
    "trash": (140, 110, 255),            # Bright Coral / Soft Salmon Pink
    "garbage": (140, 110, 255),          # Fallback
    "drainage": (255, 215, 60),          # Fallback
}


# ════════════════════════════════════════════════════════════
# 1. PERSPECTIVE VIEW EXTRACTION & INVERSE REPROJECTION
# ════════════════════════════════════════════════════════════

class EquirectangularProjector:
    """Equirectangular pinhole projection & inverse reprojection engine."""

    def __init__(self, view_width: int = TRAIN_VIEW_SIZE,
                 view_height: int = TRAIN_VIEW_SIZE):
        self.vw = view_width
        self.vh = view_height

    def extract_view(
        self,
        eq_img: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float
    ) -> np.ndarray:
        """Extracts a flat perspective view from an equirectangular 360 image."""
        eq_h, eq_w = eq_img.shape[:2]

        f = 0.5 * self.vw / math.tan(math.radians(fov_deg) / 2.0)

        x_c = np.arange(self.vw, dtype=np.float32) - (self.vw / 2.0)
        y_c = (self.vh / 2.0) - np.arange(self.vh, dtype=np.float32)
        x_grid, y_grid = np.meshgrid(x_c, y_c)
        z_grid = np.full_like(x_grid, f)

        # Normalize camera ray vectors
        norm = np.sqrt(x_grid**2 + y_grid**2 + z_grid**2)
        x_norm = x_grid / norm
        y_norm = y_grid / norm
        z_norm = z_grid / norm

        # Rotation matrices
        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)

        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        # Apply pitch rotation (around X) then yaw rotation (around Y)
        y_p = y_norm * cos_p + z_norm * sin_p
        z_p = -y_norm * sin_p + z_norm * cos_p
        x_p = x_norm

        # Yaw: x'' = x'*cos + z'*sin, z'' = -x'*sin + z'*cos
        x_w = x_p * cos_y + z_p * sin_y
        y_w = y_p
        z_w = -x_p * sin_y + z_p * cos_y

        # Convert to spherical coordinates
        lon = np.arctan2(x_w, z_w)
        lat = np.arcsin(np.clip(y_w, -1.0, 1.0))

        # Map spherical coordinates to equirectangular pixel coordinates
        u = (lon / (2.0 * np.pi) + 0.5) * eq_w
        v = (0.5 - lat / np.pi) * eq_h

        map_x = u.astype(np.float32)
        map_y = v.astype(np.float32)

        return cv2.remap(eq_img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def perspective_to_equirect_pts(
        self,
        pts_view: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        eq_w: int,
        eq_h: int
    ) -> np.ndarray:
        """Inverse maps points from perspective view pixel space (x, y) to equirectangular (u, v)."""
        f = 0.5 * self.vw / math.tan(math.radians(fov_deg) / 2.0)

        x_c = pts_view[:, 0] - (self.vw / 2.0)
        y_c = (self.vh / 2.0) - pts_view[:, 1]
        z_c = np.full_like(x_c, f)

        norm = np.sqrt(x_c**2 + y_c**2 + z_c**2)
        x_norm = x_c / norm
        y_norm = y_c / norm
        z_norm = z_c / norm

        pitch_rad = math.radians(pitch_deg)
        yaw_rad = math.radians(yaw_deg)

        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)

        y_p = y_norm * cos_p + z_norm * sin_p
        z_p = -y_norm * sin_p + z_norm * cos_p
        x_p = x_norm

        x_w = x_p * cos_y + z_p * sin_y
        y_w = y_p
        z_w = -x_p * sin_y + z_p * cos_y

        lon = np.arctan2(x_w, z_w)
        lat = np.arcsin(np.clip(y_w, -1.0, 1.0))

        u = (lon / (2.0 * np.pi) + 0.5) * eq_w
        v = (0.5 - lat / np.pi) * eq_h

        return np.column_stack([u, v])


def detection_bearing(det, view_name, projector):
    """
    Centre of a detection as a (yaw, pitch) bearing in degrees.

    Passing eq_w=360, eq_h=180 to the inverse projection makes the returned
    "pixel" coords degrees directly, so the bearing is independent of the
    equirect image resolution.
    """
    cfg = VIEW_CONFIGS[view_name]
    if "_polygon_norm" in det and len(det["_polygon_norm"]) >= 3:
        pts = np.array(det["_polygon_norm"], dtype=np.float32)
        cx = float(pts[:, 0].mean()) * projector.vw
        cy = float(pts[:, 1].mean()) * projector.vh
    else:
        x1, y1, x2, y2 = det["_raw_bbox_xyxy"]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

    uv = projector.perspective_to_equirect_pts(
        np.array([[cx, cy]], dtype=np.float32),
        cfg["yaw"], cfg["pitch"], cfg["fov"], 360.0, 180.0)
    return float(uv[0, 0]) - 180.0, 90.0 - float(uv[0, 1])


def apply_intra_view_nms(detections: List[Dict[str, Any]], iou_thresh: float = 0.65) -> List[Dict[str, Any]]:
    """
    Suppresses redundant overlapping detections of the same label & subtype within a single view.
    Prioritizes:
      1. Confirmed detections over weak candidates (not is_candidate).
      2. Polygon segmentation masks over bounding-box-only detections.
      3. Higher confidence score.
    """
    if len(detections) <= 1:
        return detections

    sorted_dets = sorted(
        detections,
        key=lambda d: (
            not d.get("is_candidate", False),
            "_polygon_norm" in d and len(d.get("_polygon_norm", [])) >= 3,
            d.get("confidence", 0.0)
        ),
        reverse=True
    )

    kept: List[Dict[str, Any]] = []
    for d in sorted_dets:
        b1 = d["box_1000"]
        lbl1 = d["label"]
        st1 = d.get("subtype")

        is_dup = False
        for k in kept:
            if k["label"] == lbl1 and k.get("subtype") == st1:
                b2 = k["box_1000"]
                inter_ymin = max(b1[0], b2[0])
                inter_xmin = max(b1[1], b2[1])
                inter_ymax = min(b1[2], b2[2])
                inter_xmax = min(b1[3], b2[3])
                if inter_ymax > inter_ymin and inter_xmax > inter_xmin:
                    inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
                    area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
                    area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
                    union_area = area1 + area2 - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0.0
                    if iou > iou_thresh:
                        is_dup = True
                        break
        if not is_dup:
            kept.append(d)

    return kept


def smooth_road_surface_sequence(records: List[Dict[str, Any]], window_size: int = 5) -> None:
    """
    Smoothes momentary 1-2 frame road surface classification flickers (e.g. noise fallback to dirt)
    using a sliding window majority vote over adjacent frames.
    """
    if len(records) < 3:
        return

    n = len(records)
    original_types = [r["road_type"]["road_type"] for r in records]
    confs = [r["road_type"].get("confidence", 0.0) for r in records]
    fallbacks = [r["road_type"].get("fallback_used", False) for r in records]

    for i in range(n):
        # Only smooth frames that were low-confidence or noise fallbacks
        if not fallbacks[i] and confs[i] >= 0.70:
            continue

        start_idx = max(0, i - window_size // 2)
        end_idx = min(n, i + window_size // 2 + 1)

        # High confidence votes around this frame
        neighborhood = [
            original_types[j] for j in range(start_idx, end_idx)
            if j != i and (not fallbacks[j] or confs[j] >= 0.50)
        ]
        if neighborhood:
            most_common = Counter(neighborhood).most_common(1)[0][0]
            if most_common != original_types[i]:
                records[i]["road_type"]["smoothed_from"] = original_types[i]
                records[i]["road_type"]["road_type"] = most_common


def deduplicate_cross_view_detections(
    frame_record: Dict[str, Any],
    projector: EquirectangularProjector,
    eq_w: int = 1024,
    eq_h: int = 1024
) -> None:
    """
    Identifies duplicate detections sitting across the 90° boundary seams of adjacent views.
    Tags duplicate entries with 'cross_view_dup': True so KPI summary does not double-count them.
    """
    reprojected = []
    for v in frame_record.get("views", []):
        vname = v["view"]
        cfg = VIEW_CONFIGS[vname]
        for det in v.get("detections", []):
            if "_polygon_norm" in det and len(det["_polygon_norm"]) >= 3:
                raw_pts = np.array(det["_polygon_norm"], dtype=np.float32)
                raw_pts[:, 0] *= projector.vw
                raw_pts[:, 1] *= projector.vh
                sampled_pts = []
                for i in range(len(raw_pts)):
                    p1 = raw_pts[i]
                    p2 = raw_pts[(i + 1) % len(raw_pts)]
                    for t in np.linspace(0, 1, 8, endpoint=False):
                        sampled_pts.append(p1 * (1 - t) + p2 * t)
                sampled_pts = np.array(sampled_pts, dtype=np.float32)
            else:
                ymin, xmin, ymax, xmax = det["box_1000"]
                x1, y1 = (xmin / 1000.0) * projector.vw, (ymin / 1000.0) * projector.vh
                x2, y2 = (xmax / 1000.0) * projector.vw, (ymax / 1000.0) * projector.vh
                sampled_pts = []
                for t in np.linspace(0, 1, 20, endpoint=False):
                    sampled_pts.append([x1 + t * (x2 - x1), y1])
                for t in np.linspace(0, 1, 20, endpoint=False):
                    sampled_pts.append([x2, y1 + t * (y2 - y1)])
                for t in np.linspace(0, 1, 20, endpoint=False):
                    sampled_pts.append([x2 - t * (x2 - x1), y2])
                for t in np.linspace(0, 1, 20, endpoint=False):
                    sampled_pts.append([x1, y2 - t * (y2 - y1)])
                sampled_pts = np.array(sampled_pts, dtype=np.float32)

            eq_pts = projector.perspective_to_equirect_pts(sampled_pts, cfg["yaw"], cfg["pitch"], cfg["fov"], eq_w, eq_h)
            u_min, u_max = float(np.min(eq_pts[:, 0])), float(np.max(eq_pts[:, 0]))
            v_min, v_max = float(np.min(eq_pts[:, 1])), float(np.max(eq_pts[:, 1]))

            reprojected.append({
                "view": vname,
                "det": det,
                "bbox_eq": [u_min, v_min, u_max, v_max],
                "conf": det.get("confidence", 0.0),
                "has_mask": "_polygon_norm" in det
            })

    # Sort so confident / mask detections are kept
    reprojected.sort(key=lambda item: (item["has_mask"], item["conf"]), reverse=True)

    kept_items = []
    for item in reprojected:
        u1, v1, u2, v2 = item["bbox_eq"]
        det = item["det"]
        is_dup = False
        for kept in kept_items:
            if kept["view"] != item["view"] and kept["det"]["label"] == det["label"] and kept["det"].get("subtype") == det.get("subtype"):
                ku1, kv1, ku2, kv2 = kept["bbox_eq"]
                iu1, iv1 = max(u1, ku1), max(v1, kv1)
                iu2, iv2 = min(u2, ku2), min(v2, kv2)
                if iu2 > iu1 and iv2 > iv1:
                    i_area = (iu2 - iu1) * (iv2 - iv1)
                    item_area = max(1.0, (u2 - u1) * (v2 - v1))
                    kept_area = max(1.0, (ku2 - ku1) * (kv2 - kv1))
                    min_area = min(item_area, kept_area)
                    union_area = item_area + kept_area - i_area
                    iou = i_area / union_area if union_area > 0 else 0.0
                    iomin = i_area / min_area if min_area > 0 else 0.0
                    if iou > 0.15 or iomin > 0.30:
                        is_dup = True
                        det["cross_view_dup"] = True
                        break
        if not is_dup:
            kept_items.append(item)


# ════════════════════════════════════════════════════════════
# 2. ANNOTATION OVERLAY & SEAM UNWRAPPING (§5 Spec)
# ════════════════════════════════════════════════════════════

def render_equirect_annotations(
    eq_img: np.ndarray,
    frame_road_type: str,
    all_view_detections: List[Tuple[str, Dict[str, Any]]],
    projector: EquirectangularProjector
) -> np.ndarray:
    """
    Draws re-projected detection polygons, label pills, and canonical road type header pill onto original equirect image.
    Applies panoramic spatial NMS to eliminate overlapping duplicate annotations across adjacent views.
    """
    annotated = eq_img.copy()
    eq_h, eq_w = annotated.shape[:2]

    # Pre-calculate equirectangular polygons & bounding boxes for spatial NMS
    reprojected_items = []
    for view_name, det in all_view_detections:
        cfg = VIEW_CONFIGS[view_name]
        yaw, pitch, fov = cfg["yaw"], cfg["pitch"], cfg["fov"]
        label = det["label"]

        if "_polygon_norm" in det and len(det["_polygon_norm"]) >= 3:
            raw_pts = np.array(det["_polygon_norm"], dtype=np.float32)
            raw_pts[:, 0] *= projector.vw
            raw_pts[:, 1] *= projector.vh
            sampled_pts = []
            for i in range(len(raw_pts)):
                p1 = raw_pts[i]
                p2 = raw_pts[(i + 1) % len(raw_pts)]
                for t in np.linspace(0, 1, 8, endpoint=False):
                    sampled_pts.append(p1 * (1 - t) + p2 * t)
            sampled_pts = np.array(sampled_pts, dtype=np.float32)
        else:
            x1, y1, x2, y2 = det["_raw_bbox_xyxy"]
            sampled_pts = []
            for t in np.linspace(0, 1, 25, endpoint=False):
                sampled_pts.append([x1 + t * (x2 - x1), y1])
            for t in np.linspace(0, 1, 25, endpoint=False):
                sampled_pts.append([x2, y1 + t * (y2 - y1)])
            for t in np.linspace(0, 1, 25, endpoint=False):
                sampled_pts.append([x2 - t * (x2 - x1), y2])
            for t in np.linspace(0, 1, 25, endpoint=False):
                sampled_pts.append([x1, y2 - t * (y2 - y1)])
            sampled_pts = np.array(sampled_pts, dtype=np.float32)

        eq_pts = projector.perspective_to_equirect_pts(sampled_pts, yaw, pitch, fov, eq_w, eq_h)
        u_min, u_max = float(np.min(eq_pts[:, 0])), float(np.max(eq_pts[:, 0]))
        v_min, v_max = float(np.min(eq_pts[:, 1])), float(np.max(eq_pts[:, 1]))

        reprojected_items.append({
            "view_name": view_name,
            "det": det,
            "eq_pts": eq_pts,
            "bbox_eq": [u_min, v_min, u_max, v_max],
            "has_mask": "_polygon_norm" in det,
            "conf": det.get("confidence", 0.0),
            "label": label,
        })

    # Sort items: prioritization by (has_mask=True, higher confidence)
    reprojected_items.sort(key=lambda item: (item["has_mask"], item["conf"]), reverse=True)

    # Perform Panoramic Spatial NMS across adjacent views
    filtered_items = []
    for item in reprojected_items:
        u1, v1, u2, v2 = item["bbox_eq"]
        item_subtype = item["det"].get("subtype")
        is_dup = False
        for kept in filtered_items:
            # Match both label and subtype (e.g. open vs covered drainage)
            if kept["label"] == item["label"] and kept["det"].get("subtype") == item_subtype:
                ku1, kv1, ku2, kv2 = kept["bbox_eq"]
                iu1, iv1 = max(u1, ku1), max(v1, kv1)
                iu2, iv2 = min(u2, ku2), min(v2, kv2)
                if iu2 > iu1 and iv2 > iv1:
                    i_area = (iu2 - iu1) * (iv2 - iv1)
                    item_area = max(1.0, (u2 - u1) * (v2 - v1))
                    kept_area = max(1.0, (ku2 - ku1) * (kv2 - kv1))
                    min_area = min(item_area, kept_area)
                    union_area = item_area + kept_area - i_area
                    iou = i_area / union_area if union_area > 0 else 0.0
                    iomin = i_area / min_area if min_area > 0 else 0.0
                    if iou > 0.15 or iomin > 0.25:
                        is_dup = True
                        break
        if not is_dup:
            filtered_items.append(item)

    for item in filtered_items:
        view_name = item["view_name"]
        det = item["det"]
        label = item["label"]
        subtype = det.get("subtype")

        if label == "drainage" and subtype:
            class_key = f"{subtype}_drainage"
        else:
            class_key = label

        color = COLOR_PALETTE.get(class_key, COLOR_PALETTE.get(label, (0, 255, 0)))
        eq_pts = item["eq_pts"]

        # Seam handling for 180° back view (§5 Spec Step 4)
        x_coords = eq_pts[:, 0]
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        poly_instances = []
        if (x_max - x_min) > (0.5 * eq_w):
            # Unwrap polygon across left/right seam
            unwrapped_pts = eq_pts.copy()
            unwrapped_pts[unwrapped_pts[:, 0] < (0.5 * eq_w), 0] += eq_w
            poly_instances.append(unwrapped_pts)

            shifted_pts = unwrapped_pts.copy()
            shifted_pts[:, 0] -= eq_w
            poly_instances.append(shifted_pts)
        else:
            poly_instances.append(eq_pts)

        for poly_pts in poly_instances:
            pts_int = np.int32([poly_pts])
            # Soft translucent mask fill per polygon instance (0.22 opacity)
            mask_layer = annotated.copy()
            cv2.fillPoly(mask_layer, pts_int, color)
            annotated = cv2.addWeighted(mask_layer, 0.22, annotated, 0.78, 0)
            # Draw fine anti-aliased outline
            cv2.polylines(annotated, pts_int, isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

            # Draw Label Badge near top of polygon
            min_y_idx = np.argmin(poly_pts[:, 1])
            pill_x = int(poly_pts[min_y_idx, 0])
            pill_y = int(poly_pts[min_y_idx, 1]) - 10

            conf_val = item.get("conf", 0.0)
            if label == "drainage" and subtype:
                label_text = f"{subtype}_drainage {conf_val:.2f}"
            else:
                label_text = f"{label} {conf_val:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            (t_w, t_h), baseline = cv2.getTextSize(label_text, font, font_scale, 1)

            pill_x1 = max(10, min(eq_w - t_w - 18, pill_x - 8))
            pill_y1 = max(10, pill_y - t_h - 8)
            pill_x2 = pill_x1 + t_w + 14
            pill_y2 = pill_y1 + t_h + 8

            # Sleek semi-transparent dark badge with colored accent strip
            pill_overlay = annotated.copy()
            cv2.rectangle(pill_overlay, (pill_x1, pill_y1), (pill_x2, pill_y2), (20, 20, 20), -1)
            annotated = cv2.addWeighted(pill_overlay, 0.65, annotated, 0.35, 0)
            cv2.rectangle(annotated, (pill_x1, pill_y1), (pill_x1 + 3, pill_y2), color, -1)
            # Crisp white text
            cv2.putText(annotated, label_text, (pill_x1 + 7, pill_y2 - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw Canonical Road-type pill once per frame at horizontal centre, ~62% of image height (§5 Spec Step 4)
    road_text = f"Road: {frame_road_type.capitalize()}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    font_thresh = 2
    (rt_w, rt_h), rt_base = cv2.getTextSize(road_text, font, font_scale, font_thresh)

    center_x = int(eq_w * 0.5)
    center_y = int(eq_h * 0.62)

    rx1 = center_x - int(rt_w / 2) - 14
    ry1 = center_y - int(rt_h / 2) - 8
    rx2 = center_x + int(rt_w / 2) + 14
    ry2 = center_y + int(rt_h / 2) + 8

    # Dark rounded background with gold accent strip
    road_bg = annotated.copy()
    cv2.rectangle(road_bg, (rx1, ry1), (rx2, ry2), (15, 15, 15), -1)
    annotated = cv2.addWeighted(annotated, 0.25, road_bg, 0.75, 0)
    cv2.rectangle(annotated, (rx1, ry1), (rx1 + 5, ry2), (0, 215, 255), -1)
    cv2.putText(annotated, road_text, (rx1 + 14, ry2 - 8), font, font_scale, (255, 255, 255), font_thresh, cv2.LINE_AA)

    return annotated


# ════════════════════════════════════════════════════════════
# 3. GPX PARSING & HAVERSINE ATTRIBUTION (§6 Spec)
# ════════════════════════════════════════════════════════════

def parse_gpx(gpx_path: str) -> List[Dict[str, Any]]:
    """Parses GPX file trkpt elements with lat, lon, and ISO timestamp."""
    tree = ET.parse(gpx_path)
    root = tree.getroot()

    trkpts = []

    # Strip XML namespace tags for robust element finding
    for elem in root.iter():
        if elem.tag.endswith("trkpt"):
            lat_str = elem.attrib.get("lat")
            lon_str = elem.attrib.get("lon")
            if lat_str is not None and lon_str is not None:
                lat = float(lat_str)
                lon = float(lon_str)
                time_dt = None
                for child in elem:
                    if child.tag.endswith("time") and child.text:
                        dt_str = child.text.strip()
                        time_dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                        break
                if time_dt is not None:
                    trkpts.append({"lat": lat, "lon": lon, "time": time_dt})

    trkpts.sort(key=lambda x: x["time"])
    return trkpts


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes distance in meters between two lat/lon coordinates."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def attribute_gpx_data(
    frame_index: int,
    index_base: int,
    fps: float,
    gpx_track: List[Dict[str, Any]],
    jitter_m: float = 0.5
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Linearly interpolates frame (lat, lon) and returns (iso_timestamp, lat, lon)."""
    if not gpx_track:
        return None, None, None

    start_time = gpx_track[0]["time"]
    frame_offset_sec = (frame_index - index_base) / fps
    frame_time = start_time + timedelta(seconds=frame_offset_sec)
    iso_timestamp = frame_time.isoformat()

    if frame_time < gpx_track[0]["time"] or frame_time > gpx_track[-1]["time"]:
        return iso_timestamp, None, None

    # Find surrounding track points for linear interpolation
    for i in range(len(gpx_track) - 1):
        t1, t2 = gpx_track[i]["time"], gpx_track[i + 1]["time"]
        if t1 <= frame_time <= t2:
            dt_total = (t2 - t1).total_seconds()
            if dt_total <= 0:
                return iso_timestamp, gpx_track[i]["lat"], gpx_track[i]["lon"]
            alpha = (frame_time - t1).total_seconds() / dt_total
            lat = gpx_track[i]["lat"] + alpha * (gpx_track[i + 1]["lat"] - gpx_track[i]["lat"])
            lon = gpx_track[i]["lon"] + alpha * (gpx_track[i + 1]["lon"] - gpx_track[i]["lon"])
            return iso_timestamp, round(lat, 6), round(lon, 6)

    return iso_timestamp, None, None


def read_exif_bytes(path: Path) -> Optional[bytes]:
    """Raw EXIF blob from a source frame, or None if it has none.

    Insta360 frames carry GPS (lat/lon/altitude + GPSDateStamp) in raw EXIF.
    PIL re-encodes on save and drops all of it unless we hand the exact raw binary
    blob back, which is essential for Mapillary Uploader to track every frame.
    """
    try:
        with Image.open(path) as im:
            return im.info.get("exif")
    except Exception:
        return None


def decimal_to_dms_rational(deg_float: float) -> Tuple[Fraction, Fraction, Fraction]:
    """Converts decimal degrees to EXIF GPS rational DMS tuple format using Fraction."""
    deg = int(abs(deg_float))
    min_float = (abs(deg_float) - deg) * 60.0
    minute = int(min_float)
    sec_float = (min_float - minute) * 60.0
    return (Fraction(deg, 1), Fraction(minute, 1), Fraction(int(round(sec_float * 10000)), 10000))


def save_annotated_frame_with_exif(
    bgr_img: np.ndarray,
    out_path: Path,
    exif_bytes: Optional[bytes] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    iso_ts: Optional[str] = None,
    quality: int = 92
) -> None:
    """Saves annotated image carrying over the exact source raw EXIF binary blob for Mapillary."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    if exif_bytes:
        try:
            pil_img.save(str(out_path), quality=quality, exif=exif_bytes)
            return
        except Exception:
            pass

    # Fallback if raw EXIF was not present in source frame but we have GPX coordinates
    exif = pil_img.getexif()
    if lat is not None and lon is not None:
        try:
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            lat_ref = "N" if lat >= 0 else "S"
            lat_dms = decimal_to_dms_rational(lat)
            gps_ifd[ExifTags.GPS.GPSLatitudeRef] = lat_ref
            gps_ifd[ExifTags.GPS.GPSLatitude] = lat_dms

            lon_ref = "E" if lon >= 0 else "W"
            lon_dms = decimal_to_dms_rational(lon)
            gps_ifd[ExifTags.GPS.GPSLongitudeRef] = lon_ref
            gps_ifd[ExifTags.GPS.GPSLongitude] = lon_dms

            if iso_ts:
                dt = datetime.fromisoformat(iso_ts)
                gps_ifd[ExifTags.GPS.GPSDateStamp] = dt.strftime("%Y:%m:%d")
                gps_ifd[ExifTags.GPS.GPSTimeStamp] = (Fraction(dt.hour, 1), Fraction(dt.minute, 1), Fraction(dt.second, 1))
            pil_img.save(str(out_path), quality=quality, exif=exif)
            return
        except Exception as e:
            logging.getLogger("village_pipeline").warning(f"Failed to encode EXIF GPS tags: {e}")

    pil_img.save(str(out_path), quality=quality)


# ════════════════════════════════════════════════════════════
# 4. KPI SUMMARY CALCULATOR (§7.2 Spec)
# ════════════════════════════════════════════════════════════

def compute_kpi_summary(
    village: str,
    frames_data: List[Dict[str, Any]],
    view_mode: str,
    fps: float,
    gpx_file: Optional[str]
) -> Tuple[Dict[str, Any], str]:
    """Generates both {village}_kpi.json dict and {village}_kpi.txt summary string."""
    total_frames = len(frames_data)
    if total_frames == 0:
        empty_summary = {"village": village, "total_frames": 0, "overall_score": 100}
        return empty_summary, "No frames processed."

    frames_with_road_damage = 0
    frames_with_garbage = 0
    frames_with_drain_hazard = 0
    frames_with_issues = 0

    class_counts = {"road_damage": 0, "trash": 0, "dump": 0, "drainage": 0}
    class_frame_counts = {"road_damage": 0, "trash": 0, "dump": 0, "drainage": 0}
    drain_subtype_counts = {"open": 0, "broken": 0, "covered": 0}

    road_type_frame_counts = {"tar": 0, "cement": 0, "concrete": 0, "gravel": 0, "dirt": 0, "unknown": 0}
    road_type_issue_counts = {"tar": 0, "cement": 0, "concrete": 0, "gravel": 0, "dirt": 0, "unknown": 0}
    road_type_damage_counts = {"tar": 0, "cement": 0, "concrete": 0, "gravel": 0, "dirt": 0, "unknown": 0}
    road_type_garbage_counts = {"tar": 0, "cement": 0, "concrete": 0, "gravel": 0, "dirt": 0, "unknown": 0}
    road_type_drain_counts = {"tar": 0, "cement": 0, "concrete": 0, "gravel": 0, "dirt": 0, "unknown": 0}

    # Distance metrics
    total_distance_m = 0.0
    road_type_distance_m = {"tar": 0.0, "cement": 0.0, "concrete": 0.0, "gravel": 0.0, "dirt": 0.0, "unknown": 0.0}

    unique_tracks = {"road_damage": set(), "trash": set(), "dump": set(), "drainage": set()}
    for f in frames_data:
        rtype = f["road_type"]["road_type"].lower()
        if rtype not in road_type_frame_counts:
            rtype = "unknown"
        road_type_frame_counts[rtype] += 1

        seg_dist = f.get("segment_distance_m", 0.0) or 0.0
        total_distance_m += seg_dist
        road_type_distance_m[rtype] += seg_dist

        has_rd = False
        has_gb = False
        has_dh = False

        for v in f["views"]:
            for d in v["detections"]:
                if d.get("cross_view_dup"):
                    continue

                lbl = d["label"]
                class_counts[lbl] = class_counts.get(lbl, 0) + 1
                tr = d.get("track_id")
                if tr and lbl in unique_tracks:
                    unique_tracks[lbl].add((v["view"], tr))

                if lbl == "road_damage":
                    has_rd = True
                elif lbl in {"trash", "dump", "garbage"}:
                    has_gb = True
                elif lbl == "drainage":
                    st = d.get("subtype")
                    if st in drain_subtype_counts:
                        drain_subtype_counts[st] += 1
                    if st in {"open", "broken"}:
                        has_dh = True

        if has_rd:
            frames_with_road_damage += 1
            class_frame_counts["road_damage"] += 1
            road_type_damage_counts[rtype] += 1
        if has_gb:
            frames_with_garbage += 1
            if "trash" in class_frame_counts:
                class_frame_counts["trash"] += 1
            if "dump" in class_frame_counts:
                class_frame_counts["dump"] += 1
            road_type_garbage_counts[rtype] += 1
        if has_dh:
            frames_with_drain_hazard += 1
            class_frame_counts["drainage"] += 1
            road_type_drain_counts[rtype] += 1

        if has_rd or has_gb or has_dh:
            frames_with_issues += 1
            road_type_issue_counts[rtype] += 1

    track_counts = {k: len(v) for k, v in unique_tracks.items()}
    issue_rate_pct = round((frames_with_issues / total_frames) * 100.0, 1)

    # Village Scores (0-100 integers) (§7.2 Spec)
    road_score = round((1.0 - (frames_with_road_damage / total_frames)) * 100.0)
    sanitation_score = round((1.0 - (frames_with_garbage / total_frames)) * 100.0)
    drainage_score = round((1.0 - (frames_with_drain_hazard / total_frames)) * 100.0)
    overall_score = round(road_score * 0.40 + sanitation_score * 0.35 + drainage_score * 0.25)

    # Per-road type scores
    per_road_type_scores = {}
    for rtype, r_count in road_type_frame_counts.items():
        if r_count > 0:
            r_rd = road_type_damage_counts[rtype]
            r_gb = road_type_garbage_counts[rtype]
            r_dh = road_type_drain_counts[rtype]
            r_score = round((1.0 - (r_rd / r_count)) * 100.0)
            s_score = round((1.0 - (r_gb / r_count)) * 100.0)
            d_score = round((1.0 - (r_dh / r_count)) * 100.0)
            o_score = round(r_score * 0.40 + s_score * 0.35 + d_score * 0.25)
            per_road_type_scores[rtype] = {
                "total_frames": r_count,
                "frames_with_issues": road_type_issue_counts[rtype],
                "road_score": r_score,
                "sanitation_score": s_score,
                "drainage_score": d_score,
                "overall_score": o_score,
                "distance_m": round(road_type_distance_m[rtype], 1),
            }

    # Road type share percentages
    road_type_share_pct = {
        rtype: round((count / total_frames) * 100.0, 1)
        for rtype, count in road_type_frame_counts.items()
    }

    kpi_dict = {
        "metadata": {
            "village": village,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "classifier_model": "mobilenet_best.pt",
            "yolo_models": ["yolo_best.pt", "weights/best.pt"],
            "view_mode": view_mode,
            "fps": fps,
            "gpx_file": gpx_file,
        },
        "overall_score": overall_score,
        "road_score": road_score,
        "sanitation_score": sanitation_score,
        "drainage_score": drainage_score,
        "total_frames": total_frames,
        "frames_with_issues": frames_with_issues,
        "issue_rate_pct": issue_rate_pct,
        "track_counts": track_counts,
        "class_counts": class_counts,
        "class_frame_counts": class_frame_counts,
        "drain_subtype_counts": drain_subtype_counts,
        "road_type_frame_counts": road_type_frame_counts,
        "road_type_share_pct": road_type_share_pct,
        "view_quality_counts": {"good": total_frames * len(VIEW_SETS[view_mode])},
        "per_road_type_scores": per_road_type_scores,
    }

    if gpx_file:
        kpi_dict["total_distance_m"] = round(total_distance_m, 1)
        kpi_dict["road_type_distance_m"] = {k: round(v, 1) for k, v in road_type_distance_m.items()}

    # Render pretty UTF-8 text summary (§7.2 Spec)
    txt_lines = [
        "============================================================",
        f"       VILLAGE STREET SURVEY KPI REPORT — {village.upper()}",
        "============================================================",
        f" Generated At : {kpi_dict['metadata']['generated_at']}",
        f" View Mode    : {view_mode} | FPS: {fps}",
        f" GPX Track    : {gpx_file or 'None'}",
        "------------------------------------------------------------",
        " 📊 OVERALL VILLAGE SCORES (0-100 scale):",
        f"    • Overall Infrastructure Score : {overall_score} / 100",
        f"    • Road Condition Score        : {road_score} / 100",
        f"    • Sanitation Score            : {sanitation_score} / 100",
        f"    • Drainage Hazard Score       : {drainage_score} / 100",
        "------------------------------------------------------------",
        " 📍 UNIQUE PHYSICAL HAZARDS / TRACKS:",
        f"    • Road Damage Sites           : {track_counts.get('road_damage', 0)} distinct locations",
        f"    • Trash Sites                 : {track_counts.get('trash', 0)} distinct trash spots",
        f"    • Dump Sites                  : {track_counts.get('dump', 0)} distinct dump sites",
        f"    • Drainage Channels           : {track_counts.get('drainage', 0)} open sections",
        "------------------------------------------------------------",
        " 🚘 FRAME STATISTICS:",
        f"    • Total Surveyed Frames       : {total_frames}",
        f"    • Frames with Hazards/Issues  : {frames_with_issues} ({issue_rate_pct}%)",
        f"    • Total Road Damage Detections: {class_counts['road_damage']}",
        f"    • Total Trash Detections      : {class_counts['trash']}",
        f"    • Total Dump Detections       : {class_counts['dump']}",
        f"    • Total Drainage Detections   : {class_counts['drainage']} "
        f"(Open: {drain_subtype_counts['open']}, Broken: {drain_subtype_counts['broken']}, Covered: {drain_subtype_counts['covered']})",
        "------------------------------------------------------------",
        " 🛣️ ROAD SURFACE BREAKDOWN:",
    ]
    for rtype, share in road_type_share_pct.items():
        if road_type_frame_counts[rtype] > 0:
            txt_lines.append(f"    • {rtype.capitalize():<8}: {road_type_frame_counts[rtype]} frames ({share}%)")

    if gpx_file:
        txt_lines.extend([
            "------------------------------------------------------------",
            f" 📍 DISTANCE COVERAGE: {round(total_distance_m, 1)} meters total",
        ])
        for rtype, dist in road_type_distance_m.items():
            if dist > 0:
                txt_lines.append(f"    • {rtype.capitalize():<8}: {round(dist, 1)} m")

    txt_lines.append("============================================================\n")
    kpi_txt = "\n".join(txt_lines)

    return kpi_dict, kpi_txt


# ════════════════════════════════════════════════════════════
# 5. MAIN PIPELINE EXECUTOR & CLI
# ════════════════════════════════════════════════════════════

def extract_frame_index(filename: str, default_base: int = 1) -> int:
    """Extracts last group of digits from filename stem (Spec §1)."""
    stem = Path(filename).stem
    matches = re.findall(r"\d+", stem)
    if matches:
        return int(matches[-1])
    return default_base


def setup_logger(output_dir: Path) -> logging.Logger:
    """Sets up console logger and rotating file logger at output/app.log (5 MB x 3) (§8 Spec)."""
    logger = logging.getLogger("StreetSurveyPipeline")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_format = logging.Formatter("[%(levelname)s] %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    log_file = output_dir / "app.log"
    f_handler = RotatingFileHandler(str(log_file), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    f_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(description="Village Street-Survey 360° Detection Pipeline")
    parser.add_argument("--input", required=True, help="Directory of equirectangular 360° JPEG frames")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--gpx", default=None, help="Optional Insta360 GPX track file")
    parser.add_argument("--fps", type=float, default=2.0, help="FPS at which frames were extracted (default 2.0)")
    parser.add_argument("--views", choices=["quad", "tri", "forward"], default="quad",
                        help="View set (default quad -- the 4 views the models were trained on)")
    parser.add_argument("--village", default="village", help="Village name for output filenames")
    parser.add_argument("--frame-index-base", type=int, default=1, help="1-based frame index start (default 1)")
    parser.add_argument("--jitter-m", type=float, default=0.5, help="Standstill GPS jitter distance threshold in meters (default 0.5)")
    parser.add_argument("--no-resume", action="store_true", help="Delete checkpoint and re-process all frames")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames to process (e.g. 20)")

    # CLI Confidence threshold defaults (§4 Spec)
    parser.add_argument("--conf-road-damage", type=float, default=0.15, help="Confidence threshold for road_damage (default 0.15)")
    parser.add_argument("--conf-garbage", type=float, default=0.25, help="Confidence threshold for garbage (default 0.25)")
    parser.add_argument("--conf-drainage", type=float, default=0.15, help="Confidence threshold for drainage (default 0.15)")

    # Model Weights paths
    parser.add_argument("--classifier", default="classifier.pth" if Path("classifier.pth").exists() else ("convnext_road_model_v2.pth" if Path("convnext_road_model_v2.pth").exists() else "classifier.pt"), help="Road classifier weights (.pth for ConvNeXt or .pt for MobileNet)")
    parser.add_argument("--no-noise-fallback", action="store_true",
                        help="Report 'noise' as-is instead of substituting the "
                             "runner-up class. Matches a plain argmax reference "
                             "implementation -- use when comparing classifiers.")
    parser.add_argument("--road-res", type=int, default=None,
                        help="Classifier input resolution. Must match training "
                             "(best_convnext_384.pth -> 384). Inferred from the "
                             "filename when omitted.")
    parser.add_argument("--gate", action="store_true",
                        help="Temporal gating: keep a weak detection only if a confident "
                             "one of the same class appears nearby in an adjacent frame. "
                             "Reduces frame-to-frame flicker. Needs a LOW --conf-* floor "
                             "so there are weak detections to rescue.")
    parser.add_argument("--gate-accept", type=float, default=0.25,
                        help="Confidence at or above which a detection is always kept")
    parser.add_argument("--gate-weak", type=float, default=0.08,
                        help="Floor below which a detection is discarded outright")
    parser.add_argument("--gate-window", type=int, default=2,
                        help="How many frames either side may vouch for a weak detection")
    parser.add_argument("--gate-angle", type=float, default=25.0,
                        help="Max angular separation (deg) for a voucher to count")
    parser.add_argument("--tta", action="store_true",
                        help="Test-time augmentation: multi-scale + flipped inference, "
                             "merged. ~2-3x slower, typically a few points of recall.")
    parser.add_argument("--yolo-det", default="detector.pt", help="YOLO detection weights (trash/dump)")
    parser.add_argument("--yolo-seg", default="segmentor.pt" if Path("segmentor.pt").exists() else "segmenter.pt", help="YOLO segmentation weights (damage/drainage)")

    # Output options
    parser.add_argument("--save-views", action="store_true", help="Save individual 2D annotated perspective view crop images in output_dir/views_annotated")
    parser.add_argument("--fill-gaps", action="store_true",
                        help="Temporal tracking & visual gap-filling: recover momentarily dropped detections across adjacent frames")
    parser.add_argument("--gap-size", type=int, default=2,
                        help="Max consecutive missing frames to bridge (default 2)")
    parser.add_argument("--min-visual-sim", type=float, default=0.40,
                        help="Minimum visual HSV correlation threshold to confirm same object (default 0.40)")
    parser.add_argument("--extrapolate", action="store_true", default=False,
                        help="Enable unanchored boundary extrapolation (disabled by default to prevent false positives on good road)")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    views_annotated_dir = None
    if args.save_views:
        views_annotated_dir = output_dir / "views_annotated"
        views_annotated_dir.mkdir(parents=True, exist_ok=True)
        # One folder per view, so each direction can be reviewed as a set
        for _v in VIEW_SETS[args.views]:
            (views_annotated_dir / _v).mkdir(parents=True, exist_ok=True)

    logger = setup_logger(output_dir)
    logger.info(f"Starting Village Street-Survey Pipeline for village: {args.village}")
    logger.info(f"Input Directory: {input_dir.resolve()}")
    logger.info(f"Output Directory: {output_dir.resolve()}")
    logger.info(f"View Set: {args.views} | FPS: {args.fps}")

    # Fallback checks for model paths
    script_dir = Path(__file__).resolve().parent
    classifier_path = args.classifier if os.path.exists(args.classifier) else (str(script_dir / args.classifier) if (script_dir / args.classifier).exists() else args.classifier)
    yolo_det_path = args.yolo_det if os.path.exists(args.yolo_det) else (str(script_dir / "detector.pt") if (script_dir / "detector.pt").exists() else args.yolo_det)
    yolo_seg_path = args.yolo_seg if os.path.exists(args.yolo_seg) else (str(script_dir / "segmentor.pt") if (script_dir / "segmentor.pt").exists() else args.yolo_seg)

    # Load Models ONCE at startup (§10 Spec)
    classifier_str = str(classifier_path).lower()
    if "mobilenet" in classifier_str:
        logger.info(f"Using MobileNet Road Classifier: {classifier_path}")
        classifier = MobileNetRoadClassifier(classifier_path)
    elif "convnext" in classifier_str or "best_road" in classifier_str:
        logger.info(f"Using ConvNeXt-Tiny Road Classifier: {classifier_path}")
        classifier = ConvNeXtRoadClassifier(classifier_path, input_res=args.road_res,
                                     noise_fallback=not args.no_noise_fallback)
    elif str(classifier_path).endswith(".pth"):
        try:
            ckpt = torch.load(classifier_path, map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            if any("features.0.0.weight" in k and sd[k].shape == torch.Size([16, 3, 3, 3]) for k in sd.keys()):
                logger.info(f"Auto-detected MobileNet architecture in {classifier_path}")
                classifier = MobileNetRoadClassifier(classifier_path)
            else:
                logger.info(f"Using ConvNeXt-Tiny Road Classifier: {classifier_path}")
                classifier = ConvNeXtRoadClassifier(classifier_path, input_res=args.road_res,
                                     noise_fallback=not args.no_noise_fallback)
        except Exception:
            logger.info(f"Using ConvNeXt-Tiny Road Classifier: {classifier_path}")
            classifier = ConvNeXtRoadClassifier(classifier_path, input_res=args.road_res,
                                     noise_fallback=not args.no_noise_fallback)
    else:
        logger.info(f"Using MobileNet Road Classifier: {classifier_path}")
        classifier = MobileNetRoadClassifier(classifier_path)
    detector_det = YoloInferenceEngine(yolo_det_path, augment=args.tta)
    detector_seg = YoloInferenceEngine(yolo_seg_path, augment=args.tta)
    if args.tta:
        logger.info("Test-time augmentation ENABLED (~2-3x slower inference)")

    conf_thresholds = {
        "road_damage": args.conf_road_damage,
        "garbage": args.conf_garbage,
        "drainage": args.conf_drainage,
    }

    # Parse GPX if supplied or auto-discover in input directory
    gpx_file_path = args.gpx
    if gpx_file_path:
        gp = Path(gpx_file_path)
        if gp.is_dir():
            gpx_candidates = list(gp.glob("*.gpx"))
            if gpx_candidates:
                gpx_file_path = str(gpx_candidates[0])
                logger.info(f"Resolved GPX track from directory: {gpx_file_path}")
    else:
        gpx_candidates = list(input_dir.glob("*.gpx"))
        if not gpx_candidates and (input_dir.name == "frames" or (input_dir / "frames").is_dir()):
            gpx_candidates = list(input_dir.parent.glob("*.gpx")) if input_dir.name == "frames" else list((input_dir / "frames").glob("*.gpx"))
        if gpx_candidates:
            gpx_file_path = str(gpx_candidates[0])
            logger.info(f"Auto-discovered GPX file: {gpx_file_path}")

    gpx_track = []
    if gpx_file_path:
        if os.path.exists(gpx_file_path) and os.path.isfile(gpx_file_path):
            logger.info(f"Parsing GPX track file: {gpx_file_path}")
            gpx_track = parse_gpx(gpx_file_path)
            logger.info(f"Loaded {len(gpx_track)} GPX track points.")
        else:
            logger.warning(f"GPX file specified but not found: {gpx_file_path}")

    # Checkpoint resumability (§9 Spec)
    checkpoint_file = output_dir / ".processed.jsonl"
    processed_records = {}

    if args.no_resume and checkpoint_file.exists():
        logger.info("--no-resume specified. Clearing previous checkpoint.")
        checkpoint_file.unlink()

    if checkpoint_file.exists():
        logger.info(f"Reading checkpoint file: {checkpoint_file.name}")
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    processed_records[rec["frame"]] = rec
        logger.info(f"Resuming pipeline: {len(processed_records)} frames already processed.")

    # Find 360° frames
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    frame_paths = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts], key=lambda x: x.name)

    if not frame_paths:
        logger.error(f"No valid image files found in input directory: {input_dir}")
        return

    logger.info(f"Found {len(frame_paths)} input 360° frame(s) to process.")
    if args.max_frames and args.max_frames > 0:
        frame_paths = frame_paths[:args.max_frames]
        logger.info(f"Limited processing to first {len(frame_paths)} frame(s) via --max-frames.")

    projector = EquirectangularProjector(view_width=TRAIN_VIEW_SIZE,
                                         view_height=TRAIN_VIEW_SIZE)
    selected_view_names = VIEW_SETS[args.views]

    final_frame_records = []
    pending_renders = []  # populated only when --gate is active
    last_confident_road = None

    for idx, frame_path in enumerate(frame_paths):
        frame_name = frame_path.name

        # Skip if already processed in checkpoint
        if frame_name in processed_records:
            logger.info(f"Skipping frame {idx + 1}/{len(frame_paths)}: {frame_name} (loaded from checkpoint)")
            final_frame_records.append(processed_records[frame_name])
            continue

        # Grab raw binary EXIF from source frame before processing (essential for Mapillary Uploader)
        src_exif = read_exif_bytes(frame_path)

        eq_img = cv2.imread(str(frame_path))
        if eq_img is None:
            logger.warning(f"Failed to read image frame: {frame_name}")
            continue

        frame_idx = extract_frame_index(frame_name, default_base=args.frame_index_base)

        # Step 1 — Extract perspective views from equirect frame (§2 Spec)
        view_images = []
        for vname in selected_view_names:
            vcfg = VIEW_CONFIGS[vname]
            vimg = projector.extract_view(eq_img, vcfg["yaw"], vcfg["pitch"], vcfg["fov"])
            view_images.append(vimg)

        # Step 2 — Road-type classification on each view (§3 Spec)
        # Canonical road type is driven by the forward view's prediction
        road_pred = classifier.predict(view_images[0])
        forward_road_type, forward_conf, _ = road_pred
        # fallback_used / primary_class were being discarded, so a substituted
        # verdict looked identical to a confident one in the logs and the JSON
        road_fallback = bool(getattr(road_pred, "fallback_used", False))
        road_primary = getattr(road_pred, "primary_class", forward_road_type)

        # Noise fallback stabilization: prefer sustained confident road type over near-zero noise fallback
        if road_fallback and forward_conf < 0.25 and last_confident_road is not None:
            forward_road_type = last_confident_road
        elif not road_fallback and forward_conf >= 0.50:
            last_confident_road = forward_road_type

        # Step 3 — YOLO detection (Batched inference across views) (§4 Spec)
        cand_floor = 0.10 if args.fill_gaps else None
        batch_dets_det = detector_det.predict_batch(view_images, conf_thresholds=conf_thresholds, candidate_floor=cand_floor)
        batch_dets_seg = detector_seg.predict_batch(view_images, conf_thresholds=conf_thresholds, candidate_floor=cand_floor)

        views_json_list = []
        all_flattened_dets_for_annotation = []
        view_det_counts = []

        for v_i, vname in enumerate(selected_view_names):
            # Prioritize segmentation detections (which have polygon masks) over bounding box detections
            seg_dets = batch_dets_seg[v_i]
            det_dets = batch_dets_det[v_i]

            # Combine with polygon mask preference & deduplication
            view_dets = list(seg_dets)
            for d in det_dets:
                # Skip classes the segmentation model owns -- drawing the
                # detector's box for these is what produced boxes instead of
                # masks on road_damage.
                if (d["label"], d.get("subtype")) in SEG_OWNED:
                    continue
                d_box = d["box_1000"]
                is_duplicate = False
                for s in list(seg_dets):
                    # subtype matters: CLASS_MAP maps open_drainage and
                    # closed_drainage to the same label "drainage", so
                    # matching on label alone lets an open mask suppress a
                    # closed box.
                    if s["label"] == d["label"] and s.get("subtype") == d.get("subtype"):
                        s_box = s["box_1000"]
                        # Calculate IoU between box_1000 rectangles [ymin, xmin, ymax, xmax]
                        inter_ymin = max(d_box[0], s_box[0])
                        inter_xmin = max(d_box[1], s_box[1])
                        inter_ymax = min(d_box[2], s_box[2])
                        inter_xmax = min(d_box[3], s_box[3])
                        if inter_ymax > inter_ymin and inter_xmax > inter_xmin:
                            inter_area = (inter_ymax - inter_ymin) * (inter_xmax - inter_xmin)
                            d_area = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
                            s_area = (s_box[2] - s_box[0]) * (s_box[3] - s_box[1])
                            union_area = d_area + s_area - inter_area
                            if union_area > 0 and (inter_area / union_area) > 0.3:
                                if s.get("is_candidate") and not d.get("is_candidate"):
                                    if s in view_dets:
                                        view_dets.remove(s)
                                    is_duplicate = False
                                    break
                                else:
                                    is_duplicate = True
                                    break
                if not is_duplicate:
                    view_dets.append(d)

            # Apply intra-view NMS to suppress duplicate overlapping boxes from the same class
            view_dets = apply_intra_view_nms(view_dets, iou_thresh=0.65)
            view_det_counts.append(len(view_dets))

            spec_dets_json = []
            for d in view_dets:
                spec_det = {
                    "label": d["label"],
                    "box_1000": d["box_1000"],
                    "confidence": d["confidence"],
                }
                if d.get("subtype"):
                    spec_det["subtype"] = d["subtype"]
                if d.get("is_candidate"):
                    spec_det["is_candidate"] = True
                if d.get("_polygon_norm"):
                    spec_det["_polygon_norm"] = d["_polygon_norm"]
                spec_dets_json.append(spec_det)

                # Do NOT render unconfirmed weak candidates on initial equirect or view crops
                if not d.get("is_candidate"):
                    all_flattened_dets_for_annotation.append((vname, d))

            # Save 2D annotated perspective view crop if --save-views is enabled.
            # Under --gate the detections have not been judged yet, so the crops
            # are rendered in pass 2 instead -- otherwise the crops and the
            # stitched equirect would disagree about what was detected.
            if views_annotated_dir and not args.gate:
                visible_dets = [d for d in view_dets if not d.get("is_candidate")]
                view_crop_annotated = annotate_perspective_view(
                    bgr_img=view_images[v_i],
                    detections=visible_dets,
                    road_type=forward_road_type if vname == "front" else None
                )
                crop_dir = views_annotated_dir / vname
                crop_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(crop_dir / f"{frame_path.stem}.jpg"), view_crop_annotated)

            views_json_list.append({
                "view": vname,
                "view_quality": "good",
                "detections": spec_dets_json,
            })

        # Step 4 — GPX distance attribution (§6 Spec)
        iso_ts, lat, lon = attribute_gpx_data(
            frame_index=frame_idx,
            index_base=args.frame_index_base,
            fps=args.fps,
            gpx_track=gpx_track,
            jitter_m=args.jitter_m
        )

        annotated_rel_path = f"annotated/annotated_{frame_name}"
        annotated_full_path = output_dir / annotated_rel_path

        # Construct frame JSON record (§7.1 Spec contract)
        frame_record = {
            "frame": frame_name,
            "annotated_path": annotated_rel_path.replace("\\", "/"),
            "road_type": {"road_type": forward_road_type, "confidence": forward_conf,
                          "fallback_used": road_fallback, "primary_class": road_primary},
            "views": views_json_list,
        }

        if gpx_track:
            frame_record["timestamp"] = iso_ts
            frame_record["lat"] = lat
            frame_record["lon"] = lon

        if args.gate:
            # Temporal gating needs to see neighbouring frames before it can
            # judge this one, so rendering is deferred to a second pass. The
            # equirect is re-read from disk then rather than held in memory.
            pending_renders.append({
                "frame_path": frame_path,
                "frame_name": frame_name,
                "road_type": forward_road_type,
                "dets": all_flattened_dets_for_annotation,
                "out_path": annotated_full_path,
                "exif": src_exif,
                "lat": lat, "lon": lon, "iso_ts": iso_ts,
                "record": frame_record,
            })
        else:
            # Step 5 — Save annotated equirect image with EXIF GPS metadata (§5 Spec)
            annotated_img = render_equirect_annotations(
                eq_img=eq_img,
                frame_road_type=forward_road_type,
                all_view_detections=all_flattened_dets_for_annotation,
                projector=projector
            )
            save_annotated_frame_with_exif(
                bgr_img=annotated_img,
                out_path=annotated_full_path,
                exif_bytes=src_exif,
                lat=lat,
                lon=lon,
                iso_ts=iso_ts,
                quality=92
            )

            final_frame_records.append(frame_record)

            # Append to checkpoint file (.processed.jsonl)
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(frame_record) + "\n")

        det_summary_str = ", ".join([f"{vname}:{cnt}" for vname, cnt in zip(selected_view_names, view_det_counts)])
        road_note = f" [noise->{forward_road_type}]" if road_fallback else ""
        logger.info(f"Processed frame {idx + 1}/{len(frame_paths)}: {frame_name} | Road: {forward_road_type} ({forward_conf:.2f}){road_note} | Detections: {det_summary_str}")

    # ── PASS 2: temporal gating, then render ─────────────────
    if args.gate and pending_renders:
        logger.info(f"Temporal gating across {len(pending_renders)} frames "
                    f"(accept {args.gate_accept}, weak floor {args.gate_weak}, "
                    f"window +/-{args.gate_window}, angle {args.gate_angle} deg)")

        # Describe every detection by its bearing so frames can be compared
        seq = []
        for pr in pending_renders:
            frame_dets = []
            for vname, det in pr["dets"]:
                yaw, pitch = detection_bearing(det, vname, projector)
                frame_dets.append({
                    "label": det.get("label"),
                    "subtype": det.get("subtype"),
                    "confidence": det.get("confidence", 0.0),
                    "yaw": yaw, "pitch": pitch,
                })
            seq.append(frame_dets)

        if gate_sequence is None:
            raise RuntimeError("temporal_gating.py is required when using the --gate flag.")
        keep_masks, stats = gate_sequence(
            seq,
            accept_conf=args.gate_accept,
            weak_conf=args.gate_weak,
            window=args.gate_window,
            max_angle_deg=args.gate_angle,
        )
        logger.info(f"Gating: {gating_summary(stats)}")

        for pr, mask in zip(pending_renders, keep_masks):
            kept = [pair for pair, k in zip(pr["dets"], mask) if k]

            eq_img = cv2.imread(str(pr["frame_path"]))
            if eq_img is None:
                logger.warning(f"Re-read failed during render: {pr['frame_name']}")
                continue

            annotated_img = render_equirect_annotations(
                eq_img=eq_img,
                frame_road_type=pr["road_type"],
                all_view_detections=kept,
                projector=projector
            )
            save_annotated_frame_with_exif(
                bgr_img=annotated_img,
                out_path=pr["out_path"],
                exif_bytes=pr["exif"],
                lat=pr["lat"], lon=pr["lon"], iso_ts=pr["iso_ts"],
                quality=92
            )

            # Annotated perspective crops, rendered here so they carry the same
            # gated detections as the equirect. The views are re-projected from
            # the frame just re-read rather than held in memory across pass 1.
            if views_annotated_dir:
                kept_by_view = {}
                for vname, det in kept:
                    kept_by_view.setdefault(vname, []).append(det)
                for vname in selected_view_names:
                    vcfg = VIEW_CONFIGS[vname]
                    crop = projector.extract_view(eq_img, vcfg["yaw"],
                                                  vcfg["pitch"], vcfg["fov"])
                    crop_annotated = annotate_perspective_view(
                        bgr_img=crop,
                        detections=kept_by_view.get(vname, []),
                        road_type=pr["road_type"] if vname == "front" else None
                    )
                    crop_dir = views_annotated_dir / vname
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(crop_dir / f"{Path(pr['frame_name']).stem}.jpg"),
                        crop_annotated)

            # Rebuild the JSON views from the surviving detections so the
            # exported contract and the KPIs match what was drawn.
            by_view = {v: [] for v in selected_view_names}
            for vname, det in kept:
                entry = {"label": det["label"],
                         "box_1000": det["box_1000"],
                         "confidence": det["confidence"]}
                if det.get("subtype"):
                    entry["subtype"] = det["subtype"]
                by_view.setdefault(vname, []).append(entry)
            pr["record"]["views"] = [
                {"view": v, "view_quality": "good", "detections": by_view.get(v, [])}
                for v in selected_view_names
            ]

            final_frame_records.append(pr["record"])
            with open(checkpoint_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(pr["record"]) + "\n")

        logger.info(f"Rendered {len(final_frame_records)} gated frames.")

    # Compute distance to next frame if GPX provided
    if gpx_track and len(final_frame_records) > 0:
        for i in range(len(final_frame_records)):
            if i < len(final_frame_records) - 1:
                cur_f = final_frame_records[i]
                next_f = final_frame_records[i + 1]
                if cur_f.get("lat") is not None and next_f.get("lat") is not None:
                    dist = haversine_distance(cur_f["lat"], cur_f["lon"], next_f["lat"], next_f["lon"])
                    if dist < args.jitter_m:
                        dist = 0.0
                    cur_f["segment_distance_m"] = round(dist, 3)
                else:
                    cur_f["segment_distance_m"] = 0.0
            else:
                final_frame_records[i]["segment_distance_m"] = 0.0

    # ── Gap-Filling (§ Temporal Multi-Object Smoothing & Interpolation) ──
    if args.fill_gaps and len(final_frame_records) > 1:
        logger.info(f"Applying Temporal Tracking & Visual Gap-Filling (gap_size={args.gap_size}, min_visual_sim={args.min_visual_sim})...")
        total_filled_across_views = 0
        for vname in selected_view_names:
            v_seq = []
            for f in final_frame_records:
                v_match = next((v for v in f.get("views", []) if v.get("view") == vname), None)
                v_seq.append(v_match.get("detections", []) if v_match else [])

            # Load perspective view crop images for visual appearance verification
            v_imgs = None
            if views_annotated_dir and (views_annotated_dir / vname).exists():
                v_imgs = []
                for f in final_frame_records:
                    stem = Path(f["frame"]).stem
                    img_p = views_annotated_dir / vname / f"{stem}.jpg"
                    v_imgs.append(cv2.imread(str(img_p)) if img_p.exists() else None)

            interpolator = TemporalTrackInterpolator(
                max_gap=args.gap_size,
                min_visual_sim=args.min_visual_sim,
                extrapolate=args.extrapolate
            )
            filled_seq, stats = interpolator.process_sequence(v_seq, frame_images=v_imgs)
            n_filled = stats.get("total_interpolated_detections", 0)
            total_filled_across_views += n_filled
            logger.info(f"  * View [{vname:<6}]: {n_filled} dropped detections recovered (tracks: {stats.get('total_tracklets', 0)}, by label: {stats.get('by_label', {})})")

            # Update final_frame_records with filled detections
            for f_idx, f in enumerate(final_frame_records):
                for v in f.get("views", []):
                    if v.get("view") == vname:
                        v["detections"] = filled_seq[f_idx]

            # Re-render ONLY missed/filled boxes on perspective views if requested
            if views_annotated_dir and (views_annotated_dir / vname).exists():
                for f_idx, f in enumerate(final_frame_records):
                    stem = Path(f["frame"]).stem
                    img_p = views_annotated_dir / vname / f"{stem}.jpg"
                    if img_p.exists():
                        annotate_interpolated_on_image(str(img_p), str(img_p), filled_seq[f_idx])

        logger.info(f"Gap-Filling Complete: Total {total_filled_across_views} missed detections recovered!")

    # ── Temporal Road Surface Smoothing Pass ──
    smooth_road_surface_sequence(final_frame_records, window_size=5)

    # ── Cross-View Boundary Deduplication for JSON & KPI ──
    for f in final_frame_records:
        deduplicate_cross_view_detections(f, projector)

    # Step 7 — Save final JSON & KPI outputs (§7 Spec)
    # Sort frames in strict filename order
    final_frame_records.sort(key=lambda x: x["frame"])

    detections_json_file = output_dir / f"{args.village}_detections.json"
    clean_frames = []
    for f in final_frame_records:
        cf = {k: v for k, v in f.items() if not k.startswith("_")}
        cf_views = []
        for v in f.get("views", []):
            cv = {k: val for k, val in v.items() if not k.startswith("_")}
            clean_dets = []
            for d in v.get("detections", []):
                clean_dets.append({k: val for k, val in d.items() if not k.startswith("_")})
            cv["detections"] = clean_dets
            cf_views.append(cv)
        cf["views"] = cf_views
        clean_frames.append(cf)
    detections_payload = {"frames": clean_frames}
    detections_json_file.write_text(json.dumps(detections_payload, indent=2), encoding="utf-8")
    logger.info(f"Saved Detections JSON: {detections_json_file.resolve()}")

    kpi_dict, kpi_txt = compute_kpi_summary(
        village=args.village,
        frames_data=final_frame_records,
        view_mode=args.views,
        fps=args.fps,
        gpx_file=Path(gpx_file_path).name if (gpx_file_path and os.path.exists(gpx_file_path) and os.path.isfile(gpx_file_path)) else None
    )

    kpi_json_file = output_dir / f"{args.village}_kpi.json"
    kpi_txt_file = output_dir / f"{args.village}_kpi.txt"

    kpi_json_file.write_text(json.dumps(kpi_dict, indent=2), encoding="utf-8")
    kpi_txt_file.write_text(kpi_txt, encoding="utf-8")

    logger.info(f"Saved KPI JSON: {kpi_json_file.resolve()}")
    logger.info(f"Saved KPI Summary Text: {kpi_txt_file.resolve()}")

    print("\n" + kpi_txt)
    logger.info("[SUCCESS] Pipeline Execution Successfully Finished!")


if __name__ == "__main__":
    main()