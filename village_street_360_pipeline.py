#!/usr/bin/env python3
"""
Indian Village Street 360° 4-View Hazard Detection, Segmentation & GPX Geotagging Pipeline
==========================================================================================
Compatible with the 4 project weight files:
  1. classifier.pth       : Stage-1 Road Surface Type Classifier (PyTorch model / state_dict / YOLO-cls)
  2. detector.pt          : Stage-2 Ultralytics YOLO26s Detector (v8.4.71+)
                            Classes -> {0: 'trash', 1: 'dump', 2: 'road_damage'}
  3. segmentation.pt      : Stage-3 Ultralytics YOLO26s-seg Segmenter (v8.4.108+)
                            Classes -> {0: 'open_drainage', 1: 'road_damage'}
  4. YOLOv8_Small_RDD.pt  : Stage-4 Ultralytics YOLOv8s Road Defect Sub-Classifier (Optional)
                            Classes -> {0: 'Longitudinal Crack', 1: 'Transverse Crack',
                                        2: 'Alligator Crack', 3: 'Potholes'}

Supports:
  - 360° equirectangular frames folder + GPX track file (--source <folder> --gpx <file.gpx>)
  - Automatic GPX trackpoint matching (by timestamp or proportional frame sequence index)
  - Exports annotated 2x2 perspective views, JSON, CSV (with lat/lon & Google Maps links), and GeoJSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fractions import Fraction
from PIL import Image, ExifTags

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("VillageStreetPipeline")


def decimal_to_dms_rational(deg_float: float):
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
    ele: Optional[float] = None,
    iso_ts: Optional[str] = None,
    quality: int = 95,
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

    # Fallback to manual GPS IFD encoding if raw EXIF binary blob is not available
    exif = pil_img.getexif()
    if lat is not None and lon is not None:
        try:
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
            lat_ref = "N" if lat >= 0 else "S"
            gps_ifd[ExifTags.GPS.GPSLatitudeRef] = lat_ref
            gps_ifd[ExifTags.GPS.GPSLatitude] = decimal_to_dms_rational(lat)

            lon_ref = "E" if lon >= 0 else "W"
            gps_ifd[ExifTags.GPS.GPSLongitudeRef] = lon_ref
            gps_ifd[ExifTags.GPS.GPSLongitude] = decimal_to_dms_rational(lon)

            if ele is not None:
                gps_ifd[ExifTags.GPS.GPSAltitudeRef] = b"\x00" if ele >= 0 else b"\x01"
                gps_ifd[ExifTags.GPS.GPSAltitude] = Fraction(int(round(abs(ele) * 100)), 100)

            if iso_ts:
                try:
                    clean_ts = iso_ts.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(clean_ts)
                    gps_ifd[ExifTags.GPS.GPSDateStamp] = dt.strftime("%Y:%m:%d")
                    gps_ifd[ExifTags.GPS.GPSTimeStamp] = (
                        Fraction(dt.hour, 1),
                        Fraction(dt.minute, 1),
                        Fraction(dt.second, 1),
                    )
                except Exception:
                    pass
            pil_img.save(str(out_path), quality=quality, exif=exif)
            return
        except Exception as e:
            logger.warning("Failed to encode EXIF GPS tags: %s", e)

    pil_img.save(str(out_path), quality=quality)


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class GpxPoint:
    lat: float
    lon: float
    ele: Optional[float] = None
    time_str: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class HazardDetection:
    view_name: str
    class_id: int
    class_name: str
    subtype: Optional[str]
    confidence: float
    bbox_xyxy: List[int]  # [x1, y1, x2, y2] in view coordinates
    mask_polygon: List[List[int]] = field(default_factory=list)  # [[x, y], ...] if segmented
    area_px: float = 0.0
    source_model: str = ""


@dataclass
class FrameReport:
    frame_id: str
    frame_index: int
    latitude: Optional[float]
    longitude: Optional[float]
    elevation: Optional[float]
    gpx_time: Optional[str]
    google_maps_url: Optional[str]
    road_type: str
    road_type_confidence: float
    is_paved_surface: bool
    counts_by_class: Dict[str, int]
    detections: List[HazardDetection]
    annotated_path: Optional[str] = None


# ---------------------------------------------------------------------------
# GPX Track Parser & Frame-to-GPS Matcher
# ---------------------------------------------------------------------------

class GpxTrackMatcher:
    """
    Parses standard .gpx files (<trkpt>, <rtept>, or <wpt>) using Python's built-in XML parser
    and matches each 360° frame to its corresponding GPS coordinate.
    """

    def __init__(self, gpx_path: Optional[str]) -> None:
        self.points: List[GpxPoint] = []
        if gpx_path and Path(gpx_path).is_file():
            self._parse_gpx(gpx_path)
        elif gpx_path:
            logger.warning("GPX file not found at '%s'. Proceeding without GPS coordinates.", gpx_path)

    @staticmethod
    def _parse_iso_time(t_str: str) -> Optional[float]:
        cleaned = t_str.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(cleaned).timestamp()
        except ValueError:
            return None

    def _parse_gpx(self, gpx_path: str) -> None:
        tree = ET.parse(gpx_path)
        root = tree.getroot()

        # Match <trkpt>, <rtept>, or <wpt> regardless of XML namespace
        raw_pts = []
        for elem in root.iter():
            tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
            if tag_local in ("trkpt", "rtept", "wpt"):
                raw_pts.append(elem)

        for pt in raw_pts:
            try:
                lat = float(pt.attrib["lat"])
                lon = float(pt.attrib["lon"])
            except (KeyError, ValueError):
                continue

            ele_val: Optional[float] = None
            time_str: Optional[str] = None
            ts_val: Optional[float] = None

            for child in pt:
                c_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if c_tag == "ele" and child.text:
                    try:
                        ele_val = round(float(child.text.strip()), 2)
                    except ValueError:
                        pass
                elif c_tag == "time" and child.text:
                    time_str = child.text.strip()
                    ts_val = self._parse_iso_time(time_str)

            self.points.append(
                GpxPoint(lat=lat, lon=lon, ele=ele_val, time_str=time_str, timestamp=ts_val)
            )

        logger.info("Loaded %d GPS trackpoints from '%s'.", len(self.points), gpx_path)

    def match_frame(self, frame_idx: int, total_frames: int) -> Optional[GpxPoint]:
        """
        Maps a frame at `frame_idx` (out of `total_frames`) to the closest GPX trackpoint.
        If frame count == trackpoint count, uses 1-to-1 indexing; otherwise interpolates proportionally
        across the GPX sequence.
        """
        if not self.points:
            return None
        if len(self.points) == 1 or total_frames <= 1:
            return self.points[0]

        if total_frames == len(self.points):
            return self.points[min(frame_idx, len(self.points) - 1)]

        ratio = float(frame_idx) / float(max(1, total_frames - 1))
        pos = ratio * (len(self.points) - 1)
        idx_lo = int(np.floor(pos))
        idx_hi = min(len(self.points) - 1, idx_lo + 1)
        alpha = pos - idx_lo

        p_lo = self.points[idx_lo]
        p_hi = self.points[idx_hi]
        lat = round((1.0 - alpha) * p_lo.lat + alpha * p_hi.lat, 7)
        lon = round((1.0 - alpha) * p_lo.lon + alpha * p_hi.lon, 7)
        ele = (
            round((1.0 - alpha) * p_lo.ele + alpha * p_hi.ele, 2)
            if (p_lo.ele is not None and p_hi.ele is not None)
            else p_lo.ele
        )
        return GpxPoint(lat=lat, lon=lon, ele=ele, time_str=p_lo.time_str, timestamp=p_lo.timestamp)


# ---------------------------------------------------------------------------
# Stage 0: 360° Equirectangular to 4 Perspective Views (with Cached LUTs)
# ---------------------------------------------------------------------------

class EquirectangularSplitter:
    """
    Splits a 360° equirectangular image into 4 planar perspective views:
    front (0°), right (90°), back (180°), left (270°).
    Uses cached OpenCV spherical-to-gnomonic remap tables for fast execution.
    Also provides inverse spherical mapping and full 360° equirectangular annotation projection.
    """

    VIEW_YAWS = {
        "front": 0.0,
        "right": 90.0,
        "back": 180.0,
        "left": 270.0,
    }

    def __init__(
        self,
        out_hw: Tuple[int, int] = (1024, 1024),
        fov_deg: float = 100.0,
        pitch_deg: float = -20.0,
    ) -> None:
        self.out_h, self.out_w = out_hw
        self.fov_deg = fov_deg
        self.pitch_deg = pitch_deg
        self._lut_cache: Dict[Tuple[int, int, str], Tuple[np.ndarray, np.ndarray]] = {}

    def _build_remap_lut(
        self, eq_h: int, eq_w: int, yaw_deg: float, pitch_deg: float, fov_deg: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        fov_rad = np.deg2rad(fov_deg)
        f = (0.5 * self.out_w) / np.tan(0.5 * fov_rad)

        u = np.arange(self.out_w, dtype=np.float32) - (self.out_w - 1) / 2.0
        v = np.arange(self.out_h, dtype=np.float32) - (self.out_h - 1) / 2.0
        xx, yy = np.meshgrid(u, -v)
        zz = np.full_like(xx, f)

        norm = np.sqrt(xx * xx + yy * yy + zz * zz)
        xyz = np.stack([xx / norm, yy / norm, zz / norm], axis=-1)

        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch), -np.sin(pitch)],
                [0.0, np.sin(pitch), np.cos(pitch)],
            ],
            dtype=np.float32,
        )
        ry = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=np.float32,
        )
        rot = ry @ rx
        xyz_rot = xyz @ rot.T

        lon = np.arctan2(xyz_rot[..., 0], xyz_rot[..., 2])
        lat = np.arcsin(np.clip(xyz_rot[..., 1], -1.0, 1.0))

        map_x = ((lon / (2.0 * np.pi) + 0.5) * (eq_w - 1)).astype(np.float32)
        map_y = ((0.5 - lat / np.pi) * (eq_h - 1)).astype(np.float32)
        return map_x, map_y

    def split_frame(self, frame: np.ndarray, mode: str = "equirectangular") -> Dict[str, np.ndarray]:
        h, w = frame.shape[:2]

        if mode == "quad_grid":
            mid_h, mid_w = h // 2, w // 2
            return {
                "front": frame[:mid_h, :mid_w].copy(),
                "right": frame[:mid_h, mid_w:].copy(),
                "back": frame[mid_h:, :mid_w].copy(),
                "left": frame[mid_h:, mid_w:].copy(),
            }

        if mode == "single_views":
            return {"front": frame.copy()}

        views: Dict[str, np.ndarray] = {}
        for view_name, yaw_deg in self.VIEW_YAWS.items():
            cache_key = (h, w, view_name)
            if cache_key not in self._lut_cache:
                self._lut_cache[cache_key] = self._build_remap_lut(
                    h, w, yaw_deg=yaw_deg, pitch_deg=self.pitch_deg, fov_deg=self.fov_deg
                )
            map_x, map_y = self._lut_cache[cache_key]
            views[view_name] = cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
        return views

    def perspective_to_equirect_pts(
        self,
        pts_view: np.ndarray,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        eq_w: int,
        eq_h: int,
    ) -> np.ndarray:
        """
        Inverse maps 2D coordinates from perspective view space (x, y)
        back to equirectangular 360° pixel coordinates (u, v).
        """
        if len(pts_view) == 0:
            return np.empty((0, 2), dtype=np.float32)

        f = (0.5 * self.out_w) / np.tan(np.deg2rad(fov_deg) / 2.0)

        xc = pts_view[:, 0] - (self.out_w - 1) / 2.0
        yc = -(pts_view[:, 1] - (self.out_h - 1) / 2.0)
        zc = np.full_like(xc, f)

        norm = np.sqrt(xc * xc + yc * yc + zc * zc)
        norm = np.maximum(norm, 1e-6)
        xyz = np.stack([xc / norm, yc / norm, zc / norm], axis=-1)

        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)

        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(pitch), -np.sin(pitch)],
                [0.0, np.sin(pitch), np.cos(pitch)],
            ],
            dtype=np.float32,
        )
        ry = np.array(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=np.float32,
        )
        rot = ry @ rx
        xyz_rot = xyz @ rot.T

        lon = np.arctan2(xyz_rot[..., 0], xyz_rot[..., 2])
        lat = np.arcsin(np.clip(xyz_rot[..., 1], -1.0, 1.0))

        u = ((lon / (2.0 * np.pi) + 0.5) * (eq_w - 1)).astype(np.float32)
        v = ((0.5 - lat / np.pi) * (eq_h - 1)).astype(np.float32)

        return np.column_stack([u, v])

    def render_equirect_annotations(
        self,
        eq_img: np.ndarray,
        road_type: str,
        detections: List[HazardDetection],
        is_paved: bool = True,
        frame_id: str = "",
        gpx_pt: Optional[GpxPoint] = None,
        class_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Re-projects bounding boxes and segmentation polygons from all perspective views
        back onto the 360° equirectangular stitched frame. Applies panoramic spatial NMS,
        0°/360° seam unwrapping, translucent polygon overlays, pill badges, and road surface pill.
        """
        if class_colors is None:
            class_colors = {
                "trash": (0, 165, 255),
                "dump": (0, 69, 255),
                "open_drainage": (0, 215, 255),
                "road_damage": (50, 50, 255),
                "pothole": (0, 0, 235),
            }

        annotated = eq_img.copy()
        eq_h, eq_w = annotated.shape[:2]

        reprojected_items = []
        for det in detections:
            yaw_deg = self.VIEW_YAWS.get(det.view_name, 0.0)
            if det.mask_polygon and len(det.mask_polygon) >= 3:
                raw_pts = np.array(det.mask_polygon, dtype=np.float32)
                sampled_pts = []
                n_pts = len(raw_pts)
                for i in range(n_pts):
                    p1 = raw_pts[i]
                    p2 = raw_pts[(i + 1) % n_pts]
                    for t in np.linspace(0.0, 1.0, 8, endpoint=False):
                        sampled_pts.append(p1 * (1.0 - t) + p2 * t)
                sampled_pts = np.array(sampled_pts, dtype=np.float32)
            else:
                x1, y1, x2, y2 = det.bbox_xyxy
                sampled_pts = []
                for t in np.linspace(0.0, 1.0, 20, endpoint=False):
                    sampled_pts.append([x1 + t * (x2 - x1), y1])
                for t in np.linspace(0.0, 1.0, 20, endpoint=False):
                    sampled_pts.append([x2, y1 + t * (y2 - y1)])
                for t in np.linspace(0.0, 1.0, 20, endpoint=False):
                    sampled_pts.append([x2 - t * (x2 - x1), y2])
                for t in np.linspace(0.0, 1.0, 20, endpoint=False):
                    sampled_pts.append([x1, y2 - t * (y2 - y1)])
                sampled_pts = np.array(sampled_pts, dtype=np.float32)

            eq_pts = self.perspective_to_equirect_pts(
                sampled_pts, yaw_deg, self.pitch_deg, self.fov_deg, eq_w, eq_h
            )
            u_min, u_max = float(np.min(eq_pts[:, 0])), float(np.max(eq_pts[:, 0]))
            v_min, v_max = float(np.min(eq_pts[:, 1])), float(np.max(eq_pts[:, 1]))

            reprojected_items.append(
                {
                    "det": det,
                    "view_name": det.view_name,
                    "eq_pts": eq_pts,
                    "bbox_eq": [u_min, v_min, u_max, v_max],
                    "has_mask": bool(det.mask_polygon and len(det.mask_polygon) >= 3),
                    "conf": det.confidence,
                    "class_name": det.class_name,
                }
            )

        # Cross-view Spatial NMS across overlapping view seams
        reprojected_items.sort(key=lambda it: (it["has_mask"], it["conf"]), reverse=True)
        filtered_items = []
        for item in reprojected_items:
            u1, v1, u2, v2 = item["bbox_eq"]
            c_name = item["class_name"]
            stype = item["det"].subtype
            is_dup = False
            for kept in filtered_items:
                is_compat = (
                    kept["class_name"] == c_name
                    or ({kept["class_name"], c_name}.issubset({"road_damage", "pothole"}))
                )
                if is_compat:
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
                        if iou > 0.15 or iomin > 0.35:
                            is_dup = True
                            break
            if not is_dup:
                filtered_items.append(item)

        # Draw translucent polygon mask overlay in one fast pass
        poly_data = []
        for item in filtered_items:
            det = item["det"]
            color = class_colors.get(det.class_name, (0, 255, 0))
            eq_pts = item["eq_pts"]

            x_coords = eq_pts[:, 0]
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            instances = []
            if (x_max - x_min) > (0.5 * eq_w):
                unwrapped = eq_pts.copy()
                unwrapped[unwrapped[:, 0] < (0.5 * eq_w), 0] += eq_w
                instances.append(unwrapped)
                shifted = unwrapped.copy()
                shifted[:, 0] -= eq_w
                instances.append(shifted)
            else:
                instances.append(eq_pts)

            poly_data.append((instances, color, det))

        if poly_data:
            mask_layer = annotated.copy()
            for instances, color, _ in poly_data:
                for poly_pts in instances:
                    cv2.fillPoly(mask_layer, [np.int32([poly_pts])], color)
            annotated = cv2.addWeighted(mask_layer, 0.25, annotated, 0.75, 0)

        # Draw fine polygon boundaries and label pill badges
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, min(1.0, eq_w / 5760.0 * 0.75))
        pill_thick = 1 if font_scale < 0.7 else 2

        for instances, color, det in poly_data:
            for poly_pts in instances:
                pts_int = np.int32([poly_pts])
                cv2.polylines(annotated, pts_int, isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

                min_y_idx = np.argmin(poly_pts[:, 1])
                pill_x = int(poly_pts[min_y_idx, 0])
                pill_y = int(poly_pts[min_y_idx, 1]) - 10

                label_text = f"{det.class_name} {det.confidence:.2f}"
                if det.subtype:
                    label_text += f" [{det.subtype}]"

                (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, pill_thick)
                pill_x1 = max(10, min(eq_w - tw - 20, pill_x - 8))
                pill_y1 = max(10, pill_y - th - 8)
                pill_x2 = pill_x1 + tw + 16
                pill_y2 = pill_y1 + th + 8

                pill_overlay = annotated[pill_y1:pill_y2, pill_x1:pill_x2].copy()
                pill_bg = np.full_like(pill_overlay, (20, 20, 20))
                blended_pill = cv2.addWeighted(pill_bg, 0.70, pill_overlay, 0.30, 0)
                blended_pill[:, :4] = color
                annotated[pill_y1:pill_y2, pill_x1:pill_x2] = blended_pill

                cv2.putText(
                    annotated,
                    label_text,
                    (pill_x1 + 8, pill_y2 - 6),
                    font,
                    font_scale,
                    (255, 255, 255),
                    pill_thick,
                    cv2.LINE_AA,
                )

        # Canonical Road Type Pill
        surface_tag = "Paved" if is_paved else "Unpaved/Mud"
        road_text = f"Road: {road_type.capitalize()} ({surface_tag})"
        rt_scale = max(0.65, min(1.3, eq_w / 5760.0 * 1.05))
        rt_thick = 2
        (rt_w, rt_h), _ = cv2.getTextSize(road_text, font, rt_scale, rt_thick)

        center_x = int(eq_w * 0.5)
        center_y = int(eq_h * 0.65)
        rx1 = max(10, center_x - int(rt_w / 2) - 16)
        ry1 = max(10, center_y - int(rt_h / 2) - 10)
        rx2 = min(eq_w - 10, rx1 + rt_w + 32)
        ry2 = min(eq_h - 10, ry1 + rt_h + 20)

        road_crop = annotated[ry1:ry2, rx1:rx2].copy()
        road_bg = np.full_like(road_crop, (15, 15, 15))
        blended_road = cv2.addWeighted(road_bg, 0.75, road_crop, 0.25, 0)
        blended_road[:, :6] = (0, 215, 255)  # Gold/amber stripe
        annotated[ry1:ry2, rx1:rx2] = blended_road

        cv2.putText(
            annotated,
            road_text,
            (rx1 + 16, ry2 - 10),
            font,
            rt_scale,
            (255, 255, 255),
            rt_thick,
            cv2.LINE_AA,
        )

        # Top Header Pill (Frame ID, GPS, Hazards)
        hdr_parts = [frame_id] if frame_id else []
        if gpx_pt and gpx_pt.lat is not None and gpx_pt.lon is not None:
            hdr_parts.append(f"GPS: ({gpx_pt.lat:.6f}, {gpx_pt.lon:.6f})")
            if gpx_pt.ele is not None:
                hdr_parts.append(f"Ele: {gpx_pt.ele:.1f}m")
        pothole_cnt = sum(1 for it in filtered_items if it["det"].class_name == "pothole")
        dmg_cnt = sum(1 for it in filtered_items if it["det"].class_name == "road_damage")
        drain_cnt = sum(1 for it in filtered_items if it["det"].class_name == "open_drainage")
        waste_cnt = sum(1 for it in filtered_items if it["det"].class_name in ("trash", "dump"))
        hdr_parts.append(
            f"Hazards: {len(filtered_items)} (Potholes: {pothole_cnt}, Damage: {dmg_cnt}, Drainage: {drain_cnt}, Waste: {waste_cnt})"
        )
        hdr_text = " | ".join(hdr_parts)

        hdr_scale = max(0.55, min(1.1, eq_w / 5760.0 * 0.85))
        (hw, hh), _ = cv2.getTextSize(hdr_text, font, hdr_scale, 2)
        hx1, hy1 = 20, 20
        hx2, hy2 = min(eq_w - 20, hx1 + hw + 28), min(eq_h - 20, hy1 + hh + 20)

        hdr_crop = annotated[hy1:hy2, hx1:hx2].copy()
        hdr_bg = np.full_like(hdr_crop, (20, 20, 20))
        blended_hdr = cv2.addWeighted(hdr_bg, 0.70, hdr_crop, 0.30, 0)
        blended_hdr[:, :5] = (0, 165, 255)  # Orange stripe
        annotated[hy1:hy2, hx1:hx2] = blended_hdr

        cv2.putText(
            annotated,
            hdr_text,
            (hx1 + 14, hy2 - 8),
            font,
            hdr_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return annotated


# ---------------------------------------------------------------------------
# Stage 1: Road Surface Type Classifier (classifier.pth)
# ---------------------------------------------------------------------------

class RoadTypeClassifier:
    """
    Loads `classifier.pth` and classifies the village street surface type
    (e.g., asphalt, concrete, mud/unpaved, gravel).
    """

    UNPAVED_KEYWORDS = {"mud", "kuchha", "unpaved", "dirt", "gravel", "earth", "sand", "soil"}

    def __init__(
        self,
        weights_path: Optional[str],
        class_names: List[str],
        arch: str = "resnet18",
        device: str = "cpu",
        input_size: Optional[int] = None,
    ) -> None:
        self.weights_path = weights_path
        self.class_names = class_names
        self.arch = arch
        import torch
        if device != "cpu" and not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = device

        if input_size is None and weights_path:
            import re
            m = re.search(r"(\d{3,4})", Path(weights_path).name)
            self.input_size = int(m.group(1)) if m and 128 <= int(m.group(1)) <= 1024 else 224
        else:
            self.input_size = input_size or 224

        self.model: Any = None
        self.backend: str = "none"

        if weights_path and Path(weights_path).is_file():
            self._load_weights(weights_path)
        elif weights_path:
            logger.warning("Classifier weights not found at '%s'. Defaulting to 'paved_road'.", weights_path)

    def _load_weights(self, path: str) -> None:
        import torch

        map_loc = "cpu" if (self.device == "cpu" or not torch.cuda.is_available()) else self.device
        ckpt = torch.load(path, map_location=map_loc, weights_only=False)

        if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "names"):
            from ultralytics import YOLO

            self.model = YOLO(path)
            self.backend = "ultralytics"
            if isinstance(self.model.names, dict):
                self.class_names = [self.model.names[i] for i in sorted(self.model.names.keys())]
            logger.info("Loaded classifier.pth via Ultralytics (%d classes: %s)", len(self.class_names), self.class_names)
            return

        if hasattr(ckpt, "eval") and callable(ckpt.eval):
            self.model = ckpt.to(self.device).eval()
            self.backend = "torch_module"
            logger.info("Loaded classifier.pth as full PyTorch nn.Module.")
            return

        state_dict = ckpt
        if isinstance(ckpt, dict):
            for key in ("state_dict", "model_state_dict", "net", "model"):
                if key in ckpt and isinstance(ckpt[key], dict):
                    state_dict = ckpt[key]
                    break
            if "class_names" in ckpt and isinstance(ckpt["class_names"], (list, tuple)):
                self.class_names = list(ckpt["class_names"])
            elif "classes" in ckpt and isinstance(ckpt["classes"], (list, tuple)):
                self.class_names = list(ckpt["classes"])

        if isinstance(state_dict, dict):
            clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
            
            # Auto-detect ConvNeXt-Tiny architecture (classifier.pth)
            if any("features.0.0.weight" in k for k in clean_sd) and any("classifier.2" in k for k in clean_sd):
                self.arch = "convnext_tiny"
                if self.class_names == ["asphalt", "concrete", "mud_unpaved", "gravel"]:
                    self.class_names = ["concrete", "dirt", "noise", "tar"]
            elif any("features.0.0.weight" in k and clean_sd[k].shape == torch.Size([16, 3, 3, 3]) for k in clean_sd):
                self.arch = "mobilenet_v3_small"

            num_classes = self._infer_num_classes(clean_sd, default=len(self.class_names))
            while len(self.class_names) < num_classes:
                self.class_names.append(f"road_type_{len(self.class_names)}")

            self.model = self._build_torchvision_backbone(self.arch, num_classes)
            missing, unexpected = self.model.load_state_dict(clean_sd, strict=False)
            self.model = self.model.to(self.device).eval()
            self.backend = "torch_module"
            logger.info(
                "Loaded classifier.pth (%s, %d classes: %s | missing keys: %d, unexpected: %d)",
                self.arch,
                num_classes,
                self.class_names[:num_classes],
                len(missing),
                len(unexpected),
            )

    @staticmethod
    def _infer_num_classes(state_dict: Dict[str, Any], default: int) -> int:
        for key in reversed(list(state_dict.keys())):
            tensor = state_dict[key]
            if hasattr(tensor, "ndim") and tensor.ndim == 2 and key.endswith("weight"):
                return int(tensor.shape[0])
        return default

    @staticmethod
    def _build_torchvision_backbone(arch: str, num_classes: int) -> Any:
        import torch.nn as nn
        import torchvision.models as models

        arch_lower = arch.lower()
        if "convnext" in arch_lower:
            net = models.convnext_tiny(weights=None)
            net.classifier[2] = nn.Linear(net.classifier[2].in_features, num_classes)
        elif arch_lower == "resnet18":
            net = models.resnet18(weights=None)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif arch_lower == "resnet34":
            net = models.resnet34(weights=None)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif arch_lower == "resnet50":
            net = models.resnet50(weights=None)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif arch_lower == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(weights=None)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        elif arch_lower == "efficientnet_b0":
            net = models.efficientnet_b0(weights=None)
            net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported torchvision classifier architecture: {arch}")
        return net

    def predict_road_type(self, views: Dict[str, np.ndarray]) -> Tuple[str, float, bool]:
        if self.model is None:
            return "paved_default", 1.0, True

        primary_view = views.get("front", views.get("Front_0deg", next(iter(views.values()))))
        h, w = primary_view.shape[:2]
        road_crop = primary_view[int(h * 0.50) : int(h * 0.95), int(w * 0.20) : int(w * 0.80)]

        if self.backend == "ultralytics":
            res = self.model.predict(road_crop, imgsz=self.input_size, verbose=False)[0]
            top1_idx = int(res.probs.top1)
            conf = float(res.probs.top1conf)
            label = self.class_names[top1_idx] if top1_idx < len(self.class_names) else f"class_{top1_idx}"
        else:
            import torch

            rgb = cv2.cvtColor(road_crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            tensor = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tensor = (tensor - mean) / std
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

            with torch.no_grad():
                inp = torch.from_numpy(tensor).to(self.device)
                logits = self.model(inp)
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
                top1_idx = int(np.argmax(probs))
                conf = float(probs[top1_idx])
                label = self.class_names[top1_idx] if top1_idx < len(self.class_names) else f"class_{top1_idx}"

                # Noise fallback logic
                if label == "noise":
                    sorted_idxs = np.argsort(probs)[::-1]
                    runner_up_idx = int(sorted_idxs[1]) if len(sorted_idxs) > 1 else top1_idx
                    label = self.class_names[runner_up_idx]
                    conf = float(probs[runner_up_idx])

        is_paved = not any(k in label.lower() for k in self.UNPAVED_KEYWORDS)
        return label, round(conf, 4), is_paved


# ---------------------------------------------------------------------------
# Stage 2, 3 & 4: Unified Multi-Model Hazard Pipeline
# ---------------------------------------------------------------------------

class VillageStreetHazardPipeline:
    CLASS_COLORS_BGR = {
        "trash": (0, 165, 255),          # Orange
        "dump": (0, 0, 255),             # Red
        "open_drainage": (255, 180, 0),  # Cyan-Blue
        "road_damage": (0, 220, 255),    # Yellow
        "pothole": (0, 0, 235),          # Crimson Red
    }

    def __init__(
        self,
        detector_weights: Optional[str],
        segmentation_weights: Optional[str],
        rdd_weights: Optional[str] = None,
        classifier: Optional[RoadTypeClassifier] = None,
        device: str = "0",
        imgsz: int = 640,
        conf_waste: float = 0.25,
        conf_drainage: float = 0.20,
        conf_road: float = 0.15,
        conf_road_unpaved: float = 0.20,
        enable_road_tiling: bool = True,
        apply_clahe: bool = False,
    ) -> None:
        from ultralytics import YOLO

        self.device = device
        self.imgsz = imgsz
        self.conf_waste = conf_waste
        self.conf_drainage = conf_drainage
        self.conf_road_paved = conf_road
        self.conf_road_unpaved = conf_road_unpaved
        self.enable_road_tiling = enable_road_tiling
        self.apply_clahe = apply_clahe
        self.classifier = classifier

        self.detector_model = YOLO(detector_weights) if detector_weights and Path(detector_weights).is_file() else None
        self.seg_model = (
            YOLO(segmentation_weights) if segmentation_weights and Path(segmentation_weights).is_file() else None
        )
        self.rdd_model = YOLO(rdd_weights) if rdd_weights and Path(rdd_weights).is_file() else None

        if self.detector_model:
            logger.info("Loaded Stage-2 Detector (%s) classes: %s", detector_weights, self.detector_model.names)
        if self.seg_model:
            logger.info("Loaded Stage-3 Segmenter (%s) classes: %s", segmentation_weights, self.seg_model.names)
        if self.rdd_model:
            logger.info("Loaded Stage-4 RDD Defect Classifier (%s) classes: %s", rdd_weights, self.rdd_model.names)

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _enhance_road_contrast(self, img: np.ndarray) -> np.ndarray:
        if not self.apply_clahe:
            return img
        out = img.copy()
        h = out.shape[0]
        roi_y = int(h * 0.35)
        lab = cv2.cvtColor(out[roi_y:], cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._clahe.apply(l_ch)
        out[roi_y:] = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        return out

    @staticmethod
    def _is_on_ground_road(bbox_xyxy: List[int], class_name: str, min_cy: int = 450, min_y2: int = 480) -> bool:
        """
        Filters out false positives on building walls, roofs, trees, or sky.
        In 1024x1024 perspective views with -20 deg pitch, ground road defects have
        cy >= 450 and bottom edge y2 >= 480.
        Roadside waste (trash, dump) can sit slightly higher (cy >= 380, y2 >= 400).
        """
        x1, y1, x2, y2 = bbox_xyxy
        cy = (y1 + y2) / 2.0
        if class_name in ("trash", "dump"):
            return cy >= 380 and y2 >= 400
        return cy >= min_cy and y2 >= min_y2

    @staticmethod
    def _compute_overlap(box_a: List[int], box_b: List[int]) -> Tuple[float, float]:
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])
        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        if inter <= 0:
            return 0.0, 0.0
        area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
        area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
        iou = inter / float(area_a + area_b - inter)
        iomin = inter / float(min(area_a, area_b))
        return iou, iomin

    def _deduplicate_detections(
        self, detections: List[HazardDetection], iou_threshold: float = 0.40, iomin_threshold: float = 0.55
    ) -> List[HazardDetection]:
        if not detections:
            return []

        def sort_priority(d: HazardDetection):
            pothole_score = 2 if d.class_name == "pothole" else (1 if d.class_name == "road_damage" else 0)
            has_mask = 1 if (d.mask_polygon and len(d.mask_polygon) >= 3) else 0
            return (pothole_score, has_mask, d.confidence)

        sorted_dets = sorted(detections, key=sort_priority, reverse=True)
        kept: List[HazardDetection] = []

        for candidate in sorted_dets:
            duplicate = False
            for k in kept:
                is_compatible_type = (
                    k.class_name == candidate.class_name
                    or ({k.class_name, candidate.class_name}.issubset({"road_damage", "pothole"}))
                )
                if not is_compatible_type:
                    continue

                iou, iomin = self._compute_overlap(candidate.bbox_xyxy, k.bbox_xyxy)
                if iou >= iou_threshold or iomin >= iomin_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)

        return kept

    def _generate_road_tiles(self, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        tiles: List[Tuple[int, int, int, int]] = []
        y_start = max(0, int(h * 0.35))
        tile_h = min(self.imgsz, h - y_start)
        tile_w = min(self.imgsz, w)
        if h <= self.imgsz and w <= self.imgsz:
            return tiles

        step_x = max(1, int(tile_w * 0.75))
        step_y = max(1, int(tile_h * 0.75))

        y_positions = list(range(y_start, max(y_start + 1, h - tile_h + 1), step_y))
        if not y_positions or y_positions[-1] + tile_h < h:
            y_positions.append(max(y_start, h - tile_h))

        x_positions = list(range(0, max(1, w - tile_w + 1), step_x))
        if not x_positions or x_positions[-1] + tile_w < w:
            x_positions.append(max(0, w - tile_w))

        for y0 in sorted(set(y_positions)):
            for x0 in sorted(set(x_positions)):
                tiles.append((x0, y0, x0 + tile_w, y0 + tile_h))
        return tiles

    def _run_detector_on_view(
        self, view_name: str, img: np.ndarray, road_conf_thresh: float, allow_road_damage: bool = True
    ) -> List[HazardDetection]:
        if self.detector_model is None:
            return []

        min_conf = min(self.conf_waste, road_conf_thresh) if allow_road_damage else self.conf_waste

        res = self.detector_model.predict(
            img,
            imgsz=self.imgsz,
            conf=min_conf,
            device=self.device,
            verbose=False,
        )[0]

        out: List[HazardDetection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out

        xyxy_arr = res.boxes.xyxy.cpu().numpy()
        conf_arr = res.boxes.conf.cpu().numpy()
        cls_arr = res.boxes.cls.cpu().numpy().astype(int)

        for xyxy, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
            cls_name = self.detector_model.names.get(cls_id, str(cls_id))
            if cls_name == "open_drainage":
                continue
            if cls_name == "road_damage" and not allow_road_damage:
                continue
            req_conf = road_conf_thresh if cls_name == "road_damage" else self.conf_waste
            if conf < req_conf:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            if not self._is_on_ground_road([x1, y1, x2, y2], cls_name):
                continue

            area = float(max(0, x2 - x1) * max(0, y2 - y1))
            out.append(
                HazardDetection(
                    view_name=view_name,
                    class_id=int(cls_id),
                    class_name=cls_name,
                    subtype=None,
                    confidence=round(float(conf), 4),
                    bbox_xyxy=[x1, y1, x2, y2],
                    mask_polygon=[],
                    area_px=area,
                    source_model="detector.pt",
                )
            )
        return out

    def _extract_seg_predictions(
        self,
        res: Any,
        view_name: str,
        offset_x: int,
        offset_y: int,
        road_conf_thresh: float,
        allow_road_damage: bool = True,
    ) -> List[HazardDetection]:
        out: List[HazardDetection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out

        xyxy_arr = res.boxes.xyxy.cpu().numpy()
        conf_arr = res.boxes.conf.cpu().numpy()
        cls_arr = res.boxes.cls.cpu().numpy().astype(int)
        polys = res.masks.xy if (res.masks is not None and hasattr(res.masks, "xy")) else [None] * len(xyxy_arr)

        for xyxy, conf, cls_id, poly in zip(xyxy_arr, conf_arr, cls_arr, polys):
            cls_name = self.seg_model.names.get(cls_id, str(cls_id))
            if cls_name == "road_damage" and not allow_road_damage:
                continue
            req_conf = road_conf_thresh if cls_name == "road_damage" else self.conf_drainage
            if conf < req_conf:
                continue

            x1 = int(round(xyxy[0])) + offset_x
            y1 = int(round(xyxy[1])) + offset_y
            x2 = int(round(xyxy[2])) + offset_x
            y2 = int(round(xyxy[3])) + offset_y

            if not self._is_on_ground_road([x1, y1, x2, y2], cls_name):
                continue

            polygon_pts: List[List[int]] = []
            area = float(max(0, x2 - x1) * max(0, y2 - y1))
            if poly is not None and len(poly) >= 3:
                shifted = np.round(poly).astype(int)
                shifted[:, 0] += offset_x
                shifted[:, 1] += offset_y
                polygon_pts = shifted.tolist()
                area = float(cv2.contourArea(shifted.astype(np.float32)))

            out.append(
                HazardDetection(
                    view_name=view_name,
                    class_id=int(cls_id),
                    class_name=cls_name,
                    subtype=None,
                    confidence=round(float(conf), 4),
                    bbox_xyxy=[x1, y1, x2, y2],
                    mask_polygon=polygon_pts,
                    area_px=round(area, 1),
                    source_model="segmentation.pt",
                )
            )
        return out

    def _run_segmenter_on_view(
        self,
        view_name: str,
        img: np.ndarray,
        road_conf_thresh: float,
        allow_road_damage: bool = True,
        enable_tiling: bool = True,
    ) -> List[HazardDetection]:
        if self.seg_model is None:
            return []

        min_conf = min(self.conf_drainage, road_conf_thresh) if allow_road_damage else self.conf_drainage
        enhanced = self._enhance_road_contrast(img)
        h, w = enhanced.shape[:2]

        res_full = self.seg_model.predict(
            enhanced,
            imgsz=self.imgsz,
            conf=min_conf,
            device=self.device,
            verbose=False,
        )[0]
        detections = self._extract_seg_predictions(
            res_full, view_name, 0, 0, road_conf_thresh, allow_road_damage=allow_road_damage
        )

        if self.enable_road_tiling and enable_tiling:
            for x1, y1, x2, y2 in self._generate_road_tiles(h, w):
                tile = enhanced[y1:y2, x1:x2]
                res_tile = self.seg_model.predict(
                    tile,
                    imgsz=self.imgsz,
                    conf=min_conf,
                    device=self.device,
                    verbose=False,
                )[0]
                detections.extend(
                    self._extract_seg_predictions(
                        res_tile, view_name, x1, y1, road_conf_thresh, allow_road_damage=allow_road_damage
                    )
                )

        return self._deduplicate_detections(detections, iou_threshold=0.40, iomin_threshold=0.55)

    def _run_rdd_on_view(
        self, view_name: str, img: np.ndarray, road_conf_thresh: float, is_paved: bool, is_forward: bool = True
    ) -> List[HazardDetection]:
        if self.rdd_model is None or not is_forward:
            return []

        # Potholes are high-priority hazards: use sensitive threshold to catch dark cavities
        pothole_conf_thresh = min(road_conf_thresh, 0.15 if is_paved else 0.18)
        min_conf = min(pothole_conf_thresh, road_conf_thresh)
        enhanced = self._enhance_road_contrast(img)
        h, w = enhanced.shape[:2]

        def _extract_rdd_preds(res: Any, offset_x: int, offset_y: int) -> List[HazardDetection]:
            dets = []
            if res.boxes is None or len(res.boxes) == 0:
                return dets
            xyxy_arr = res.boxes.xyxy.cpu().numpy()
            conf_arr = res.boxes.conf.cpu().numpy()
            cls_arr = res.boxes.cls.cpu().numpy().astype(int)

            for xyxy, conf, cls_id in zip(xyxy_arr, conf_arr, cls_arr):
                cls_name = self.rdd_model.names.get(cls_id, str(cls_id))
                is_pothole = (cls_id == 3 or "pothole" in cls_name.lower())
                req_conf = pothole_conf_thresh if is_pothole else road_conf_thresh
                if conf < req_conf:
                    continue

                x1 = int(round(xyxy[0])) + offset_x
                y1 = int(round(xyxy[1])) + offset_y
                x2 = int(round(xyxy[2])) + offset_x
                y2 = int(round(xyxy[3])) + offset_y

                final_cls_name = "pothole" if is_pothole else "road_damage"
                if not self._is_on_ground_road([x1, y1, x2, y2], final_cls_name):
                    continue

                area = float(max(0, x2 - x1) * max(0, y2 - y1))
                dets.append(
                    HazardDetection(
                        view_name=view_name,
                        class_id=int(cls_id),
                        class_name=final_cls_name,
                        subtype="Pothole" if is_pothole else cls_name,
                        confidence=round(float(conf), 4),
                        bbox_xyxy=[x1, y1, x2, y2],
                        mask_polygon=[],
                        area_px=area,
                        source_model="rdd.pt",
                    )
                )
            return dets

        res_full = self.rdd_model.predict(
            enhanced,
            imgsz=self.imgsz,
            conf=min_conf,
            device=self.device,
            verbose=False,
        )[0]
        detections = _extract_rdd_preds(res_full, 0, 0)

        if self.enable_road_tiling:
            for x1, y1, x2, y2 in self._generate_road_tiles(h, w):
                tile = enhanced[y1:y2, x1:x2]
                res_tile = self.rdd_model.predict(
                    tile,
                    imgsz=self.imgsz,
                    conf=min_conf,
                    device=self.device,
                    verbose=False,
                )[0]
                detections.extend(_extract_rdd_preds(res_tile, x1, y1))

        return self._deduplicate_detections(detections, iou_threshold=0.40, iomin_threshold=0.55)

    @staticmethod
    def _extract_organic_contour(crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts a localized organic polygon boundary for road defects
        when neural segmentation models output an empty mask.
        Uses adaptive Otsu dark-defect thresholding and contour approximation.
        """
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        med = float(np.median(blurred))
        thresh = cv2.threshold(blurred, int(med * 0.88), 255, cv2.THRESH_BINARY_INV)[1]
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 25:
                poly = cv2.approxPolyDP(largest, 0.02 * cv2.arcLength(largest, True), True)
                if len(poly) >= 3:
                    return poly.reshape(-1, 2)
        # Fallback to smooth 8-point rounded polygon within the box
        h, w = crop.shape[:2]
        pts = np.array(
            [
                [int(0.25 * w), int(0.08 * h)],
                [int(0.75 * w), int(0.08 * h)],
                [int(0.94 * w), int(0.35 * h)],
                [int(0.94 * w), int(0.65 * h)],
                [int(0.75 * w), int(0.92 * h)],
                [int(0.25 * w), int(0.92 * h)],
                [int(0.06 * w), int(0.65 * h)],
                [int(0.06 * w), int(0.35 * h)],
            ],
            dtype=np.int32,
        )
        return pts

    def _mask_defect_candidates(
        self,
        view_img: np.ndarray,
        unsegmented_dets: List[HazardDetection],
        existing_seg_dets: List[HazardDetection],
        road_conf_thresh: float,
    ) -> List[HazardDetection]:
        """
        Refines candidate defect boxes from detector.pt and rdd.pt into high-fidelity
        segmentation polygon masks so that NO potholes/road defects are missed and NO boxes are drawn.
        If an existing segmented detection covers the defect, inherits/promotes that mask;
        otherwise runs neural crop segmentation or organic dark-cavity contouring.
        """
        if not unsegmented_dets:
            return []

        h, w = view_img.shape[:2]
        refined: List[HazardDetection] = []

        for d in unsegmented_dets:
            # Check if an existing segmentation already covers this defect
            matched_seg: Optional[HazardDetection] = None
            for s in existing_seg_dets:
                if bool(s.mask_polygon and len(s.mask_polygon) >= 3):
                    iou, iomin = self._compute_overlap(d.bbox_xyxy, s.bbox_xyxy)
                    if iou >= 0.35 or iomin >= 0.50:
                        matched_seg = s
                        break

            if matched_seg is not None:
                if d.class_name == "pothole":
                    matched_seg.class_name = "pothole"
                    matched_seg.subtype = "Pothole"
                    matched_seg.confidence = max(matched_seg.confidence, d.confidence)
                elif d.subtype and not matched_seg.subtype:
                    matched_seg.subtype = d.subtype
                continue

            # Defect region crop with 15% context padding
            x1, y1, x2, y2 = d.bbox_xyxy
            pad_x = max(12, int(0.15 * (x2 - x1)))
            pad_y = max(12, int(0.15 * (y2 - y1)))
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w, x2 + pad_x)
            cy2 = min(h, y2 + pad_y)

            crop = view_img[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                continue

            extracted_mask: Optional[List[List[int]]] = None

            # 1. Targeted neural segmentation pass on zoomed-in crop
            if self.seg_model is not None:
                crop_res = self.seg_model.predict(
                    crop,
                    imgsz=self.imgsz,
                    conf=min(0.06, road_conf_thresh),
                    device=self.device,
                    verbose=False,
                )[0]
                if crop_res.masks is not None and len(crop_res.masks) > 0:
                    polys = crop_res.masks.xy
                    boxes = crop_res.boxes
                    for p_idx, p in enumerate(polys):
                        if len(p) >= 3:
                            cls_id = int(boxes.cls[p_idx]) if boxes is not None else 1
                            cls_name = self.seg_model.names.get(cls_id, "road_damage")
                            if cls_name == "road_damage":
                                p_arr = np.array(p, dtype=np.float32)
                                p_arr[:, 0] += cx1
                                p_arr[:, 1] += cy1
                                extracted_mask = [[int(round(pt[0])), int(round(pt[1]))] for pt in p_arr]
                                break
                    if extracted_mask is None and len(polys) > 0 and len(polys[0]) >= 3:
                        p_arr = np.array(polys[0], dtype=np.float32)
                        p_arr[:, 0] += cx1
                        p_arr[:, 1] += cy1
                        extracted_mask = [[int(round(pt[0])), int(round(pt[1]))] for pt in p_arr]

            # 2. Organic contour fallback if neural segmentor did not produce mask
            if extracted_mask is not None:
                d.mask_polygon = extracted_mask
                d.source_model += "+crop_seg"
                refined.append(d)
            else:
                local_poly = self._extract_organic_contour(crop)
                if local_poly is not None and len(local_poly) >= 3:
                    local_poly = local_poly.astype(np.float32)
                    local_poly[:, 0] += cx1
                    local_poly[:, 1] += cy1
                    d.mask_polygon = [[int(round(pt[0])), int(round(pt[1]))] for pt in local_poly]
                    d.source_model += "+contour_mask"
                    refined.append(d)

        return refined

    def process_frame(
        self,
        frame_id: str,
        frame_index: int,
        views: Dict[str, np.ndarray],
        gpx_pt: Optional[GpxPoint] = None,
    ) -> Tuple[FrameReport, Dict[str, np.ndarray]]:
        if self.classifier is not None:
            road_type, road_conf, is_paved = self.classifier.predict_road_type(views)
        else:
            road_type, road_conf, is_paved = "paved_default", 1.0, True

        road_conf_thresh = self.conf_road_paved if is_paved else self.conf_road_unpaved

        all_detections: List[HazardDetection] = []
        annotated_views: Dict[str, np.ndarray] = {}

        for view_name, view_img in views.items():
            is_forward = view_name.lower() in ("front", "front_0deg", "forward")

            det_dets = self._run_detector_on_view(
                view_name, view_img, road_conf_thresh, allow_road_damage=is_forward
            )
            seg_dets = self._run_segmenter_on_view(
                view_name,
                view_img,
                road_conf_thresh,
                allow_road_damage=is_forward,
                enable_tiling=(self.enable_road_tiling and is_forward),
            )
            rdd_dets = self._run_rdd_on_view(
                view_name, view_img, road_conf_thresh, is_paved=is_paved, is_forward=is_forward
            )

            # Separate waste boxes from defect candidates
            waste_dets = [d for d in det_dets if d.class_name in ("trash", "dump")]
            unsegmented_defects = [d for d in (det_dets + rdd_dets) if d.class_name in ("road_damage", "pothole")]

            # Ensure all defect candidates have polygon masks
            masked_defects = self._mask_defect_candidates(view_img, unsegmented_defects, seg_dets, road_conf_thresh)

            # Deduplicate view detections with IoU + IoMin containment fusion
            combined_candidates = waste_dets + seg_dets + masked_defects
            view_dets = self._deduplicate_detections(combined_candidates, iou_threshold=0.40, iomin_threshold=0.55)

            all_detections.extend(view_dets)
            annotated_views[view_name] = self.draw_view_annotations(
                view_img, view_name, view_dets, road_type, is_paved
            )

        counts: Dict[str, int] = {"trash": 0, "dump": 0, "open_drainage": 0, "road_damage": 0, "pothole": 0}
        for det in all_detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1

        lat = gpx_pt.lat if gpx_pt else None
        lon = gpx_pt.lon if gpx_pt else None
        ele = gpx_pt.ele if gpx_pt else None
        gpx_time = gpx_pt.time_str if gpx_pt else None
        maps_url = f"https://www.google.com/maps?q={lat},{lon}" if (lat is not None and lon is not None) else None

        report = FrameReport(
            frame_id=frame_id,
            frame_index=frame_index,
            latitude=lat,
            longitude=lon,
            elevation=ele,
            gpx_time=gpx_time,
            google_maps_url=maps_url,
            road_type=road_type,
            road_type_confidence=road_conf,
            is_paved_surface=is_paved,
            counts_by_class=counts,
            detections=all_detections,
        )
        return report, annotated_views

    def draw_view_annotations(
        self,
        img: np.ndarray,
        view_name: str,
        detections: List[HazardDetection],
        road_type: str,
        is_paved: bool,
    ) -> np.ndarray:
        canvas = img.copy()
        overlay = img.copy()

        # Pass 1: Translucent polygon mask overlay
        for det in detections:
            color = self.CLASS_COLORS_BGR.get(det.class_name, (0, 255, 0))
            if det.mask_polygon and len(det.mask_polygon) >= 3:
                pts = np.array(det.mask_polygon, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], color)
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        # Pass 2: Bounding boxes for waste ONLY (Zero boxes for road damage, potholes, drainage)
        for det in detections:
            color = self.CLASS_COLORS_BGR.get(det.class_name, (0, 255, 0))
            x1, y1, x2, y2 = det.bbox_xyxy
            has_mask = bool(det.mask_polygon and len(det.mask_polygon) >= 3)
            if not has_mask and det.class_name in ("trash", "dump"):
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Pass 3: Anti-Collision Label Layout Engine
        placed_badges: List[Tuple[int, int, int, int]] = []  # [(bx1, by1, bx2, by2), ...]

        for det in detections:
            color = self.CLASS_COLORS_BGR.get(det.class_name, (0, 255, 0))
            x1, y1, x2, y2 = det.bbox_xyxy
            has_mask = bool(det.mask_polygon and len(det.mask_polygon) >= 3)

            label = f"{det.class_name} {det.confidence:.2f}"
            if det.subtype and det.subtype.lower() != det.class_name.lower():
                label += f" [{det.subtype}]"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)

            if has_mask:
                poly_arr = np.array(det.mask_polygon, dtype=np.int32)
                min_idx = int(np.argmin(poly_arr[:, 1]))
                anchor_x, anchor_y = int(poly_arr[min_idx, 0]), int(poly_arr[min_idx, 1])
                tx1 = max(6, min(canvas.shape[1] - tw - 16, anchor_x - 4))
                ty1 = max(th + 8, anchor_y - 4)
            else:
                anchor_x, anchor_y = x1, y1
                tx1 = max(6, min(canvas.shape[1] - tw - 16, x1))
                ty1 = max(th + 8, y1 - 4)

            badge_w = tw + 8
            badge_h = th + 8
            bx1, by1 = tx1, ty1 - badge_h
            bx2, by2 = tx1 + badge_w, ty1

            shifted = False
            for _ in range(8):
                collision = False
                for px1, py1, px2, py2 in placed_badges:
                    if not (bx2 < px1 - 3 or bx1 > px2 + 3 or by2 < py1 - 3 or by1 > py2 + 3):
                        collision = True
                        break
                if collision:
                    if by1 - (badge_h + 4) >= 36:
                        by1 -= (badge_h + 4)
                        by2 -= (badge_h + 4)
                        ty1 -= (badge_h + 4)
                    else:
                        by1 += (badge_h + 4)
                        by2 += (badge_h + 4)
                        ty1 += (badge_h + 4)
                    shifted = True
                else:
                    break

            placed_badges.append((bx1, by1, bx2, by2))

            if shifted and abs(ty1 - anchor_y) > 20:
                cv2.line(canvas, (anchor_x, anchor_y), (bx1 + int(badge_w / 2), by2), color, 1, cv2.LINE_AA)

            # Draw crisp badge with dark contrast border
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), color, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (20, 20, 20), 1)
            cv2.putText(
                canvas,
                label,
                (bx1 + 4, ty1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        surface_tag = "Paved" if is_paved else "Unpaved/Mud"
        header = f"{view_name} | Surface: {road_type} ({surface_tag}) | Hits: {len(detections)}"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            header,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    @staticmethod
    def compose_quad_grid(annotated_views: Dict[str, np.ndarray], report: FrameReport) -> np.ndarray:
        keys = list(annotated_views.keys())
        if len(keys) == 1:
            return annotated_views[keys[0]]

        ordered_keys = ["front", "right", "back", "left"]
        panels = [annotated_views[k] for k in ordered_keys if k in annotated_views]
        if len(panels) < 4:
            alt_keys = ["Front_0deg", "Right_90deg", "Back_180deg", "Left_270deg"]
            panels = [annotated_views[k] for k in alt_keys if k in annotated_views]
        if len(panels) < 4:
            panels = list(annotated_views.values())

        h, w = panels[0].shape[:2]
        resized = [cv2.resize(p, (w, h)) for p in panels[:4]]
        while len(resized) < 4:
            resized.append(np.zeros_like(resized[0]))

        top_row = np.hstack((resized[0], resized[1]))
        bot_row = np.hstack((resized[2], resized[3]))
        grid = np.vstack((top_row, bot_row))

        banner = np.zeros((48, grid.shape[1], 3), dtype=np.uint8)
        banner[:] = (35, 35, 35)
        gps_str = (
            f"GPS: ({report.latitude:.6f}, {report.longitude:.6f})"
            if (report.latitude is not None and report.longitude is not None)
            else "GPS: N/A"
        )
        summary_text = (
            f"Frame: {report.frame_id} | {gps_str} | Road: {report.road_type} ({report.road_type_confidence:.2f}) | "
            f"Potholes: {report.counts_by_class.get('pothole', 0)} "
            f"Damage: {report.counts_by_class.get('road_damage', 0)} "
            f"Drainage: {report.counts_by_class.get('open_drainage', 0)} "
            f"Trash: {report.counts_by_class.get('trash', 0)} "
            f"Dump: {report.counts_by_class.get('dump', 0)}"
        )
        cv2.putText(
            banner,
            summary_text,
            (14, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 200),
            2,
            cv2.LINE_AA,
        )
        return np.vstack((banner, grid))


# ---------------------------------------------------------------------------
# CLI & Batch / GPX Geotagged Runner
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-End 360° 4-View Village Street Hazard Detection, Segmentation & GPX Geotagging Pipeline"
    )
    parser.add_argument(
        "--source",
        "--input",
        dest="source",
        type=str,
        required=True,
        help="Path to input 360° frames folder, single image, or .mp4/.avi video file.",
    )
    parser.add_argument(
        "--gpx",
        type=str,
        default=None,
        help="Path to the .gpx GPS track file corresponding to the 360° frames.",
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default="equirectangular",
        choices=["equirectangular", "quad_grid", "single_views"],
        help="How to interpret input frames: 'equirectangular' (splits 360° into 4 views), "
        "'quad_grid' (splits a 2x2 image into 4 views), or 'single_views' (processes as-is).",
    )
    parser.add_argument(
        "--detector-weights",
        type=str,
        default="detector.pt",
        help="Path to YOLO26s detector.pt weights (classes: trash, dump, road_damage).",
    )
    parser.add_argument(
        "--seg-weights",
        type=str,
        default="segmentation.pt",
        help="Path to YOLO26s-seg segmentation.pt weights (classes: open_drainage, road_damage).",
    )
    parser.add_argument(
        "--rdd-weights",
        type=str,
        default="YOLOv8_Small_RDD.pt",
        nargs="?",
        const="YOLOv8_Small_RDD.pt",
        help="Optional path to YOLOv8_Small_RDD.pt for specialized defect & pothole detection (pass 'none' to disable).",
    )
    parser.add_argument(
        "--classifier-weights",
        type=str,
        default="classifier.pth",
        help="Optional path to classifier.pth road-type classifier weights.",
    )
    parser.add_argument(
        "--classifier-arch",
        type=str,
        default="auto",
        choices=["auto", "convnext_tiny", "resnet18", "resnet34", "resnet50", "mobilenet_v3_small", "efficientnet_b0"],
        help="Backbone architecture if classifier.pth is a PyTorch state_dict.",
    )
    parser.add_argument(
        "--classifier-classes",
        type=str,
        default="concrete,dirt,noise,tar",
        help="Comma-separated road type class names for classifier.pth (default: concrete,dirt,noise,tar).",
    )
    parser.add_argument(
        "--classifier-res",
        type=int,
        default=None,
        help="Input resolution for classifier (default: auto-inferred from weights filename, else 224).",
    )
    parser.add_argument("--fov", type=float, default=100.0, help="Horizontal FOV (degrees) for 360° perspective views.")
    parser.add_argument(
        "--pitch",
        type=float,
        default=-20.0,
        help="Vertical pitch angle (degrees, negative tilts down toward the road).",
    )
    parser.add_argument("--view-width", type=int, default=1024, help="Output width of each split perspective view.")
    parser.add_argument("--view-height", type=int, default=1024, help="Output height of each split perspective view.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference resolution (trained at 640).")
    parser.add_argument("--device", type=str, default="0", help="CUDA device ('0') or 'cpu'.")
    parser.add_argument("--vid-stride", type=int, default=5, help="Process every Nth frame when input is a video.")
    parser.add_argument("--no-tiling", action="store_true", help="Disable 640x640 tiled inference on the road strip.")
    parser.add_argument(
        "--apply-clahe",
        action="store_true",
        default=False,
        help="Enable CLAHE local contrast enhancement (disabled by default to preserve neural detection confidence).",
    )
    parser.add_argument(
        "--save-views",
        action="store_true",
        default=True,
        help="Save individual 2D annotated perspective view crop images in output_dir/annotated_views (front, right, back, left). Default is True.",
    )
    parser.add_argument(
        "--no-save-views",
        dest="save_views",
        action="store_false",
        help="Disable saving individual perspective view crop images in annotated_views/.",
    )
    parser.add_argument(
        "--save-quad",
        action="store_true",
        default=False,
        help="Save 2x2 quad grid composite images in output_dir/quad_grids.",
    )
    parser.add_argument(
        "--save-raw-views",
        action="store_true",
        default=False,
        help="Save raw unannotated perspective view crop images in output_dir/raw_views.",
    )
    parser.add_argument(
        "--conf-road-damage",
        "--conf-road",
        dest="conf_road_damage",
        type=float,
        default=0.10,
        help="Confidence threshold for road damage detection/segmentation on paved roads (default: 0.10).",
    )
    parser.add_argument(
        "--conf-road-unpaved",
        type=float,
        default=0.18,
        help="Confidence threshold for road damage on unpaved/dirt roads (default: 0.18).",
    )
    parser.add_argument(
        "--conf-drainage",
        type=float,
        default=0.20,
        help="Confidence threshold for open drainage segmentation (default: 0.20).",
    )
    parser.add_argument(
        "--conf-waste",
        "--conf-garbage",
        dest="conf_waste",
        type=float,
        default=0.25,
        help="Confidence threshold for trash/dump waste detection (default: 0.25).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/village_street_inspection",
        help="Directory to save annotated frames, views, JSON, CSV, and GeoJSON reports.",
    )
    return parser.parse_args()


def save_reports(reports: List[FrameReport], out_dir: Path) -> None:
    json_path = out_dir / "detections_report.json"
    csv_path = out_dir / "detections_summary.csv"
    geojson_path = out_dir / "hazards_geotagged.geojson"

    with open(json_path, "w", encoding="utf-8") as f_json:
        json.dump([asdict(r) for r in reports], f_json, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(
            [
                "frame_id",
                "frame_index",
                "latitude",
                "longitude",
                "elevation",
                "gpx_time",
                "google_maps_url",
                "road_type",
                "road_type_conf",
                "is_paved",
                "annotated_path",
                "view_name",
                "class_name",
                "subtype",
                "confidence",
                "bbox_xyxy",
                "area_px",
                "source_model",
            ]
        )
        for rep in reports:
            if not rep.detections:
                writer.writerow(
                    [
                        rep.frame_id,
                        rep.frame_index,
                        rep.latitude if rep.latitude is not None else "",
                        rep.longitude if rep.longitude is not None else "",
                        rep.elevation if rep.elevation is not None else "",
                        rep.gpx_time or "",
                        rep.google_maps_url or "",
                        rep.road_type,
                        rep.road_type_confidence,
                        rep.is_paved_surface,
                        rep.annotated_path or "",
                        "ALL",
                        "none",
                        "",
                        0.0,
                        "[]",
                        0.0,
                        "",
                    ]
                )
            for det in rep.detections:
                writer.writerow(
                    [
                        rep.frame_id,
                        rep.frame_index,
                        rep.latitude if rep.latitude is not None else "",
                        rep.longitude if rep.longitude is not None else "",
                        rep.elevation if rep.elevation is not None else "",
                        rep.gpx_time or "",
                        rep.google_maps_url or "",
                        rep.road_type,
                        rep.road_type_confidence,
                        rep.is_paved_surface,
                        rep.annotated_path or "",
                        det.view_name,
                        det.class_name,
                        det.subtype or "",
                        det.confidence,
                        json.dumps(det.bbox_xyxy),
                        det.area_px,
                        det.source_model,
                    ]
                )

    # Build GeoJSON FeatureCollection for GIS / Google Earth / Web Maps
    features = []
    for rep in reports:
        if rep.latitude is None or rep.longitude is None:
            continue
        if not rep.detections:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [rep.longitude, rep.latitude],
                },
                "properties": {
                    "frame_id": rep.frame_id,
                    "frame_index": rep.frame_index,
                    "gpx_time": rep.gpx_time,
                    "elevation": rep.elevation,
                    "google_maps_url": rep.google_maps_url,
                    "annotated_path": rep.annotated_path or "",
                    "road_type": rep.road_type,
                    "is_paved_surface": rep.is_paved_surface,
                    "trash_count": rep.counts_by_class.get("trash", 0),
                    "dump_count": rep.counts_by_class.get("dump", 0),
                    "open_drainage_count": rep.counts_by_class.get("open_drainage", 0),
                    "road_damage_count": rep.counts_by_class.get("road_damage", 0),
                    "pothole_count": rep.counts_by_class.get("pothole", 0),
                    "detections": [
                        {
                            "view": d.view_name,
                            "class": d.class_name,
                            "subtype": d.subtype,
                            "confidence": d.confidence,
                            "area_px": d.area_px,
                        }
                        for d in rep.detections
                    ],
                },
            }
        )

    with open(geojson_path, "w", encoding="utf-8") as f_geo:
        json.dump({"type": "FeatureCollection", "features": features}, f_geo, indent=2)

    logger.info("Saved JSON report    -> %s", json_path)
    logger.info("Saved CSV summary    -> %s", csv_path)
    logger.info("Saved GeoJSON map    -> %s (%d geotagged hazard frames)", geojson_path, len(features))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Annotated stitched 360° panoramic frames directory
    annotated_frames_dir = out_dir / "annotated_frames"
    annotated_frames_dir.mkdir(parents=True, exist_ok=True)

    # 2. Annotated perspective views directory with subfolders: left, right, front, back
    annotated_views_dir = out_dir / "annotated_views"
    if args.save_views:
        annotated_views_dir.mkdir(parents=True, exist_ok=True)
        for vname in ["front", "right", "back", "left"]:
            (annotated_views_dir / vname).mkdir(parents=True, exist_ok=True)

    raw_views_dir = out_dir / "raw_views"
    quad_grids_dir = out_dir / "quad_grids"

    gpx_matcher = GpxTrackMatcher(args.gpx)

    splitter = EquirectangularSplitter(
        out_hw=(args.view_height, args.view_width),
        fov_deg=args.fov,
        pitch_deg=args.pitch,
    )

    import torch
    requested_device = args.device
    if requested_device != "cpu" and not torch.cuda.is_available():
        logger.warning(
            "CUDA device '%s' was requested, but torch.cuda.is_available() is False. Falling back to 'cpu'.",
            requested_device,
        )
        resolved_device = "cpu"
    else:
        resolved_device = "cpu" if requested_device == "cpu" else f"cuda:{requested_device.lstrip('cuda:')}"

    # Auto-resolve weight file aliases
    seg_weights = args.seg_weights
    if seg_weights and not Path(seg_weights).is_file():
        for alt in ["segmentor.pt", "segmenter.pt", "segmentation.pt"]:
            if Path(alt).is_file():
                logger.info("Resolved seg-weights '%s' -> '%s'", seg_weights, alt)
                seg_weights = alt
                break

    det_weights = args.detector_weights
    if det_weights and not Path(det_weights).is_file():
        for alt in ["detector.pt", "yolo_detector.pt"]:
            if Path(alt).is_file():
                logger.info("Resolved detector-weights '%s' -> '%s'", det_weights, alt)
                det_weights = alt
                break

    cls_weights = args.classifier_weights
    if cls_weights and not Path(cls_weights).is_file():
        for alt in ["classifier.pth", "convnext_road_model_v2.pth", "classifier.pt"]:
            if Path(alt).is_file():
                logger.info("Resolved classifier-weights '%s' -> '%s'", cls_weights, alt)
                cls_weights = alt
                break

    cls_names = [c.strip() for c in args.classifier_classes.split(",") if c.strip()]
    classifier = RoadTypeClassifier(
        weights_path=cls_weights,
        class_names=cls_names,
        arch=args.classifier_arch,
        device=resolved_device,
        input_size=args.classifier_res,
    )

    rdd_weights = args.rdd_weights
    if str(rdd_weights).strip().lower() in ("none", "null", "false"):
        rdd_weights = None
    elif not rdd_weights or not Path(rdd_weights).is_file():
        for alt in ["YOLOv8_Small_RDD.pt", "rdd.pt", "rdd_model.pt"]:
            if Path(alt).is_file():
                logger.info("Resolved rdd-weights '%s' -> '%s'", rdd_weights, alt)
                rdd_weights = alt
                break
        else:
            if rdd_weights:
                logger.warning("RDD weights '%s' not found.", rdd_weights)
            rdd_weights = None

    pipeline = VillageStreetHazardPipeline(
        detector_weights=det_weights,
        segmentation_weights=seg_weights,
        rdd_weights=rdd_weights,
        classifier=classifier,
        device=resolved_device,
        imgsz=args.imgsz,
        conf_waste=args.conf_waste,
        conf_drainage=args.conf_drainage,
        conf_road=args.conf_road_damage,
        conf_road_unpaved=args.conf_road_unpaved,
        enable_road_tiling=not args.no_tiling,
        apply_clahe=args.apply_clahe,
    )

    source_path = Path(args.source)
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    reports: List[FrameReport] = []

    if source_path.is_file() and source_path.suffix.lower() in video_exts:
        cap = cv2.VideoCapture(str(source_path))
        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % max(1, args.vid_stride) == 0:
                frame_id = f"{source_path.stem}_frame_{frame_idx:06d}"
                gpx_pt = gpx_matcher.match_frame(frame_idx, total_frames)
                views = splitter.split_frame(frame, mode=args.input_mode)
                report, annotated_views = pipeline.process_frame(frame_id, frame_idx, views, gpx_pt=gpx_pt)
                reports.append(report)

                # Save 2D annotated perspective views in annotated_views/{view_name}/{frame_id}.jpg
                if args.save_views:
                    for v_name, v_ann in annotated_views.items():
                        v_dir = annotated_views_dir / v_name
                        v_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(v_dir / f"{frame_id}.jpg"), v_ann)

                if args.save_raw_views:
                    for v_name, v_raw in views.items():
                        v_dir = raw_views_dir / v_name
                        v_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(v_dir / f"{frame_id}.jpg"), v_raw)

                # Render and save annotated 360° stitched equirectangular frame
                if args.input_mode == "equirectangular":
                    annotated_eq = splitter.render_equirect_annotations(
                        eq_img=frame,
                        road_type=report.road_type,
                        detections=report.detections,
                        is_paved=report.is_paved_surface,
                        frame_id=frame_id,
                        gpx_pt=gpx_pt,
                        class_colors=pipeline.CLASS_COLORS_BGR,
                    )
                else:
                    annotated_eq = pipeline.compose_quad_grid(annotated_views, report)

                ann_filename = f"annotated_{frame_id}.jpg" if not frame_id.startswith("annotated_") else f"{frame_id}.jpg"
                save_annotated_frame_with_exif(
                    bgr_img=annotated_eq,
                    out_path=annotated_frames_dir / ann_filename,
                    exif_bytes=None,
                    lat=report.latitude,
                    lon=report.longitude,
                    ele=report.elevation,
                    iso_ts=report.gpx_time,
                    quality=95,
                )
                report.annotated_path = f"annotated_frames/{ann_filename}"

                if args.save_quad:
                    quad_grids_dir.mkdir(parents=True, exist_ok=True)
                    grid_img = pipeline.compose_quad_grid(annotated_views, report)
                    cv2.imwrite(str(quad_grids_dir / f"{frame_id}_quad.jpg"), grid_img)

                logger.info(
                    "Processed %s (GPS: %s, %s) -> %s",
                    frame_id,
                    report.latitude,
                    report.longitude,
                    report.counts_by_class,
                )
            frame_idx += 1
        cap.release()
    else:
        img_files = (
            [source_path]
            if source_path.is_file()
            else sorted([p for p in source_path.rglob("*") if p.suffix.lower() in image_exts])
        )
        total_imgs = len(img_files)
        logger.info("Found %d 360° frames in '%s'.", total_imgs, source_path)
        for idx, img_path in enumerate(img_files):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            frame_id = img_path.stem
            gpx_pt = gpx_matcher.match_frame(idx, total_imgs)
            views = splitter.split_frame(frame, mode=args.input_mode)
            report, annotated_views = pipeline.process_frame(frame_id, idx, views, gpx_pt=gpx_pt)
            reports.append(report)

            # Save 2D annotated perspective views in annotated_views/{view_name}/{frame_id}.jpg
            if args.save_views:
                for v_name, v_ann in annotated_views.items():
                    v_dir = annotated_views_dir / v_name
                    v_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(v_dir / f"{frame_id}.jpg"), v_ann)

            if args.save_raw_views:
                for v_name, v_raw in views.items():
                    v_dir = raw_views_dir / v_name
                    v_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(v_dir / f"{frame_id}.jpg"), v_raw)

            # Render and save annotated 360° stitched equirectangular frame
            if args.input_mode == "equirectangular":
                annotated_eq = splitter.render_equirect_annotations(
                    eq_img=frame,
                    road_type=report.road_type,
                    detections=report.detections,
                    is_paved=report.is_paved_surface,
                    frame_id=frame_id,
                    gpx_pt=gpx_pt,
                    class_colors=pipeline.CLASS_COLORS_BGR,
                )
            else:
                annotated_eq = pipeline.compose_quad_grid(annotated_views, report)

            ann_filename = f"annotated_{frame_id}.jpg" if not frame_id.startswith("annotated_") else f"{frame_id}.jpg"
            src_exif = None
            try:
                with Image.open(img_path) as s_img:
                    src_exif = s_img.info.get("exif")
            except Exception:
                pass

            save_annotated_frame_with_exif(
                bgr_img=annotated_eq,
                out_path=annotated_frames_dir / ann_filename,
                exif_bytes=src_exif,
                lat=report.latitude,
                lon=report.longitude,
                ele=report.elevation,
                iso_ts=report.gpx_time,
                quality=95,
            )
            report.annotated_path = f"annotated_frames/{ann_filename}"

            if args.save_quad:
                quad_grids_dir.mkdir(parents=True, exist_ok=True)
                grid_img = pipeline.compose_quad_grid(annotated_views, report)
                cv2.imwrite(str(quad_grids_dir / f"{frame_id}_quad.jpg"), grid_img)

            logger.info(
                "Processed [%d/%d] %s (GPS: %s, %s) -> %s",
                idx + 1,
                total_imgs,
                frame_id,
                report.latitude,
                report.longitude,
                report.counts_by_class,
            )

    save_reports(reports, out_dir)


if __name__ == "__main__":
    main()