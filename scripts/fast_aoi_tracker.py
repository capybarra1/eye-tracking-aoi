from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("outputs/eye_aoi_validation_29zt/normal_triscreen_sample_25HLY")
DEFAULT_VIDEO = ROOT / "25HLY_normal_triscreen_90_120s.mp4"
DEFAULT_PROMPTS = ROOT / "25HLY_sam2_polygon_prompts.json"
DEFAULT_OUT_DIR = ROOT / "fast_aoi_tracker"

PolygonMap = dict[str, list[np.ndarray]]


@dataclass
class StepStats:
    frame_index: int
    time_s: float
    status: str
    confidence: float
    inliers: int
    tracked_points: int


@dataclass
class RegionTrack:
    aoi: str
    polygon_index: int
    polygon: np.ndarray
    points: np.ndarray | None = None
    status: str = "seed"
    confidence: float = 0.0
    inliers: int = 0
    tracked_points: int = 0


def load_prompt_polygons(prompt_json: Path, output_size: tuple[int, int]) -> PolygonMap:
    data = json.loads(prompt_json.read_text(encoding="utf-8"))
    source_w = float(data["width"])
    source_h = float(data["height"])
    target_w, target_h = output_size
    scale = np.array([target_w / source_w, target_h / source_h], dtype=np.float32)

    grouped: PolygonMap = {"screen": [], "tablet": []}
    for name, spec in data.get("objects", {}).items():
        label = str(spec.get("label") or name)
        if label.startswith("screen") or name.startswith("screen"):
            aoi = "screen"
        elif label == "tablet" or name == "tablet":
            aoi = "tablet"
        else:
            continue
        for polygon in spec.get("polygons", []):
            pts = np.asarray(polygon, dtype=np.float32) * scale
            if len(pts) >= 3:
                grouped[aoi].append(pts)

    return {name: polygons for name, polygons in grouped.items() if polygons}


def enhance_frame_for_detection(frame: np.ndarray, alpha: float = 1.35, beta: float = 26.0) -> np.ndarray:
    bright = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    lab = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_chan)
    enhanced = cv2.merge((enhanced_l, a_chan, b_chan))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def expand_polygon_outward(polygon: np.ndarray, padding_px: float) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32)
    center = pts.mean(axis=0)
    vectors = pts - center
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 1e-6)
    return (pts + directions * float(padding_px)).astype(np.float32)


def bottom_edge_from_polygon(polygon: np.ndarray) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32)
    order = np.argsort(pts[:, 1])[-2:]
    bottom = pts[order]
    return bottom[np.argsort(bottom[:, 0])].astype(np.float32)


def top_edge_from_polygon(polygon: np.ndarray) -> np.ndarray:
    pts = np.asarray(polygon, dtype=np.float32)
    order = np.argsort(pts[:, 1])[:2]
    top = pts[order]
    return top[np.argsort(top[:, 0])].astype(np.float32)


def screen_polygon_from_bottom_edge(seed_polygon: np.ndarray, detected_bottom: np.ndarray) -> np.ndarray:
    top = top_edge_from_polygon(seed_polygon)
    old_bottom = bottom_edge_from_polygon(seed_polygon)
    bottom = np.asarray(detected_bottom, dtype=np.float32)
    bottom = bottom[np.argsort(bottom[:, 0])]
    left_shift = bottom[0] - old_bottom[0]
    right_shift = bottom[1] - old_bottom[1]
    new_top_left = top[0] + left_shift
    new_top_right = top[1] + right_shift
    return np.array([new_top_left, new_top_right, bottom[1], bottom[0]], dtype=np.float32)


def y_on_line_at_x(p1: np.ndarray, p2: np.ndarray, x: float) -> float:
    dx = float(p2[0] - p1[0])
    if abs(dx) < 1e-6:
        return float((p1[1] + p2[1]) / 2)
    alpha = (float(x) - float(p1[0])) / dx
    return float(p1[1] + alpha * (p2[1] - p1[1]))


def detect_screen_lower_edge(frame: np.ndarray, prior_polygon: np.ndarray, output_size: tuple[int, int]) -> tuple[np.ndarray | None, float]:
    width, height = output_size
    prior_bottom = bottom_edge_from_polygon(prior_polygon)
    x_min = max(0, int(np.floor(prior_bottom[:, 0].min() - 42)))
    x_max = min(width - 1, int(np.ceil(prior_bottom[:, 0].max() + 42)))
    y_min = max(0, int(np.floor(prior_bottom[:, 1].min() - 56)))
    y_max = min(height - 1, int(np.ceil(prior_bottom[:, 1].max() + 56)))
    if x_max <= x_min + 20 or y_max <= y_min + 8:
        return None, 0.0

    roi = frame[y_min : y_max + 1, x_min : x_max + 1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=42,
        minLineLength=max(28, int((x_max - x_min) * 0.42)),
        maxLineGap=18,
    )
    if lines is None:
        return None, 0.0

    seed_dx = prior_bottom[1, 0] - prior_bottom[0, 0]
    seed_slope = (prior_bottom[1, 1] - prior_bottom[0, 1]) / max(abs(seed_dx), 1.0)
    candidates = []
    for raw in lines.reshape(-1, 4):
        x1, y1, x2, y2 = raw.astype(np.float32)
        dx = x2 - x1
        if abs(dx) < 18:
            continue
        slope = (y2 - y1) / dx
        if abs(slope - seed_slope) > 0.16:
            continue
        p1 = np.array([x1 + x_min, y1 + y_min], dtype=np.float32)
        p2 = np.array([x2 + x_min, y2 + y_min], dtype=np.float32)
        if p1[0] > p2[0]:
            p1, p2 = p2, p1
        line_center_y = (p1[1] + p2[1]) / 2
        prior_center_y = float(prior_bottom[:, 1].mean())
        y_distance = abs(float(line_center_y) - prior_center_y)
        if y_distance > 58:
            continue
        length = float(np.linalg.norm(p2 - p1))
        score = length - y_distance * 1.7 - abs(float(slope - seed_slope)) * 120
        candidates.append((score, p1, p2, length, y_distance))

    if not candidates:
        return None, 0.0

    score, p1, p2, length, y_distance = max(candidates, key=lambda item: item[0])
    left_x = float(prior_bottom[0, 0])
    right_x = float(prior_bottom[1, 0])
    detected = np.array(
        [
            [left_x, y_on_line_at_x(p1, p2, left_x)],
            [right_x, y_on_line_at_x(p1, p2, right_x)],
        ],
        dtype=np.float32,
    )
    confidence = max(0.0, min(1.0, (length / max(right_x - left_x, 1.0)) * 0.75 + max(0.0, 1.0 - y_distance / 36) * 0.25))
    return detected, float(confidence)


def make_context_mask(output_size: tuple[int, int], polygons: PolygonMap, interior_margin: int = 18) -> np.ndarray:
    width, height = output_size
    mask = np.full((height, width), 255, dtype=np.uint8)
    interiors = np.zeros((height, width), dtype=np.uint8)
    for polygon_list in polygons.values():
        for polygon in polygon_list:
            cv2.fillPoly(interiors, [polygon.astype(np.int32)], 255)

    if interior_margin > 0:
        kernel_size = max(3, interior_margin * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        interiors = cv2.erode(interiors, kernel, iterations=1)
    mask[interiors > 0] = 0
    return mask


def make_local_tracking_mask(
    output_size: tuple[int, int],
    polygon: np.ndarray,
    outer_margin: int = 42,
    inner_margin: int = 10,
) -> np.ndarray:
    vertices = tuple(tuple(int(value) for value in point) for point in polygon.astype(np.int32))
    return _cached_local_tracking_mask(output_size, vertices, outer_margin, inner_margin).copy()


@lru_cache(maxsize=32)
def _cached_local_tracking_mask(
    output_size: tuple[int, int],
    vertices: tuple[tuple[int, int], ...],
    outer_margin: int,
    inner_margin: int,
) -> np.ndarray:
    width, height = output_size
    base = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(base, [np.asarray(vertices, dtype=np.int32)], 255)

    outer_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_margin * 2 + 1, outer_margin * 2 + 1))
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_margin * 2 + 1, inner_margin * 2 + 1))
    outer = cv2.dilate(base, outer_kernel, iterations=1)
    inner = cv2.erode(base, inner_kernel, iterations=1)
    ring = cv2.subtract(outer, inner)
    return ring


def apply_homography(polygons: PolygonMap, H: np.ndarray) -> PolygonMap:
    moved: PolygonMap = {}
    for aoi, polygon_list in polygons.items():
        moved[aoi] = []
        for polygon in polygon_list:
            transformed = cv2.perspectiveTransform(polygon.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            moved[aoi].append(transformed.astype(np.float32))
    return moved


def transform_polygon(polygon: np.ndarray, H: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(polygon.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2).astype(np.float32)


def flatten_polygons(polygons: PolygonMap) -> list[RegionTrack]:
    regions: list[RegionTrack] = []
    for aoi, polygon_list in polygons.items():
        for polygon_index, polygon in enumerate(polygon_list):
            regions.append(RegionTrack(aoi=aoi, polygon_index=polygon_index, polygon=polygon.copy()))
    return regions


def regions_to_polygons(regions: list[RegionTrack]) -> PolygonMap:
    grouped: PolygonMap = {}
    for region in regions:
        grouped.setdefault(region.aoi, [])
        while len(grouped[region.aoi]) <= region.polygon_index:
            grouped[region.aoi].append(np.empty((0, 2), dtype=np.float32))
        grouped[region.aoi][region.polygon_index] = region.polygon.copy()
    return grouped


def apply_region_transforms(regions: list[RegionTrack], transforms: dict[tuple[str, int], np.ndarray]) -> PolygonMap:
    moved_regions: list[RegionTrack] = []
    identity = np.eye(3, dtype=np.float32)
    for region in regions:
        H = transforms.get((region.aoi, region.polygon_index), identity)
        moved_regions.append(
            RegionTrack(
                aoi=region.aoi,
                polygon_index=region.polygon_index,
                polygon=transform_polygon(region.polygon, H),
            )
        )
    return regions_to_polygons(moved_regions)


def prepare_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def detect_features(gray: np.ndarray, mask: np.ndarray, max_corners: int = 900) -> np.ndarray | None:
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.008,
        minDistance=7,
        blockSize=7,
        mask=mask,
    )
    if points is None or len(points) < 20:
        return None
    return points.astype(np.float32)


def is_reasonable_homography(H: np.ndarray, output_size: tuple[int, int]) -> bool:
    width, height = output_size
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    moved = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H).reshape(-1, 2)
    if not np.isfinite(moved).all():
        return False

    original_area = max(cv2.contourArea(corners), 1.0)
    moved_area = abs(cv2.contourArea(moved))
    area_ratio = moved_area / original_area
    mean_shift = float(np.mean(np.linalg.norm(moved - corners, axis=1)))
    return 0.65 <= area_ratio <= 1.55 and mean_shift <= max(width, height) * 0.22


def is_reasonable_region_transform(H: np.ndarray, polygon: np.ndarray, output_size: tuple[int, int]) -> bool:
    moved = transform_polygon(polygon, H)
    if not np.isfinite(moved).all():
        return False

    source_area = max(abs(float(cv2.contourArea(polygon.astype(np.float32)))), 1.0)
    moved_area = abs(float(cv2.contourArea(moved.astype(np.float32))))
    area_ratio = moved_area / source_area
    mean_shift = float(np.mean(np.linalg.norm(moved - polygon, axis=1)))
    return 0.72 <= area_ratio <= 1.38 and mean_shift <= max(output_size) * 0.16


def estimate_translation_from_matches(
    old: np.ndarray,
    new: np.ndarray,
    max_error: float = 5.0,
) -> tuple[np.ndarray, float, int]:
    identity = np.eye(3, dtype=np.float32)
    if len(old) == 0 or len(new) == 0:
        return identity, 0.0, 0

    displacements = new.astype(np.float32) - old.astype(np.float32)
    median_shift = np.median(displacements, axis=0)
    errors = np.linalg.norm(displacements - median_shift, axis=1)
    inlier_mask = errors <= max_error
    inliers = int(inlier_mask.sum())
    if inliers > 0:
        median_shift = np.median(displacements[inlier_mask], axis=0)

    H = np.eye(3, dtype=np.float32)
    H[0, 2] = float(median_shift[0])
    H[1, 2] = float(median_shift[1])
    return H, inliers / max(len(old), 1), inliers


def estimate_homography(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_points: np.ndarray | None,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None, str, float, int, int]:
    identity = np.eye(3, dtype=np.float32)
    if prev_points is None or len(prev_points) < 20:
        return identity, None, "few_features", 0.0, 0, 0

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        return identity, None, "flow_failed", 0.0, 0, int(len(prev_points))

    valid = status.reshape(-1) == 1
    old = prev_points[valid].reshape(-1, 2)
    new = next_points[valid].reshape(-1, 2)
    tracked = int(len(old))
    if tracked < 20:
        return identity, None, "few_tracked", 0.0, 0, tracked

    H, inlier_mask = cv2.findHomography(old, new, cv2.RANSAC, 4.0)
    if H is None or inlier_mask is None:
        return identity, None, "homography_failed", 0.0, 0, tracked

    inliers = int(inlier_mask.sum())
    confidence = inliers / max(tracked, 1)
    if inliers < 18 or confidence < 0.28 or not is_reasonable_homography(H.astype(np.float32), output_size):
        return identity, None, "rejected", float(confidence), inliers, tracked

    kept_points = new[inlier_mask.reshape(-1) == 1].reshape(-1, 1, 2).astype(np.float32)
    return H.astype(np.float32), kept_points, "tracked", float(confidence), inliers, tracked


def estimate_region_motion(
    prev_gray: np.ndarray,
    gray: np.ndarray,
    prev_points: np.ndarray | None,
    polygon: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None, str, float, int, int]:
    identity = np.eye(3, dtype=np.float32)
    if prev_points is None or len(prev_points) < 12:
        return identity, None, "few_features", 0.0, 0, 0

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        gray,
        prev_points,
        None,
        winSize=(25, 25),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
    )
    if next_points is None or status is None:
        return identity, None, "flow_failed", 0.0, 0, int(len(prev_points))

    valid = status.reshape(-1) == 1
    old = prev_points[valid].reshape(-1, 2)
    new = next_points[valid].reshape(-1, 2)
    tracked = int(len(old))
    if tracked < 12:
        return identity, None, "few_tracked", 0.0, 0, tracked

    H, confidence, inliers = estimate_translation_from_matches(old, new, max_error=5.0)
    if inliers < 10 or confidence < 0.25 or not is_reasonable_region_transform(H, polygon, output_size):
        return identity, None, "rejected", float(confidence), inliers, tracked

    shift = H[:2, 2]
    errors = np.linalg.norm((new - old) - shift, axis=1)
    kept_points = new[errors <= 5.0].reshape(-1, 1, 2).astype(np.float32)
    return H, kept_points, "tracked", float(confidence), inliers, tracked


def reseed_region(gray: np.ndarray, region: RegionTrack, output_size: tuple[int, int]) -> None:
    if region.aoi == "screen":
        mask = make_local_tracking_mask(output_size, region.polygon, outer_margin=34, inner_margin=22)
        max_corners = 260
    else:
        mask = make_local_tracking_mask(output_size, region.polygon, outer_margin=48, inner_margin=8)
        max_corners = 240
    region.points = detect_features(gray, mask, max_corners=max_corners)
    if region.points is not None:
        region.tracked_points = int(len(region.points))


def summarize_regions(frame_index: int, time_s: float, regions: list[RegionTrack]) -> StepStats:
    tracked = [region for region in regions if region.status in {"seed", "tracked"}]
    status = "tracked" if len(tracked) == len(regions) else "partial"
    if all(region.status == "seed" for region in regions):
        status = "seed"
    confidence = float(np.mean([region.confidence for region in tracked])) if tracked else 0.0
    inliers = int(sum(region.inliers for region in regions))
    tracked_points = int(sum(0 if region.points is None else len(region.points) for region in regions))
    return StepStats(frame_index, time_s, status, confidence, inliers, tracked_points)


def clone_polygons(polygons: PolygonMap) -> PolygonMap:
    return {aoi: [polygon.copy() for polygon in polygon_list] for aoi, polygon_list in polygons.items()}


def draw_polygons(frame: np.ndarray, polygons: PolygonMap, stats: StepStats) -> np.ndarray:
    out = frame.copy()
    colors = {"screen": (0, 255, 0), "tablet": (255, 255, 0)}
    labels = {"screen": "screen", "tablet": "tablet"}

    for aoi in ("screen", "tablet"):
        for polygon in polygons.get(aoi, []):
            color = colors[aoi]
            pts = polygon.astype(np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.28, out, 0.72, 0, dst=out)
            cv2.polylines(out, [pts], True, color, 3)
        if polygons.get(aoi):
            first = polygons[aoi][0][0].astype(int)
            cv2.putText(out, labels[aoi], tuple(first + np.array([6, -10])), cv2.FONT_HERSHEY_SIMPLEX, 0.65, colors[aoi], 2)

    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    text = (
        f"Fast AOI | t={stats.time_s:.2f}s | {stats.status} | "
        f"conf={stats.confidence:.2f} | pts={stats.tracked_points}"
    )
    cv2.putText(out, text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return out


def make_contact(images: list[np.ndarray], columns: int = 2) -> np.ndarray | None:
    if not images:
        return None
    thumbs = [cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA) for image in images]
    blank = np.zeros_like(thumbs[0])
    rows = []
    for start in range(0, len(thumbs), columns):
        row = thumbs[start : start + columns]
        while len(row) < columns:
            row.append(blank.copy())
        rows.append(np.hstack(row))
    return np.vstack(rows)


def append_polygon_rows(rows: list[dict[str, int | float | str]], frame_index: int, time_s: float, polygons: PolygonMap) -> None:
    for aoi, polygon_list in polygons.items():
        for polygon_index, polygon in enumerate(polygon_list):
            for point_index, (x, y) in enumerate(polygon, start=1):
                rows.append(
                    {
                        "frame_index": frame_index,
                        "time_s": round(time_s, 4),
                        "aoi": aoi,
                        "polygon_index": polygon_index,
                        "point_index": point_index,
                        "x": round(float(x), 3),
                        "y": round(float(y), 3),
                    }
                )


def serialize_polygons(polygons: PolygonMap) -> dict[str, list[list[list[float]]]]:
    return {
        aoi: [
            [[round(float(x), 3), round(float(y), 3)] for x, y in polygon]
            for polygon in polygon_list
        ]
        for aoi, polygon_list in polygons.items()
    }


def run_tracker(
    video_path: Path,
    prompt_json: Path,
    out_dir: Path,
    work_width: int = 960,
    track_fps: float = 5.0,
    max_sec: float | None = None,
    review_every_sec: float = 5.0,
    tablet_padding_px: float = 28.0,
    brightness_alpha: float = 1.35,
    brightness_beta: float = 26.0,
) -> dict[str, Path | float | int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    work_height = int(round(source_h * work_width / max(source_w, 1)))
    output_size = (work_width, work_height)
    frame_step = max(1, int(round(source_fps / track_fps)))
    effective_fps = source_fps / frame_step
    end_frame = total_frames if max_sec is None else min(total_frames, int(round(max_sec * source_fps)))

    seed_polygons = load_prompt_polygons(prompt_json, output_size=output_size)
    if "screen" not in seed_polygons or "tablet" not in seed_polygons:
        raise RuntimeError("Prompt file must include screen and tablet AOIs")
    screen_seed_polygons = [polygon.copy() for polygon in seed_polygons["screen"]]
    screen_polygons = [polygon.copy() for polygon in screen_seed_polygons]
    tablet_polygons = {
        "tablet": [expand_polygon_outward(polygon, tablet_padding_px) for polygon in seed_polygons["tablet"]],
    }
    tablet_regions = flatten_polygons(tablet_polygons)

    review_video = out_dir / "fast_aoi_review.mp4"
    writer = cv2.VideoWriter(str(review_video), cv2.VideoWriter_fourcc(*"mp4v"), effective_fps, output_size)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write review video: {review_video}")

    polygon_rows: list[dict[str, int | float | str]] = []
    frame_records: list[dict[str, object]] = []
    contact_images: list[np.ndarray] = []
    prev_gray: np.ndarray | None = None
    processed = 0
    start_time = time.perf_counter()

    for frame_index in range(end_frame):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % frame_step != 0:
            continue

        resized = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
        enhanced = enhance_frame_for_detection(resized, alpha=brightness_alpha, beta=brightness_beta)
        gray = prepare_gray(enhanced)
        time_s = frame_index / source_fps

        if prev_gray is None:
            for region in tablet_regions:
                region.status = "seed"
                region.confidence = 1.0
                region.inliers = 0
                reseed_region(gray, region, output_size)
            stats = summarize_regions(frame_index, time_s, tablet_regions)
        else:
            screen_edge_records: list[dict[str, int | float | str]] = []
            updated_screen_polygons: list[np.ndarray] = []
            for polygon_index, screen_seed_polygon in enumerate(screen_seed_polygons):
                edge, edge_confidence = detect_screen_lower_edge(enhanced, screen_seed_polygon, output_size)
                if edge is None:
                    updated_screen_polygons.append(screen_polygons[polygon_index])
                    screen_edge_records.append(
                        {
                            "aoi": "screen",
                            "polygon_index": polygon_index,
                            "status": "held",
                            "confidence": 0.0,
                            "inliers": 0,
                            "tracked_points": 0,
                        }
                    )
                else:
                    updated = screen_polygon_from_bottom_edge(screen_seed_polygon, edge)
                    updated_screen_polygons.append(updated)
                    screen_edge_records.append(
                        {
                            "aoi": "screen",
                            "polygon_index": polygon_index,
                            "status": "edge",
                            "confidence": round(edge_confidence, 4),
                            "inliers": 1,
                            "tracked_points": 0,
                        }
                    )
            screen_polygons = updated_screen_polygons

            for region in tablet_regions:
                H, tracked_points, status, confidence, inliers, tracked = estimate_region_motion(
                    prev_gray,
                    gray,
                    region.points,
                    region.polygon,
                    output_size,
                )
                if status == "tracked":
                    region.polygon = transform_polygon(region.polygon, H)
                    region.points = tracked_points
                region.status = status
                region.confidence = confidence
                region.inliers = inliers
                region.tracked_points = tracked

                need_reseed = region.points is None or len(region.points) < 45 or processed % 18 == 0 or status != "tracked"
                if need_reseed:
                    reseed_region(gray, region, output_size)
            tablet_stats = summarize_regions(frame_index, time_s, tablet_regions)
            edge_confidences = [float(record["confidence"]) for record in screen_edge_records if record["status"] == "edge"]
            edge_hits = len(edge_confidences)
            confidence_values = edge_confidences + ([tablet_stats.confidence] if tablet_regions else [])
            status = "edge_tracked" if edge_hits == len(screen_polygons) and tablet_stats.status == "tracked" else "partial"
            stats = StepStats(
                frame_index=frame_index,
                time_s=time_s,
                status=status,
                confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
                inliers=edge_hits + tablet_stats.inliers,
                tracked_points=tablet_stats.tracked_points,
            )

        tablet_current = regions_to_polygons(tablet_regions)
        current_polygons = {
            "screen": [polygon.copy() for polygon in screen_polygons],
            "tablet": tablet_current.get("tablet", []),
        }
        append_polygon_rows(polygon_rows, frame_index, time_s, current_polygons)
        if prev_gray is None:
            screen_edge_records = [
                {
                    "aoi": "screen",
                    "polygon_index": i,
                    "status": "seed",
                    "confidence": 1.0,
                    "inliers": 0,
                    "tracked_points": 0,
                }
                for i, _ in enumerate(screen_polygons)
            ]
        frame_records.append(
            {
                "frame_index": frame_index,
                "time_s": round(time_s, 4),
                "status": stats.status,
                "confidence": round(stats.confidence, 4),
                "inliers": stats.inliers,
                "tracked_points": stats.tracked_points,
                "regions": screen_edge_records
                + [
                    {
                        "aoi": region.aoi,
                        "polygon_index": region.polygon_index,
                        "status": region.status,
                        "confidence": round(region.confidence, 4),
                        "inliers": region.inliers,
                        "tracked_points": 0 if region.points is None else int(len(region.points)),
                    }
                    for region in tablet_regions
                ],
                "aois": serialize_polygons(current_polygons),
            }
        )

        overlay = draw_polygons(resized, current_polygons, stats)
        writer.write(overlay)
        if processed == 0 or time_s >= len(contact_images) * review_every_sec:
            contact_images.append(overlay.copy())

        prev_gray = gray
        processed += 1

    cap.release()
    writer.release()

    polygons_csv = out_dir / "fast_aoi_polygons_long.csv"
    with polygons_csv.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["frame_index", "time_s", "aoi", "polygon_index", "point_index", "x", "y"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(polygon_rows)

    polygons_json = out_dir / "fast_aoi_polygons.json"
    polygons_json.write_text(json.dumps({"frames": frame_records}, ensure_ascii=False, indent=2), encoding="utf-8")

    contact_path = out_dir / "fast_aoi_contact.jpg"
    contact = make_contact(contact_images)
    if contact is not None:
        cv2.imwrite(str(contact_path), contact)

    elapsed = time.perf_counter() - start_time
    meta = {
        "video": str(video_path),
        "prompt_json": str(prompt_json),
        "source_fps": source_fps,
        "source_size": [source_w, source_h],
        "work_size": [work_width, work_height],
        "track_fps": effective_fps,
        "frame_step": frame_step,
        "screen_rule": "detect each tri-screen lower edge after brightness enhancement; AOI is the seeded screen polygon rebuilt upward from that lower edge",
        "tablet_padding_px": tablet_padding_px,
        "brightness_alpha": brightness_alpha,
        "brightness_beta": brightness_beta,
        "processed_frames": processed,
        "elapsed_sec": round(elapsed, 3),
        "review_video": str(review_video.resolve()),
        "contact": str(contact_path.resolve()),
        "polygons_csv": str(polygons_csv.resolve()),
        "polygons_json": str(polygons_json.resolve()),
    }
    meta_path = out_dir / "fast_aoi_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "review_video": review_video.resolve(),
        "contact": contact_path.resolve(),
        "polygons_csv": polygons_csv.resolve(),
        "polygons_json": polygons_json.resolve(),
        "meta": meta_path.resolve(),
        "processed_frames": processed,
        "elapsed_sec": elapsed,
        "track_fps": effective_fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast rough AOI tracking from manually seeded polygons.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--work-width", type=int, default=960)
    parser.add_argument("--track-fps", type=float, default=5.0)
    parser.add_argument("--max-sec", type=float, default=None)
    parser.add_argument("--tablet-padding", type=float, default=28.0)
    parser.add_argument("--brightness-alpha", type=float, default=1.35)
    parser.add_argument("--brightness-beta", type=float, default=26.0)
    args = parser.parse_args()

    result = run_tracker(
        video_path=args.video,
        prompt_json=args.prompts,
        out_dir=args.out_dir,
        work_width=args.work_width,
        track_fps=args.track_fps,
        max_sec=args.max_sec,
        tablet_padding_px=args.tablet_padding,
        brightness_alpha=args.brightness_alpha,
        brightness_beta=args.brightness_beta,
    )
    for key, value in result.items():
        print(key, value)


if __name__ == "__main__":
    main()
