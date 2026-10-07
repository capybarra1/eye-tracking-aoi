"""Optional, local-only AOI relocation candidates from a reviewed pilot model.

Importing/constructing this module does not import torch or load weights. A
missing dependency is reported as unavailable; inference errors are explicit.
Candidates are never approvals, and a missing tablet prediction is UNKNOWN.
The caller must require clear, stable observations before using any candidate.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from scripts.aoi_detector_geometry import GeometryConfig, mask_to_geometry


class DetectorUnavailable(RuntimeError):
    """The explicitly requested local model or its runtime is unavailable."""


@dataclass(frozen=True)
class DetectorConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    tablet_min_flip_iou_in_quad: float = .80
    tablet_min_flip_iou_global: float = .75
    screen_min_flip_iou: float = .95
    screen_max_flip_boundary_median_px: float = 8.
    screen_max_flip_boundary_p95_px: float = 18.
    tablet_only: bool = False


@dataclass
class _Backend:
    torch: object
    model: object
    device: object
    size: tuple[int, int]
    metadata: dict
    lock: threading.RLock = field(default_factory=threading.RLock)

    def predict_masks(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        rgb = cv2.cvtColor(cv2.resize(frame, self.size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = (rgb - np.float32([.485, .456, .406])) / np.float32([.229, .224, .225])
        array = np.ascontiguousarray(rgb.transpose(2, 0, 1))
        # This lock is shared by every detector instance using these weights.
        with self.lock, torch.inference_mode():
            tensor = torch.from_numpy(array)[None].to(self.device)
            normal = self.model(tensor)['out']
            flipped = self.model(tensor.flip(-1))['out'].flip(-1)
            masks = []
            for logits in (normal, flipped):
                if not bool(torch.isfinite(logits).all().item()):
                    raise RuntimeError('Model returned non-finite logits.')
                probabilities = torch.softmax(logits, dim=1)
                full = torch.nn.functional.interpolate(probabilities, size=frame.shape[:2],
                                                        mode='bilinear', align_corners=False)
                masks.append(full.argmax(1)[0].to('cpu').numpy().astype(np.uint8))
        return masks[0], masks[1]


_CACHE: OrderedDict[tuple, _Backend] = OrderedDict()
_CACHE_LOCK = threading.RLock()
_CACHE_LIMIT = 2


def _load_backend(path: Path, requested_device: str) -> _Backend:
    try:
        torch = importlib.import_module('torch')
        segmentation = importlib.import_module('torchvision.models.segmentation')
    except (ImportError, OSError) as exc:
        raise DetectorUnavailable(f'Optional local torch/torchvision runtime is unavailable: {exc}') from exc
    if requested_device not in ('auto', 'cpu', 'mps'):
        raise ValueError('device must be auto, cpu, or mps')
    can_mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    chosen = 'mps' if requested_device == 'auto' and can_mps else requested_device
    if chosen == 'auto':
        chosen = 'cpu'
    if chosen == 'mps' and not can_mps:
        raise DetectorUnavailable('Requested MPS is not available to this process.')
    device = torch.device(chosen)
    # Both weights arguments must stay None: runtime operation never downloads.
    model = segmentation.lraspp_mobilenet_v3_large(weights=None, weights_backbone=None)
    for name in ('low_classifier', 'high_classifier'):
        previous = getattr(model.classifier, name)
        setattr(model.classifier, name, torch.nn.Conv2d(previous.in_channels, 3, 1))
    checkpoint = torch.load(str(path), map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('manifest'), dict):
        raise ValueError('Checkpoint must contain state_dict and a training manifest.')
    manifest = checkpoint['manifest']
    size = manifest.get('size')
    if (not isinstance(size, (list, tuple)) or len(size) != 2
            or any(not isinstance(value, int) or value < 32 or value > 4096 for value in size)):
        raise ValueError('Checkpoint manifest has an invalid [width, height] input size.')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.to(device).eval()
    metadata = dict(source='local_lraspp_mobilenet_v3', model_scope='development_pilot',
                    checkpoint_path=str(path), checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    training_subjects=list(manifest.get('train_subjects', [])),
                    development_holdout_subjects=list(manifest.get('heldout_subjects', [])),
                    training_data_sha256=manifest.get('source_sha256'),
                    input_size=list(size), device=str(device),
                    candidate_only=True, confidence_is_calibrated=False,
                    class_labels={'0': 'other', '1': 'screen', '2': 'tablet'},
                    requires_clear_temporally_stable_frames=True)
    return _Backend(torch, model, device, tuple(size), metadata)


def _get_backend(path: Path, requested_device: str) -> _Backend:
    try:
        stat = path.stat()
    except OSError as exc:
        raise DetectorUnavailable(f'Explicit local checkpoint cannot be read: {path}: {exc}') from exc
    if not path.is_file():
        raise DetectorUnavailable(f'Checkpoint is not a file: {path}')
    key = (str(path), stat.st_size, stat.st_mtime_ns, requested_device)
    with _CACHE_LOCK:
        if key not in _CACHE:
            backend = _load_backend(path, requested_device)
            # Do not cache a checkpoint replaced while loading.
            current = path.stat()
            if (current.st_size, current.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                raise RuntimeError('Checkpoint changed while loading; retry the new version.')
            _CACHE[key] = backend
            while len(_CACHE) > _CACHE_LIMIT:
                _CACHE.popitem(last=False)
        _CACHE.move_to_end(key)
        return _CACHE[key]


def _iou(first: np.ndarray, second: np.ndarray, roi: np.ndarray | None = None) -> float | None:
    if roi is not None:
        first, second = first & roi, second & roi
    union = np.count_nonzero(first | second)
    return float(np.count_nonzero(first & second) / union) if union else None


def _reject(result: dict, key: str, reason: str) -> None:
    result['polygons'][key] = None
    result['diagnostics'][key].update(status='unknown', reason=reason)


def _gate_masks(normal: np.ndarray, flipped: np.ndarray, config: DetectorConfig) -> dict:
    if normal.shape != flipped.shape:
        raise ValueError('Normal and flipped model masks have different dimensions.')
    if config.tablet_only:
        if not np.issubdtype(flipped.dtype, np.integer) or not np.isin(flipped, [0, 1, 2]).all():
            raise ValueError('Flipped mask labels must be integer values 0=other, 1=screen, 2=tablet.')
        result = mask_to_geometry(normal, config.geometry, tablet_only=True)
        other = None
        targets = [('tablet', 2)]
    else:
        result = mask_to_geometry(normal, config.geometry)
        other = mask_to_geometry(flipped, config.geometry)
        targets = [('screen', 1), ('tablet', 2)]
    for key, label in targets:
        evidence = result['diagnostics'][key]
        evidence['flip_geometry_reason'] = (other['diagnostics'][key]['reason'] if other is not None
                                             else 'not_fitted_mask_consistency_only')
        evidence['flip_iou_global'] = _iou(normal == label, flipped == label)
        evidence['consistency_is_calibrated_probability'] = False
        if result['polygons'][key] is None:
            continue
        # Tablet agreement compares the actual masks; demanding a second quad
        # approximation would discard agreeing masks due only to contour fitting.
        # Screen agreement uses fitted lower boundaries, so both fits are needed.
        if key == 'screen' and other['polygons'][key] is None:
            _reject(result, key, 'flipped_prediction_has_no_supported_geometry')
            continue
        if key == 'tablet':
            roi = np.zeros_like(normal, dtype=np.uint8)
            cv2.fillPoly(roi, [np.rint(result['polygons'][key][0]).astype(np.int32)], 1)
            inside = _iou(normal == 2, flipped == 2, roi.astype(bool))
            evidence['flip_iou_in_quad'] = inside
            if (inside is None or inside < config.tablet_min_flip_iou_in_quad
                    or evidence['flip_iou_global'] is None
                    or evidence['flip_iou_global'] < config.tablet_min_flip_iou_global):
                _reject(result, key, 'tablet_flip_disagreement')
        else:
            points = np.asarray(result['polygons'][key][0])
            other_points = np.asarray(other['polygons'][key][0])
            xs = np.arange(normal.shape[1])
            difference = np.abs(np.interp(xs, points[:, 0], points[:, 1])
                                - np.interp(xs, other_points[:, 0], other_points[:, 1]))
            median, p95 = float(np.median(difference)), float(np.percentile(difference, 95))
            evidence.update(flip_boundary_median_px=median, flip_boundary_p95_px=p95)
            scale = normal.shape[0] / 540.
            if (evidence['flip_iou_global'] is None
                    or evidence['flip_iou_global'] < config.screen_min_flip_iou
                    or median > config.screen_max_flip_boundary_median_px * scale
                    or p95 > config.screen_max_flip_boundary_p95_px * scale):
                _reject(result, key, 'screen_flip_disagreement')
    return result


class LearnedAOIDetector:
    """Thread-safe lazy candidate provider. It has no project or approval access."""

    def __init__(self, checkpoint_path: str | Path, *, config: DetectorConfig | None = None,
                 device: str = 'auto') -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.config = config or DetectorConfig()
        self.device = device

    def predict(self, frame: np.ndarray) -> dict:
        started = time.perf_counter()
        try:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                    or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 16):
                raise ValueError('frame must be a uint8 BGR image, at least 16 pixels per side.')
            backend = _get_backend(self.checkpoint_path, self.device)
            normal, flipped = backend.predict_masks(frame)
            if normal.shape != frame.shape[:2] or flipped.shape != frame.shape[:2]:
                raise ValueError('Model masks do not match the original image dimensions.')
            result = _gate_masks(normal, flipped, self.config)
            result['status'] = 'candidate' if any(v is not None for v in result['polygons'].values()) else 'unknown'
            result['metadata'] = dict(backend.metadata, elapsed_ms=(time.perf_counter()-started)*1000,
                                      flip_consistency_checked=True, candidate_gates=asdict(self.config))
            return result
        except Exception as exc:
            status = 'unavailable' if isinstance(exc, DetectorUnavailable) else 'error'
            return dict(status=status, polygons=dict(screen=None, tablet=None),
                        diagnostics={name: dict(status='unknown', reason=f'detector_{status}',
                                                 confirmed_absent=False) for name in ('screen', 'tablet')},
                        metadata=dict(source='local_lraspp_mobilenet_v3', model_scope='development_pilot',
                                      checkpoint_path=str(self.checkpoint_path), candidate_only=True,
                                      confidence_is_calibrated=False, elapsed_ms=(time.perf_counter()-started)*1000),
                        error=dict(type=type(exc).__name__, message=str(exc)))
