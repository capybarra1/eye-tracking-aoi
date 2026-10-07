"""Explicit local configuration for candidate-only tablet recovery.

Call at startup/configuration changes. Repeated calls reuse a checked instance;
status payloads never need to load a model. No network or annotation writes.
"""
from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import threading

import numpy as np

from scripts.aoi_detector_geometry import GeometryConfig
from scripts.aoi_learned_detector import DetectorConfig, LearnedAOIDetector


_CACHE: OrderedDict[tuple, tuple[object | None, dict]] = OrderedDict()
_LOCK = threading.RLock()
_GEOMETRY = dict(tablet_min_area_fraction=.01, tablet_min_solidity=.90,
                 tablet_min_polygon_iou=.85, tablet_min_component_fraction=.90)


class _TabletOnlyDetector:
    def __init__(self, detector: LearnedAOIDetector, metadata: dict) -> None:
        self.detector = detector
        self.metadata = metadata
        self.checkpoint = Path(metadata['checkpoint_path'])
        stat = self.checkpoint.stat()
        self.signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)

    def predict(self, frame: np.ndarray) -> dict:
        try:
            stat = self.checkpoint.stat()
            if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino) != self.signature:
                raise ValueError('本地模型已变化，请重新加载平板自动找回配置')
        except (OSError, ValueError) as exc:
            return dict(status='error', polygons=dict(screen=None, tablet=None),
                        diagnostics={key: dict(status='unknown', reason='configured_model_changed',
                                               confirmed_absent=False) for key in ('screen', 'tablet')},
                        metadata=dict(self.metadata, candidate_only=True, allowed_targets=['tablet']),
                        error=dict(type=type(exc).__name__, message=str(exc)))
        result = copy.deepcopy(self.detector.predict(frame))
        result['polygons']['screen'] = None
        result['diagnostics']['screen'] = dict(status='unknown', reason='traditional_screen_tracking_only')
        if result['status'] in ('candidate', 'unknown'):
            result['status'] = 'candidate' if result['polygons'].get('tablet') is not None else 'unknown'
        result.setdefault('metadata', {}).update(self.metadata)
        result['metadata'].update(candidate_only=True, allowed_targets=['tablet'],
                                  screen_policy='traditional_tracker_only', model_scope='development_pilot')
        return result


def _status(path: Path, *, enabled: bool = False, message: str, error: str | None = None,
            **metadata) -> dict:
    return dict(enabled=enabled, error=error, message=message, candidate_only=True,
                model_scope='development_pilot', allowed_targets=['tablet'],
                config_path=str(path), **metadata)


def _failure(path: Path, message: str, **metadata) -> tuple[None, dict]:
    return None, _status(path, message=message, error=message, **metadata)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def configure_detector(config_path: str | Path) -> tuple[object | None, dict]:
    """Return ``(tablet_only_detector_or_none, status)``; all failures are visible.

    JSON: enabled(bool), checkpoint(local path relative to this JSON),
    device(auto/cpu/mps), optional checkpoint_sha256 and geometry(tablet fields).
    An enabled configuration is prewarmed once per configuration/weight version.
    Failed versions are also cached; restart or edit the configuration to retry.
    """
    path = Path(config_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None, _status(path, message='平板自动找回未配置，使用原跟踪。')
    except OSError as exc:
        return _failure(path, '无法读取平板自动找回配置，已使用原跟踪。', detail=str(exc))
    config_hash = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
        if not isinstance(value, dict) or not isinstance(value.get('enabled', False), bool):
            raise ValueError('enabled 必须是 true 或 false')
        if not value.get('enabled', False):
            return None, _status(path, message='平板自动找回已关闭，使用原跟踪。', config_sha256=config_hash)
        unknown = set(value) - {'enabled', 'checkpoint', 'device', 'checkpoint_sha256', 'geometry'}
        if unknown:
            raise ValueError('不支持的配置项：' + '、'.join(sorted(unknown)))
        checkpoint_value = value.get('checkpoint')
        if not isinstance(checkpoint_value, str) or not checkpoint_value.strip() or '://' in checkpoint_value:
            raise ValueError('checkpoint 必须填写本地模型文件路径，不能填写网址')
        checkpoint = Path(checkpoint_value).expanduser()
        checkpoint = (path.parent / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint.resolve()
        device = value.get('device', 'auto')
        if device not in ('auto', 'cpu', 'mps'):
            raise ValueError('device 只能是 auto、cpu 或 mps')
        expected_hash = value.get('checkpoint_sha256')
        if expected_hash is not None and (not isinstance(expected_hash, str) or len(expected_hash) != 64
                                         or any(c not in '0123456789abcdefABCDEF' for c in expected_hash)):
            raise ValueError('checkpoint_sha256 必须是 64 位十六进制校验值')
        geometry = value.get('geometry', {})
        if not isinstance(geometry, dict) or set(geometry) - set(_GEOMETRY):
            raise ValueError('geometry 仅支持四个平板几何参数')
        geometry = dict(_GEOMETRY, **geometry)
        for key, number in geometry.items():
            if (isinstance(number, bool) or not isinstance(number, (float, int))
                    or not math.isfinite(number) or not 0 < number <= 1):
                raise ValueError(f'{key} 必须大于 0 且不超过 1')
    except (ValueError, TypeError, UnicodeError) as exc:
        return _failure(path, f'平板自动找回配置无效：{exc}。已使用原跟踪。', config_sha256=config_hash)
    try:
        stat = checkpoint.stat()
        if not checkpoint.is_file():
            raise OSError('不是文件')
    except OSError as exc:
        return _failure(path, '找不到本地模型文件，平板自动找回未启用。',
                        checkpoint_path=str(checkpoint), config_sha256=config_hash, detail=str(exc))
    key = (str(path), config_hash, str(checkpoint), stat.st_size, stat.st_mtime_ns,
           stat.st_ctime_ns, stat.st_ino)
    with _LOCK:
        if key in _CACHE:
            detector, status = _CACHE[key]
            _CACHE.move_to_end(key)
            return detector, copy.deepcopy(status)
        metadata = dict(config_sha256=config_hash, checkpoint_path=str(checkpoint), requested_device=device)
        try:
            digest = _sha256(checkpoint)
            metadata['checkpoint_sha256'] = digest
            if expected_hash is not None and digest != expected_hash.lower():
                raise ValueError('模型文件校验不符')
            detector = LearnedAOIDetector(checkpoint, config=DetectorConfig(geometry=GeometryConfig(**geometry), tablet_only=True), device=device)
            warm = detector.predict(np.zeros((540, 960, 3), np.uint8))
            if warm.get('status') not in ('candidate', 'unknown'):
                detail = warm.get('error', {})
                text = detail.get('message', '') if isinstance(detail, dict) else str(detail)
                if warm.get('status') == 'unavailable':
                    reason = '本地模型运行依赖或指定计算设备不可用'
                else:
                    reason = '模型预热检查失败'
                result = _failure(path, f'{reason}，平板自动找回未启用；使用原跟踪。',
                                  **metadata, detail=text)
            else:
                model_metadata = warm.get('metadata', {})
                if model_metadata.get('checkpoint_sha256') != digest:
                    raise ValueError('预热模型与配置中的文件校验不一致')
                loaded = checkpoint.stat()
                if (loaded.st_size, loaded.st_mtime_ns, loaded.st_ctime_ns, loaded.st_ino) != key[3:]:
                    raise ValueError('预热期间模型文件发生变化')
                wrapper = _TabletOnlyDetector(detector, dict(metadata, allowed_targets=['tablet']))
                actual_device = model_metadata.get('device', device)
                acceleration = 'GPU 加速' if actual_device == 'mps' else 'CPU 运行'
                status = _status(path, enabled=True, message=f'{acceleration} · 平板自动找回（试运行）。',
                                 **metadata, device=actual_device, geometry=geometry,
                                 training_subjects=model_metadata.get('training_subjects', []),
                                 development_holdout_subjects=model_metadata.get('development_holdout_subjects', []),
                                 warmup_elapsed_ms=model_metadata.get('elapsed_ms'), ready=True)
                result = (wrapper, status)
        except Exception as exc:
            result = _failure(path, f'平板自动找回启动失败：{exc}。已使用原跟踪。', **metadata)
        _CACHE[key] = result
        while len(_CACHE) > 4:
            _CACHE.popitem(last=False)
        return result[0], copy.deepcopy(result[1])
