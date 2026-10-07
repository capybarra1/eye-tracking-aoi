"""Conservative two-clear-frame gate for optional learned AOI rescue.

No model dependency, annotation storage, or visibility inference lives here.
The caller provides traditional tracking status and applies accepted proposals.
"""
from __future__ import annotations

import copy
from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading
import cv2
import numpy as np


_PREFETCH_SLOT = threading.BoundedSemaphore(1)
_PREFETCH_EXECUTOR = None
_PREFETCH_LOCK = threading.Lock()


def _submit_prefetch(detector, frame):
    """One shared worker, no waiting queue and no per-Tracker threads."""
    global _PREFETCH_EXECUTOR
    if not _PREFETCH_SLOT.acquire(blocking=False):
        return None
    try:
        with _PREFETCH_LOCK:
            if _PREFETCH_EXECUTOR is None:
                _PREFETCH_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='aoi-detector')
        future = _PREFETCH_EXECUTOR.submit(detector.predict, frame.copy())
        future.add_done_callback(lambda completed: _PREFETCH_SLOT.release())
        return future
    except Exception:
        _PREFETCH_SLOT.release()
        raise


class DetectorRescue:
    def __init__(self, detector, size, *, parallel=False):
        self.detector = detector
        self.size = size
        self.pending = dict(screen=None, tablet=None)
        self.cooldown = dict(screen=0, tablet=0)
        self.attempts = 0
        self.last_result = None
        self.tablet_requires_redetection = False
        self.redetect_interval = 6
        self.pending_complete = False
        self.accepted_complete = False
        self.parallel = bool(parallel)
        self._future = None
        self._prefetch_frame = None
        self._prediction_epoch = 0
        self._closed = False
        self.prefetch_stats = dict(submitted=0, consumed=0, discarded=0, synchronous=0)

    def discard_prefetch(self):
        self._prediction_epoch += 1
        future, self._future = self._future, None
        self._prefetch_frame = None
        if future is not None:
            future.cancel()  # Running inference finishes in the single bounded worker.
            self.prefetch_stats['discarded'] += 1

    def close(self):
        self._closed = True
        self.interrupt()

    def prefetch(self, frame, needed: dict) -> bool:
        """Speculate without advancing cooldown or confirmation state."""
        if (not self.parallel or self._closed or not needed.get('tablet', False)
                or (self.pending['tablet'] is None and self.cooldown['tablet'] != 0)):
            return False
        if self._future is not None:
            if self._prefetch_frame is frame:
                return True
            self.discard_prefetch()
        try:
            future = _submit_prefetch(self.detector, frame)
        except Exception:
            return False  # Preserve the established synchronous error handling.
        if future is None:
            return False
        self._future = future
        self._prefetch_frame = frame
        self.prefetch_stats['submitted'] += 1
        return True

    def _predict(self, frame):
        if self._future is not None and self._prefetch_frame is frame:
            future, self._future = self._future, None
            self._prefetch_frame = None
            epoch = self._prediction_epoch
            result = future.result()
            if self._closed or epoch != self._prediction_epoch:
                raise CancelledError('Prediction invalidated by pause/reset.')
            self.prefetch_stats['consumed'] += 1
            return result
        self.discard_prefetch()
        self.prefetch_stats['synchronous'] += 1
        return self.detector.predict(frame)

    def interrupt(self):
        self.discard_prefetch()
        for key in self.pending:
            self.pending[key] = None
            self.cooldown[key] = 0
        self.pending_complete = False

    def clear_tablet(self):
        """Explicit hidden/manual initialization discards partial-recovery state."""
        self.interrupt()
        self.tablet_requires_redetection = False
        self.accepted_complete = False

    def tablet_applied(self):
        self.tablet_requires_redetection = not self.accepted_complete
        if self.tablet_requires_redetection:
            self.cooldown['tablet'] = self.redetect_interval

    def _complete_tablet(self, result, points):
        evidence = result.get('diagnostics', {}).get('tablet', {})
        w, h = self.size
        return (evidence.get('partial') is False
                and evidence.get('out_of_frame_unknown') is False
                and bool(np.all(points > [1, 1]) and np.all(points < [w-2, h-2])))

    def _geometry(self, key, values):
        try:
            p = np.asarray(values, np.float32)
        except (TypeError, ValueError):
            return None
        if p.shape != (1, 4, 2) or not np.isfinite(p).all():
            return None
        p = p[0]
        w, h = self.size
        if key == 'screen':
            if (np.any(np.diff(p[:, 0]) < 2) or abs(p[0, 0]) > 1
                    or abs(p[-1, 0]-(w-1)) > 1 or np.max(np.abs(p[:, 1])) > 2*h):
                return None
        else:
            if not cv2.isContourConvex(p) or abs(cv2.contourArea(p)) < 4:
                return None
            frame = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
            if cv2.intersectConvexConvex(p, frame)[0] <= 0:
                return None
        return p

    def _consistent(self, key, first, second):
        w, h = self.size
        if key == 'screen':
            xs = np.linspace(0,w-1,65)
            delta = np.abs(np.interp(xs,first[:,0],first[:,1])
                           - np.interp(xs,second[:,0],second[:,1]))
            return float(delta.max()) <= 18*h/540
        intersection = cv2.intersectConvexConvex(first,second)[0]
        union = abs(cv2.contourArea(first))+abs(cv2.contourArea(second))-intersection
        center_delta = (np.mean(first,axis=0)-np.mean(second,axis=0))*np.float32([960/w,540/h])
        return union > 0 and intersection/union >= .8 and np.linalg.norm(center_delta) <= 20

    def observe(self, frame, needed: dict) -> dict:
        """Call once per clear frame. Only failed, enabled AOIs may be accepted."""
        if self._closed:
            return {}
        # Screen rescue worsened a real trajectory; retain its traditional path.
        needed = dict(screen=False, tablet=needed.get('tablet', False))
        due = []
        for key in self.pending:
            if not needed.get(key,False):
                self.pending[key] = None
                self.cooldown[key] = 0
                if key == 'tablet':self.pending_complete = False
            elif self.pending[key] is not None or self.cooldown[key] == 0:
                due.append(key)
            else:
                self.cooldown[key] -= 1
        if not due:
            self.discard_prefetch()
            return {}
        self.attempts += 1
        try:
            result = self._predict(frame)
            if not isinstance(result,dict):
                raise ValueError('detector result must be a dictionary')
        except Exception as exc:
            result = dict(status='error',polygons={},error=dict(type=type(exc).__name__,message=str(exc)))
        self.last_result = result
        accepted = {}
        if result.get('status') != 'candidate':
            # Unknown and runtime errors break consecutiveness, never mean absent.
            for key in self.pending:
                self.pending[key] = None
                if needed.get(key,False):self.cooldown[key] = self.redetect_interval if self.tablet_requires_redetection else 2
            self.pending_complete = False
            return accepted
        polygons = result.get('polygons',{})
        if not isinstance(polygons,dict):polygons = {}
        for key in due:
            points = self._geometry(key,polygons.get(key))
            if points is None:
                self.pending[key] = None
                self.pending_complete = False
                self.cooldown[key] = self.redetect_interval if self.tablet_requires_redetection else 2
                continue
            previous = self.pending[key]
            complete = self._complete_tablet(result, points)
            if previous is not None and self._consistent(key,previous,points) and complete == self.pending_complete:
                accepted[key] = points.copy()
                self.pending[key] = None
                self.accepted_complete = complete
            else:
                self.pending[key] = points.copy()
                self.pending_complete = complete
            self.cooldown[key] = 0
        return accepted

    def audit(self):
        result = self.last_result or {}
        return dict(attempts=self.attempts,candidate_only=True,
                    tablet_requires_redetection=self.tablet_requires_redetection,
                    parallel_prefetch=self.parallel,prefetch=copy.deepcopy(self.prefetch_stats),
                    metadata=copy.deepcopy(result.get('metadata',{})),
                    diagnostics=copy.deepcopy(result.get('diagnostics',{})))
