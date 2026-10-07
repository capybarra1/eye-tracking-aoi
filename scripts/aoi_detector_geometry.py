"""Turn semantic masks into conservative AOI *candidates*.

No image model confidence is implied by these geometric checks. In particular,
an empty tablet mask is unknown, never confirmed absence. A border-touching
tablet describes only the visible intersection, not inferred offscreen corners.
This module does not read/write projects or approve annotations.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GeometryConfig:
    """Conservative candidate filters; pixel tolerances scale from 960x540."""

    screen_min_valid_fraction: float = .65
    screen_max_gap_fraction: float = .28
    screen_max_edge_gap_fraction: float = .08
    screen_min_segment_fraction: float = .08
    screen_max_residual_median_px: float = 3.
    screen_max_residual_p95_px: float = 8.
    tablet_min_area_fraction: float = .0015
    tablet_min_component_fraction: float = .90
    tablet_min_solidity: float = .94
    tablet_min_polygon_iou: float = .92


def _unknown(reason, **evidence):
    return dict(status='unknown', reason=reason, **evidence)


def _largest_gap(valid):
    padded = np.r_[True, valid, True]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    if not len(changes):
        return 0
    return int(np.max(changes[1::2]-changes[::2]))


def _basis(x, first, second):
    return np.column_stack([np.ones_like(x), x,
                            np.maximum(x-first, 0), np.maximum(x-second, 0)])


def _fit_three_segments(xs, ys, width, minimum_segment):
    """Continuous least-squares segments, coarse/fine turns, robust final fit."""
    scale = width-1.
    # Bound the search cost independently of video resolution.
    sample = np.unique(np.linspace(0, len(xs)-1, min(240, len(xs))).round().astype(int))
    x, y = xs[sample]/scale, ys[sample]
    minimum = max(4/scale, minimum_segment)
    coarse_step = 1/60.
    coarse = np.arange(minimum, 1-minimum+coarse_step*.5, coarse_step)
    best = None

    def search(first_values, second_values):
        nonlocal best
        for first in first_values:
            for second in second_values:
                if second-first < minimum or first < minimum or second > 1-minimum:
                    continue
                support = [np.count_nonzero(x < first),
                           np.count_nonzero((x >= first) & (x < second)),
                           np.count_nonzero(x >= second)]
                if min(support) < 8:
                    continue
                matrix = _basis(x, first, second)
                coefficients, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
                residual = matrix@coefficients-y
                # Sparse mask glitches should not move a screen's two joins.
                loss = float(np.mean(np.minimum(residual**2, 36.)))
                loss += 1e-5*((first-1/3)**2+(second-2/3)**2)
                if best is None or loss < best[0]:
                    best = (loss, first, second, coefficients)

    search(coarse, coarse)
    if best is None:
        return None
    _, first, second, _ = best
    fine_step = max(1/scale, coarse_step/10)
    search(np.arange(first-coarse_step, first+coarse_step+fine_step*.5, fine_step),
           np.arange(second-coarse_step, second+coarse_step+fine_step*.5, fine_step))
    _, first, second, coefficients = best
    matrix = _basis(xs/scale, first, second)
    for _ in range(5):
        residual = matrix@coefficients-ys
        sigma = max(1., 1.4826*float(np.median(np.abs(residual-np.median(residual)))))
        weights = np.sqrt(np.minimum(1., 2.5*sigma/np.maximum(np.abs(residual), 1e-6)))
        coefficients, _, _, _ = np.linalg.lstsq(matrix*weights[:, None], ys*weights, rcond=None)
    positions = np.array([0., first, second, 1.])
    points = np.column_stack([positions*scale, _basis(positions, first, second)@coefficients])
    return points, matrix@coefficients-ys


def _screen(mask, config):
    height, width = mask.shape
    evidence = dict(valid_column_fraction=0., tablet_occluded_columns=0,
                    longest_gap_fraction=1., residual_median_px=None,
                    residual_p95_px=None, residual_rmse_px=None)
    _, labels, stats, _ = cv2.connectedComponentsWithStats((mask == 1).astype(np.uint8), connectivity=8)
    top_labels = np.unique(labels[0])
    top_labels = top_labels[top_labels > 0]
    if not len(top_labels):
        return None, _unknown('no_top_connected_screen', **evidence)
    label = max(top_labels, key=lambda value: int(stats[value, cv2.CC_STAT_AREA]))
    region = labels == label
    observed = np.any(region, axis=0)
    # Only a bottom edge inside the frame is evidence. A full-height screen
    # column says the lower edge is outside, not that it equals the frame edge.
    bottom = height-1-np.argmax(region[::-1], axis=0)
    observed &= bottom < height-1
    tablet_columns = np.any(mask == 2, axis=0)
    first_tablet = np.argmax(mask == 2, axis=0)
    occluded = tablet_columns & (first_tablet <= bottom+2)
    valid = observed & ~occluded
    valid_x = np.flatnonzero(valid)
    evidence.update(valid_column_fraction=float(valid.mean()),
                    tablet_occluded_columns=int(occluded.sum()),
                    longest_gap_fraction=_largest_gap(valid)/width,
                    top_contact_fraction=float(region[0].mean()),
                    largest_component_pixels=int(stats[label, cv2.CC_STAT_AREA]))
    if len(valid_x) < max(32, width*config.screen_min_valid_fraction):
        return None, _unknown('insufficient_valid_columns', **evidence)
    edge_gap = max(int(valid_x[0]), int(width-1-valid_x[-1]))/width
    evidence['edge_gap_fraction'] = edge_gap
    if evidence['longest_gap_fraction'] > config.screen_max_gap_fraction:
        return None, _unknown('screen_boundary_gap_too_large', **evidence)
    if edge_gap > config.screen_max_edge_gap_fraction:
        return None, _unknown('screen_edge_extrapolation_too_large', **evidence)
    xs, ys = valid_x.astype(np.float64), bottom[valid].astype(np.float64)
    fitted = _fit_three_segments(xs, ys, width, config.screen_min_segment_fraction)
    if fitted is None:
        return None, _unknown('insufficient_segment_support', **evidence)
    points, residual = fitted
    absolute = np.abs(residual)
    evidence.update(residual_median_px=float(np.median(absolute)),
                    residual_p95_px=float(np.percentile(absolute, 95)),
                    residual_rmse_px=float(np.sqrt(np.mean(residual**2))))
    resolution_scale = height/540.
    if (evidence['residual_median_px'] > config.screen_max_residual_median_px*resolution_scale
            or evidence['residual_p95_px'] > config.screen_max_residual_p95_px*resolution_scale):
        return None, _unknown('screen_boundary_fit_residual_too_large', **evidence)
    if np.any(points[:, 1] < -3*resolution_scale) or np.any(points[:, 1] > height-1+3*resolution_scale):
        return None, _unknown('screen_boundary_extrapolates_outside_frame', **evidence)
    # This score summarizes mask geometry only; it is not calibrated accuracy.
    evidence['geometry_score'] = float(valid.mean()*max(0., 1-evidence['residual_p95_px']/(8*resolution_scale)))
    return points.tolist(), dict(status='candidate', reason='supported_three_segment_boundary', **evidence)


def _ordered_quad(points):
    points = np.asarray(points, np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1]-center[1], points[:, 0]-center[0]))]
    return np.roll(points, -int(np.argmin(points[:, 0]+points[:, 1])), axis=0)


def _tablet(mask, config):
    height, width = mask.shape
    binary = (mask == 2).astype(np.uint8)
    total = int(binary.sum())
    evidence = dict(partial=None, out_of_frame_unknown=True, confirmed_absent=False,
                    polygon_iou=None, component_fraction=0., area_pixels=0)
    if not total:
        return None, _unknown('no_tablet_mask_is_not_confirmed_absence', **evidence)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest = 1+int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[largest, cv2.CC_STAT_AREA])
    evidence.update(component_fraction=area/total, area_pixels=area)
    if area < max(64, width*height*config.tablet_min_area_fraction):
        return None, _unknown('tablet_region_too_small', **evidence)
    if area/total < config.tablet_min_component_fraction:
        return None, _unknown('tablet_mask_is_fragmented', **evidence)
    region = (labels == largest).astype(np.uint8)
    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = contour_area/max(hull_area, 1.)
    evidence['solidity'] = solidity
    if solidity < config.tablet_min_solidity:
        return None, _unknown('tablet_contour_not_convex_enough', **evidence)
    perimeter = cv2.arcLength(contour, True)
    best = None
    for fraction in (.003, .005, .008, .01, .015, .02, .025, .035, .045):
        polygon = cv2.approxPolyDP(contour, fraction*perimeter, True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        points = _ordered_quad(polygon)
        if np.min(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1)) < max(4., height*.015):
            continue
        rendered = np.zeros_like(region)
        cv2.fillPoly(rendered, [points.astype(np.int32)], 1)
        intersection = np.count_nonzero(rendered & region)
        union = np.count_nonzero(rendered | region)
        iou = intersection/max(union, 1)
        if best is None or iou > best[0]:
            best = (iou, points)
    if best is None:
        return None, _unknown('tablet_contour_has_no_supported_four_corners', **evidence)
    iou, points = best
    evidence['polygon_iou'] = float(iou)
    if iou < config.tablet_min_polygon_iou:
        return None, _unknown('tablet_quad_fit_too_inaccurate', **evidence)
    border_margin = max(1, round(height/540))
    left, top, component_width, component_height = stats[largest, :4]
    # Read the actual component extent, not just the simplified four corners:
    # approximation can otherwise erase a small border-touching clipped corner.
    partial = bool(left <= border_margin or top <= border_margin
                   or left+component_width-1 >= width-1-border_margin
                   or top+component_height-1 >= height-1-border_margin)
    evidence.update(partial=partial, out_of_frame_unknown=partial,
                    geometry_score=float(iou*(area/total)),
                    geometry_scope='visible_intersection_only' if partial else 'visible_four_corners')
    reason = 'partial_visible_quad_offscreen_corners_unknown' if partial else 'supported_visible_quadrilateral'
    return points.tolist(), dict(status='candidate', reason=reason, **evidence)


def mask_to_geometry(mask: np.ndarray, config: GeometryConfig | None = None, *,
                     tablet_only: bool = False) -> dict:
    """Return ``polygons`` and ``diagnostics`` for labels 0/1/2.

    ``polygons[name]`` is ``[four_points]`` in the existing AOI format, or None
    when geometry cannot be supported. None never means absent. No ``visible``
    or approval value is produced. Diagnostic scores concern geometry only.
    ``tablet_only`` skips screen fitting; tablet geometry is unchanged.
    """
    array = np.asarray(mask)
    if array.ndim != 2 or min(array.shape) < 16:
        raise ValueError('mask must be a two-dimensional label image at least 16 pixels per side')
    if not np.issubdtype(array.dtype, np.integer) or not np.isin(array, [0, 1, 2]).all():
        raise ValueError('mask labels must be integer values 0=other, 1=screen, 2=tablet')
    config = config or GeometryConfig()
    screen, screen_evidence = (None, _unknown('screen_geometry_not_requested')) if tablet_only else _screen(array, config)
    tablet, tablet_evidence = _tablet(array, config)
    return dict(polygons=dict(screen=[screen] if screen is not None else None,
                              tablet=[tablet] if tablet is not None else None),
                diagnostics=dict(screen=screen_evidence, tablet=tablet_evidence))
