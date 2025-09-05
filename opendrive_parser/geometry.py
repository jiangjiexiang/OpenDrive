"""
geometry.py

道路几何处理：采样、平滑与中心线提取。

包含：
- _compute_road_widths(road_elem)
- _catmull_rom_point / _catmull_rom_segment
- _smooth_local_gap
- extract_road_geometries(xodr_path, force_high_res_road=None) -> Dict[str, Dict]
"""
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import math
import xml.etree.ElementTree as ET

from .utils import _strip_ns, _load_root, _find_elements_by_tag, _get_child_by_tag


def _compute_road_widths(road_elem: ET.Element) -> Tuple[float, float]:
    """
    Compute approximate left and right road widths (meters) from the first laneSection.
    Returns (left_total, right_total).
    """
    left_total = 0.0
    right_total = 0.0
    lanes = _get_child_by_tag(road_elem, "lanes")
    if lanes is None:
        return left_total, right_total

    lane_sections = [c for c in list(lanes) if _strip_ns(c.tag) == "laneSection"]
    if not lane_sections:
        return left_total, right_total

    ls = lane_sections[0]
    left = _get_child_by_tag(ls, "left")
    right = _get_child_by_tag(ls, "right")

    def _sum_side(side_elem: Optional[ET.Element]) -> float:
        total = 0.0
        if side_elem is None:
            return total
        for lane in [c for c in list(side_elem) if _strip_ns(c.tag) == "lane"]:
            width_elem = _get_child_by_tag(lane, "width")
            if width_elem is None:
                continue
            w_a = width_elem.attrib.get("a")
            try:
                w_val = float(w_a) if w_a is not None else 0.0
            except Exception:
                w_val = 0.0
            total += abs(w_val)
        return total

    left_total = _sum_side(left)
    right_total = _sum_side(right)
    return left_total, right_total


def _catmull_rom_point(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2 * p1[0]) +
        (-p0[0] + p2[0]) * t +
        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
    )
    y = 0.5 * (
        (2 * p1[1]) +
        (-p0[1] + p2[1]) * t +
        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
    )
    return (x, y)


def _catmull_rom_segment(p0, p1, p2, p3, n):
    if n < 2:
        return [p1, p2]
    pts = []
    for i in range(n):
        t = i / (n - 1)
        pts.append(_catmull_rom_point(p0, p1, p2, p3, t))
    return pts


def _smooth_local_gap(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not pts or len(pts) < 4:
        return pts
    max_d = 0.0
    max_i = 0
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
        if d > max_d:
            max_d = d
            max_i = i
    gap_threshold = 1.6
    if max_d <= gap_threshold:
        return pts

    i = max_i
    p0 = pts[i-1] if i - 1 >= 0 else pts[i]
    p1 = pts[i]
    p2 = pts[i+1]
    p3 = pts[i+2] if i + 2 < len(pts) else pts[i+1]

    samples = min(max(8, int(math.ceil(max_d / 0.05))), 400)

    try:
        new_segment = _catmull_rom_segment(p0, p1, p2, p3, samples)
    except Exception:
        new_segment = []
        n_lin = min(max(8, int(math.ceil(max_d / 0.25))), 200)
        for k in range(n_lin + 1):
            t = k / n_lin
            new_segment.append((p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t))

    new_pts = pts[:i] + new_segment + pts[i+2:]
    cleaned = []
    lastp = None
    for p in new_pts:
        if lastp is None or (abs(p[0] - lastp[0]) > 1e-6 or abs(p[1] - lastp[1]) > 1e-6):
            cleaned.append(p)
            lastp = p
    return cleaned


def extract_road_geometries(xodr_path: str, force_high_res_road: Optional[str] = None) -> Dict[str, Dict]:
    """
    Extract approximate centerline polylines for each road.

    Returns a dict keyed by road id with values:
    {
      'name': <name>,
      'poly': [(x,y), ...], 'width_left': float, 'width_right': float
    }
    """
    root = _load_root(xodr_path)
    roads_out: Dict[str, Dict] = {}
    road_elems = _find_elements_by_tag(root, "road")

    for r in road_elems:
        rid = r.attrib.get("id", "")
        rname = r.attrib.get("name", "")
        poly: List[Tuple[float, float]] = []

        planview = _get_child_by_tag(r, "planView")
        if planview is None:
            wl, wr = _compute_road_widths(r)
            roads_out[rid] = {"name": rname, "poly": poly, "width_left": wl, "width_right": wr}
            continue

        for geom in [g for g in list(planview) if _strip_ns(g.tag) == "geometry"]:
            try:
                x0 = float(geom.attrib.get("x", "0"))
                y0 = float(geom.attrib.get("y", "0"))
                hdg = float(geom.attrib.get("hdg", "0"))
                length = float(geom.attrib.get("length", "0"))
            except ValueError:
                continue

            gchild = None
            for c in list(geom):
                gchild = c
                break

            base_samples = int(math.ceil(length / 1.0)) + 1
            samples = max(2, min(base_samples, 200))
            if force_high_res_road is not None and rid == force_high_res_road:
                samples = max(samples, min(int(math.ceil(length / 0.1)) + 1, 5000))
            if gchild is not None and _strip_ns(gchild.tag) == "arc":
                try:
                    k_tmp = float(gchild.attrib.get("curvature", "0"))
                except Exception:
                    k_tmp = 0.0
                if abs(k_tmp) > 1e-6:
                    step = max(0.2, min(1.0, 0.5 / (abs(k_tmp) * 2.0)))
                    samples = max(samples, min(int(math.ceil(length / step)) + 1, 2000))

            if gchild is None:
                for i in range(samples):
                    s = (i / (samples - 1)) * length
                    xi = x0 + s * math.cos(hdg)
                    yi = y0 + s * math.sin(hdg)
                    poly.append((xi, yi))
                continue

            gtype = _strip_ns(gchild.tag)
            if gtype == "line":
                for i in range(samples):
                    s = (i / (samples - 1)) * length
                    xi = x0 + s * math.cos(hdg)
                    yi = y0 + s * math.sin(hdg)
                    poly.append((xi, yi))
            elif gtype == "arc":
                try:
                    k = float(gchild.attrib.get("curvature", "0"))
                except ValueError:
                    k = 0.0
                if abs(k) < 1e-12:
                    for i in range(samples):
                        s = (i / (samples - 1)) * length
                        xi = x0 + s * math.cos(hdg)
                        yi = y0 + s * math.sin(hdg)
                        poly.append((xi, yi))
                else:
                    for i in range(samples):
                        s = (i / (samples - 1)) * length
                        ang = hdg + k * s
                        dx = (math.sin(ang) - math.sin(hdg)) / k
                        dy = (-math.cos(ang) + math.cos(hdg)) / k
                        xi = x0 + dx
                        yi = y0 + dy
                        poly.append((xi, yi))
            else:
                curv_s = gchild.attrib.get("curvStart") or gchild.attrib.get("curvstart") or gchild.attrib.get("curvStart".lower())
                curv_e = gchild.attrib.get("curvEnd") or gchild.attrib.get("curvend") or gchild.attrib.get("curvEnd".lower())
                try:
                    k0 = float(curv_s) if curv_s is not None else 0.0
                except Exception:
                    k0 = 0.0
                try:
                    k1 = float(curv_e) if curv_e is not None else 0.0
                except Exception:
                    k1 = 0.0
                if samples < 2:
                    samples = 2
                ds = length / (samples - 1) if samples > 1 else length
                x = x0
                y = y0
                hd = hdg
                poly.append((x, y))
                for i in range(1, samples):
                    s = i * ds
                    k = k0 + (k1 - k0) * (s / length) if length != 0 else k0
                    hd = hd + k * ds
                    x = x + math.cos(hd) * ds
                    y = y + math.sin(hd) * ds
                    poly.append((x, y))

        collapsed: List[Tuple[float, float]] = []
        last = None
        for p in poly:
            if last is None or (abs(p[0] - last[0]) > 1e-6 or abs(p[1] - last[1]) > 1e-6):
                collapsed.append(p)
                last = p

        def _postprocess_poly(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            if not pts:
                return pts
            out_pts = pts[:]
            i = 1
            while i < len(out_pts) - 1:
                x0, y0 = out_pts[i - 1]
                x1, y1 = out_pts[i]
                x2, y2 = out_pts[i + 1]
                seg_len = math.hypot(x2 - x0, y2 - y0)
                if seg_len > 1e-9:
                    t = ((x1 - x0) * (x2 - x0) + (y1 - y0) * (y2 - y0)) / (seg_len * seg_len)
                    proj_x = x0 + t * (x2 - x0)
                    proj_y = y0 + t * (y2 - y0)
                    dist_to_seg = math.hypot(x1 - proj_x, y1 - proj_y)
                else:
                    dist_to_seg = math.hypot(x1 - x0, y1 - y0)
                spike_thresh = 2.0
                surrounding_span = math.hypot(x2 - x0, y2 - y0)
                if dist_to_seg > spike_thresh and surrounding_span < 6.0:
                    out_pts[i] = ((x0 + x2) / 2.0, (y0 + y2) / 2.0)
                    i = max(1, i - 1)
                else:
                    i += 1

            max_gap = 1.5
            result: List[Tuple[float, float]] = []
            for idx in range(len(out_pts) - 1):
                ax, ay = out_pts[idx]
                bx, by = out_pts[idx + 1]
                result.append((ax, ay))
                d = math.hypot(bx - ax, by - ay)
                if d > max_gap:
                    n_add = min(int(math.ceil(d / max_gap)) - 1, 20)
                    for k in range(1, n_add + 1):
                        t = k / (n_add + 1)
                        result.append((ax + (bx - ax) * t, ay + (by - ay) * t))
            result.append(out_pts[-1])
            return result

        processed = _postprocess_poly(collapsed)
        if force_high_res_road is not None and rid == force_high_res_road:
            try:
                processed = _smooth_local_gap(processed)
            except Exception:
                pass
        wl, wr = _compute_road_widths(r)
        roads_out[rid] = {
            "name": rname,
            "poly": processed,
            "width_left": wl,
            "width_right": wr,
            "length": r.attrib.get("length"),
        }

    return roads_out
