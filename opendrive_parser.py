#!/usr/bin/env python3
"""
opendrive_parser.py

Minimal OpenDRIVE (.xodr) parser:
- count_roads(path) -> int
- list_roads(path) -> list of dicts with id/name/length (if present)
- count_junctions(path) -> int
- list_junctions(path) -> list of dicts with id/name/connection_count (if present)
- extract_road_geometries(path) -> dict of road_id -> polyline points

CLI:
    python opendrive_parser.py path/to/test.xodr
    python opendrive_parser.py path/to/test.xodr --list
    python opendrive_parser.py path/to/test.xodr --junctions
    python opendrive_parser.py path/to/test.xodr --list-junctions
    python opendrive_parser.py path/to/test.xodr --visualize [out.html]

Output (default):
    这个地图有<N>条路
"""
from __future__ import annotations
import sys
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import math
import os
import json


def _strip_ns(tag: Optional[str]) -> str:
    """Remove namespace from an element tag, e.g. '{ns}road' -> 'road'"""
    if tag is None:
        return ""
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _load_root(xodr_path: str) -> ET.Element:
    """Parse XML and return root element. Raises ET.ParseError on bad XML."""
    tree = ET.parse(xodr_path)
    return tree.getroot()


def _find_elements_by_tag(root: ET.Element, tag: str) -> List[ET.Element]:
    """
    Return a list of elements matching tag (namespace-agnostic).
    Prefer direct children of root; if none found, search the whole tree.
    """
    candidates = [child for child in list(root) if _strip_ns(child.tag) == tag]
    if candidates:
        return candidates
    return [elem for elem in root.iter() if _strip_ns(elem.tag) == tag]


def count_roads(xodr_path: str) -> int:
    root = _load_root(xodr_path)
    return len(_find_elements_by_tag(root, "road"))


def list_roads(xodr_path: str) -> List[Dict[str, str]]:
    root = _load_root(xodr_path)
    roads = []
    candidates = _find_elements_by_tag(root, "road")
    for r in candidates:
        road_info = {
            "id": r.attrib.get("id", ""),
            "name": r.attrib.get("name", ""),
            "length": r.attrib.get("length", ""),
        }
        roads.append(road_info)
    return roads


def count_junctions(xodr_path: str) -> int:
    root = _load_root(xodr_path)
    return len(_find_elements_by_tag(root, "junction"))


def list_junctions(xodr_path: str) -> List[Dict[str, str]]:
    root = _load_root(xodr_path)
    junctions = []
    candidates = _find_elements_by_tag(root, "junction")
    for j in candidates:
        connections = [child for child in list(j) if _strip_ns(child.tag) == "connection"]
        conn_summaries = []
        for c in connections:
            pairs = [f"{k}={v}" for k, v in c.attrib.items()]
            conn_summaries.append(";" .join(pairs) if pairs else "<no-attrs>")
        junction_info = {
            "id": j.attrib.get("id", ""),
            "name": j.attrib.get("name", ""),
            "connection_count": str(len(connections)),
            "connections": conn_summaries,
            "element": j,  # keep raw element for reference if needed
        }
        junctions.append(junction_info)
    return junctions


def _get_child_by_tag(parent: ET.Element, tag: str) -> Optional[ET.Element]:
    """Return first child whose stripped tag equals tag, else None."""
    for c in list(parent):
        if _strip_ns(c.tag) == tag:
            return c
    return None


def _compute_road_widths(road_elem: ET.Element) -> Tuple[float, float]:
    """
    Compute approximate left and right road widths (meters) from the first laneSection.

    Strategy (approximate):
    - Find <lanes>/<laneSection> (take the first laneSection).
    - Sum the first <width> 'a' attribute for lanes on the left and right sides.
    - Return (left_total, right_total). If lanes/widths missing, return (0.0, 0.0).
    This is a simple heuristic that gives a reasonable visual road width for most maps.
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
            # take the first width entry's 'a' coefficient as base width
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
    """
    Return a single point on a Catmull-Rom spline segment between p1 and p2 for parameter t in [0,1].
    Uses the standard uniform Catmull-Rom basis with 0.5 tension.
    """
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
    """
    Generate n points along the Catmull-Rom segment from p1 to p2 inclusive.
    n should be >= 2. Returns list of (x,y) with length n.
    """
    if n < 2:
        return [p1, p2]
    pts = []
    for i in range(n):
        t = i / (n - 1)
        pts.append(_catmull_rom_point(p0, p1, p2, p3, t))
    return pts


def _smooth_local_gap(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Detect the single largest gap in pts and replace that segment with a Catmull-Rom
    resample between the surrounding control points. This is intentionally local and conservative.
    Returns a new point list.
    """
    if not pts or len(pts) < 4:
        return pts
    # find largest consecutive gap
    max_d = 0.0
    max_i = 0
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
        if d > max_d:
            max_d = d
            max_i = i
    # only act on reasonably large gaps (tuneable)
    gap_threshold = 1.6  # meters; earlier diagnostics showed ~1.75m gap
    if max_d <= gap_threshold:
        return pts

    i = max_i
    # choose control points, duplicating endpoints at boundaries
    p0 = pts[i-1] if i - 1 >= 0 else pts[i]
    p1 = pts[i]
    p2 = pts[i+1]
    p3 = pts[i+2] if i + 2 < len(pts) else pts[i+1]

    # decide number of samples based on gap length (denser for larger gaps), cap to avoid runaway
    samples = min(max(8, int(math.ceil(max_d / 0.05))), 400)

    try:
        new_segment = _catmull_rom_segment(p0, p1, p2, p3, samples)
    except Exception:
        # fallback: simple linear interpolation if anything goes wrong
        new_segment = []
        n_lin = min(max(8, int(math.ceil(max_d / 0.25))), 200)
        for k in range(n_lin + 1):
            t = k / n_lin
            new_segment.append((p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t))

    # replace pts[i:i+2] with new_segment (ensure endpoints align)
    new_pts = pts[:i] + new_segment + pts[i+2:]
    # remove any accidental duplicates (very small epsilon)
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
      'poly': [(x,y), ...]  # list of points in map coordinates
    }

    Supports geometry children of type <line> and <arc curvature="...">.
    Each geometry provides absolute start x,y and hdg, and a length.
    We sample each geometry at ~1m intervals (clamped) to produce points.
    """
    root = _load_root(xodr_path)
    roads_out: Dict[str, Dict] = {}
    road_elems = _find_elements_by_tag(root, "road")

    for r in road_elems:
        rid = r.attrib.get("id", "")
        rname = r.attrib.get("name", "")
        poly: List[Tuple[float, float]] = []

        # find planView under road
        planview = _get_child_by_tag(r, "planView")
        if planview is None:
            wl, wr = _compute_road_widths(r)
            roads_out[rid] = {"name": rname, "poly": poly, "width_left": wl, "width_right": wr}
            continue

        # iterate geometry elements
        for geom in [g for g in list(planview) if _strip_ns(g.tag) == "geometry"]:
            # geometry attributes
            try:
                x0 = float(geom.attrib.get("x", "0"))
                y0 = float(geom.attrib.get("y", "0"))
                hdg = float(geom.attrib.get("hdg", "0"))
                length = float(geom.attrib.get("length", "0"))
            except ValueError:
                # malformed numbers - skip this geometry
                continue

            # find geometry type child (line/arc/spiral)
            gchild = None
            for c in list(geom):
                gchild = c
                break

            # choose sampling resolution: base ~1m; increase for curved geometries to smooth arcs
            base_samples = int(math.ceil(length / 1.0)) + 1
            samples = max(2, min(base_samples, 200))
            # allow forcing very high resolution for a specific road (debug/fix)
            if force_high_res_road is not None and rid == force_high_res_road:
                # use a fine step (0.1m) but cap samples to avoid runaway memory/cpu
                samples = max(samples, min(int(math.ceil(length / 0.1)) + 1, 5000))
            # if geometry is an arc, increase sampling density based on curvature
            if gchild is not None and _strip_ns(gchild.tag) == "arc":
                try:
                    k_tmp = float(gchild.attrib.get("curvature", "0"))
                except Exception:
                    k_tmp = 0.0
                if abs(k_tmp) > 1e-6:
                    # smaller step for higher curvature; clamp samples to a reasonable maximum
                    # step is in meters: higher curvature -> smaller step -> more samples
                    step = max(0.2, min(1.0, 0.5 / (abs(k_tmp) * 2.0)))
                    samples = max(samples, min(int(math.ceil(length / step)) + 1, 2000))

            if gchild is None:
                # no explicit type, treat as line
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
                # curvature attribute
                try:
                    k = float(gchild.attrib.get("curvature", "0"))
                except ValueError:
                    k = 0.0
                if abs(k) < 1e-12:
                    # degenerate to line
                    for i in range(samples):
                        s = (i / (samples - 1)) * length
                        xi = x0 + s * math.cos(hdg)
                        yi = y0 + s * math.sin(hdg)
                        poly.append((xi, yi))
                else:
                    # param s along length: position by integrating
                    for i in range(samples):
                        s = (i / (samples - 1)) * length
                        ang = hdg + k * s
                        dx = (math.sin(ang) - math.sin(hdg)) / k
                        dy = (-math.cos(ang) + math.cos(hdg)) / k
                        xi = x0 + dx
                        yi = y0 + dy
                        poly.append((xi, yi))
            else:
                # handle spiral (clothoid) approximation: curvature varies linearly from curvStart to curvEnd
                # We'll numerically integrate along the geometry using a simple Euler step.
                # Read possible attribute names for start/end curvature (OpenDRIVE may use curvStart/curvEnd)
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
                # ensure at least 2 samples
                if samples < 2:
                    samples = 2
                # integration step
                ds = length / (samples - 1) if samples > 1 else length
                # start at provided x0,y0,hdg
                x = x0
                y = y0
                hd = hdg
                # append start point
                poly.append((x, y))
                # integrate forward using Euler steps; this is a simple but robust approximation
                for i in range(1, samples):
                    s = i * ds
                    # linear curvature interpolation
                    k = k0 + (k1 - k0) * (s / length) if length != 0 else k0
                    # advance heading by curvature * ds
                    hd = hd + k * ds
                    # move along heading
                    x = x + math.cos(hd) * ds
                    y = y + math.sin(hd) * ds
                    poly.append((x, y))

        # Optionally, collapse consecutive duplicate points
        collapsed: List[Tuple[float, float]] = []
        last = None
        for p in poly:
            if last is None or (abs(p[0] - last[0]) > 1e-6 or abs(p[1] - last[1]) > 1e-6):
                collapsed.append(p)
                last = p

        # Post-process to remove spikes and fill large gaps:
        # - detect isolated spike points that deviate strongly from their neighbors and replace with midpoint
        # - insert linear interpolated points for gaps larger than max_gap
        def _postprocess_poly(pts: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
            if not pts:
                return pts
            # shallow copy to mutate
            out_pts = pts[:]
            # remove isolated spikes
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
                # if middle point deviates > spike_thresh and the surrounding span is not huge, treat as spike
                spike_thresh = 2.0  # meters
                surrounding_span = math.hypot(x2 - x0, y2 - y0)
                if dist_to_seg > spike_thresh and surrounding_span < 6.0:
                    out_pts[i] = ((x0 + x2) / 2.0, (y0 + y2) / 2.0)
                    i = max(1, i - 1)
                else:
                    i += 1

            # fill large gaps with linear interpolation
            max_gap = 1.5  # meters; if gap > max_gap, insert intermediate points
            result: List[Tuple[float, float]] = []
            for idx in range(len(out_pts) - 1):
                ax, ay = out_pts[idx]
                bx, by = out_pts[idx + 1]
                result.append((ax, ay))
                d = math.hypot(bx - ax, by - ay)
                if d > max_gap:
                    # number of points to insert, cap to avoid runaway
                    n_add = min(int(math.ceil(d / max_gap)) - 1, 20)
                    for k in range(1, n_add + 1):
                        t = k / (n_add + 1)
                        result.append((ax + (bx - ax) * t, ay + (by - ay) * t))
            result.append(out_pts[-1])
            return result

        processed = _postprocess_poly(collapsed)
        # If a specific road is being resampled/fixed, apply a local smoothing pass to
        # repair residual gaps/steps (conservative: only targets the largest gap).
        if force_high_res_road is not None and rid == force_high_res_road:
            try:
                processed = _smooth_local_gap(processed)
            except Exception:
                # be resilient: if smoothing fails, keep the original processed poly
                pass
        wl, wr = _compute_road_widths(r)
        roads_out[rid] = {"name": rname, "poly": processed, "width_left": wl, "width_right": wr}

    return roads_out


def _junction_marker_positions(root: ET.Element, roads_geoms: Dict[str, Dict]) -> Dict[str, Tuple[float, float, str, List[str]]]:
    """
    For each junction element, compute a robust representative coordinate, name and list of associated incoming road ids.

    Strategy improvements:
    - Collect incomingRoad ids (preserving order/duplicates).
    - Attempt to compute pairwise closest points between the polylines of incoming roads.
      * For each pair of incoming roads with geometry, compute the closest point between their polylines
        (by projecting endpoints onto opposing segments). Collect these closest points.
    - If pairwise closest points exist, use their centroid as the junction marker.
    - Fallbacks (in order): centroid of endpoint points collected from contactPoint; first-point fallback as before.
    - This keeps the computation local and robust to roads whose endpoints are not precisely aligned.
    """
    def _proj_point_to_seg(px, py, ax, ay, bx, by):
        """Project point P onto segment AB; return (cx,cy,dist) where C is closest point on segment."""
        vx = bx - ax
        vy = by - ay
        wx = px - ax
        wy = py - ay
        denom = vx * vx + vy * vy
        if denom == 0:
            # A and B coincide
            cx, cy = ax, ay
        else:
            t = (wx * vx + wy * vy) / denom
            if t < 0:
                t = 0
            elif t > 1:
                t = 1
            cx = ax + t * vx
            cy = ay + t * vy
        dist = math.hypot(px - cx, py - cy)
        return cx, cy, dist

    def _closest_point_between_polylines(pa: List[Tuple[float, float]], pb: List[Tuple[float, float]]):
        """
        Find an approximate closest point between two polylines by checking projections of endpoints
        onto opposing segments. Returns (cx, cy, distance) or (None, None, inf) if failed.
        """
        best = (None, None, float("inf"))
        if not pa or not pb:
            return best
        # iterate segments
        for i in range(len(pa) - 1):
            ax, ay = pa[i]
            bx, by = pa[i + 1]
            # project endpoints of pb onto segment AB
            for (px, py) in (pb[0],):
                cx, cy, d = _proj_point_to_seg(px, py, ax, ay, bx, by)
                if d < best[2]:
                    best = (cx, cy, d)
            for j in range(len(pb) - 1):
                # also project both pb segment endpoints onto pa segment
                px, py = pb[j]
                cx, cy, d = _proj_point_to_seg(px, py, ax, ay, bx, by)
                if d < best[2]:
                    best = (cx, cy, d)
        # symmetric pass: project endpoints of pa onto segments of pb
        for i in range(len(pb) - 1):
            ax, ay = pb[i]
            bx, by = pb[i + 1]
            for j in range(len(pa) - 1):
                px, py = pa[j]
                cx, cy, d = _proj_point_to_seg(px, py, ax, ay, bx, by)
                if d < best[2]:
                    best = (cx, cy, d)
        return best

    markers: Dict[str, Tuple[float, float, str, List[str]]] = {}
    j_elems = _find_elements_by_tag(root, "junction")
    for j in j_elems:
        jid = j.attrib.get("id", "")
        jname = j.attrib.get("name", "") or ""
        endpoint_pts = []
        incoming_list: List[str] = []
        # iterate all connection children
        for c in list(j):
            if _strip_ns(c.tag) != "connection":
                continue
            inc = c.attrib.get("incomingRoad")
            contact = c.attrib.get("contactPoint", "start")
            if not inc:
                continue
            incoming_list.append(inc)
            # if we have geometry for this road, pick the endpoint indicated by contactPoint
            if inc in roads_geoms:
                poly = roads_geoms[inc].get("poly", [])
                if not poly:
                    continue
                if contact == "end":
                    endpoint_pts.append(poly[-1])
                else:
                    endpoint_pts.append(poly[0])
        # remove duplicates in incoming_list while preserving order
        seen = set()
        incoming_unique: List[str] = []
        for r in incoming_list:
            if r not in seen:
                incoming_unique.append(r)
                seen.add(r)

        # collect pairwise closest points between all incoming road polylines
        pairwise_points: List[Tuple[float, float]] = []
        for i in range(len(incoming_unique)):
            for k in range(i + 1, len(incoming_unique)):
                a = incoming_unique[i]
                b = incoming_unique[k]
                if a in roads_geoms and b in roads_geoms:
                    pa = roads_geoms[a].get("poly", [])
                    pb = roads_geoms[b].get("poly", [])
                    if not pa or not pb:
                        continue
                    cx, cy, d = _closest_point_between_polylines(pa, pb)
                    if cx is not None and not math.isinf(d):
                        pairwise_points.append((cx, cy))

        chosen_point = None
        if pairwise_points:
            avg_x = sum(p[0] for p in pairwise_points) / len(pairwise_points)
            avg_y = sum(p[1] for p in pairwise_points) / len(pairwise_points)
            chosen_point = (avg_x, avg_y)
        elif endpoint_pts:
            # fallback: centroid of collected endpoint points (contactPoint)
            avg_x = sum(p[0] for p in endpoint_pts) / len(endpoint_pts)
            avg_y = sum(p[1] for p in endpoint_pts) / len(endpoint_pts)
            chosen_point = (avg_x, avg_y)
        else:
            # deep fallback: use first-point of first incoming road with geometry
            for inc in incoming_unique:
                if inc in roads_geoms:
                    p = roads_geoms[inc].get("poly", [])
                    if p:
                        chosen_point = p[0]
                        break

        if chosen_point is None:
            # nothing usable - skip
            continue

        markers[jid] = (chosen_point[0], chosen_point[1], jname, incoming_unique)
    return markers


def extract_objects_from_root(root: ET.Element) -> Dict[str, List[Dict[str, str]]]:
    """
    Parse scene objects in the OpenDRIVE and return a mapping:
      road_id -> [ { id, name, elem_tag, type, subtype, s, t, zOffset, has_outline, raw_attrs }, ... ]

    Improvements:
    - Recognize a broader set of element tags commonly used to place objects:
      e.g. <object>, <signal>, <controller>, <pole>, <surface>, <trafficLight>, ...
    - Keep the element tag in the parsed record as 'elem_tag' so the UI can show what kind
      of element it was in the source file.
    - Collect objects located under each <road>/<objects> preferentially, but also scan the
      whole document and place unattached objects under the "_global" key.
    - Preserve other attributes in raw_attrs for debugging and display.
    """
    objects_map: Dict[str, List[Dict[str, str]]] = {}

    # helper to add object to a road bucket
    def _add_obj_to(road_key: str, obj_dict: Dict[str, str]):
        objects_map.setdefault(road_key, []).append(obj_dict)

    # tags we consider as "objects" for visualization purposes
    extra_tags = set(["object", "signal", "controller", "pole", "surface", "trafficLight", "stopLine", "misc", "station"])

    # First: collect objects placed directly under each <road>/<objects>
    road_elems = _find_elements_by_tag(root, "road")
    for r in road_elems:
        rid = r.attrib.get("id", "")
        objs_container = _get_child_by_tag(r, "objects")
        if objs_container is None:
            continue
        for child in list(objs_container):
            tag = _strip_ns(child.tag)
            if tag not in extra_tags:
                continue
            try:
                oid = child.attrib.get("id", "") or ""
                oname = child.attrib.get("name", "") or ""
                otype = child.attrib.get("type", "") or child.attrib.get("objType", "") or ""
                subtype = child.attrib.get("subtype", "") or child.attrib.get("subType", "") or ""
                s_attr = child.attrib.get("s", "") or ""
                t_attr = child.attrib.get("t", "") or ""
                zoff = child.attrib.get("zOffset", "") or child.attrib.get("zoffset", "") or ""
                outline = _get_child_by_tag(child, "outline")
                has_outline = "yes" if outline is not None else "no"
                other_attrs = {k: v for k, v in child.attrib.items() if k not in ("id", "name", "type", "objType", "subtype", "subType", "s", "t", "zOffset", "zoffset")}
                # parse numeric dims if present and compute an estimated volume (m^3)
                def _parse_num(val):
                    try:
                        return float(val) if val is not None and val != "" else None
                    except Exception:
                        return None

                l_val = _parse_num(child.attrib.get("length", "") or "")
                w_val = _parse_num(child.attrib.get("width", "") or "")
                h_val = _parse_num(child.attrib.get("height", "") or "")

                # compute area from outline (shoelace) if outline provided
                def _outline_area(outline_elem):
                    if outline_elem is None:
                        return None
                    pts = []
                    for cc in list(outline_elem):
                        tagc = _strip_ns(cc.tag)
                        if tagc.startswith("corner"):
                            try:
                                x = float(cc.attrib.get("x", cc.attrib.get("X", "0")))
                                y = float(cc.attrib.get("y", cc.attrib.get("Y", "0")))
                                pts.append((x, y))
                            except Exception:
                                continue
                    if len(pts) < 3:
                        return None
                    area = 0.0
                    for i in range(len(pts)):
                        x1, y1 = pts[i]
                        x2, y2 = pts[(i + 1) % len(pts)]
                        area += x1 * y2 - x2 * y1
                    return abs(area) / 2.0

                default_height = 1.0
                volume = None
                volume_source = "unknown"

                if l_val is not None and w_val is not None and h_val is not None:
                    volume = l_val * w_val * h_val
                    volume_source = "length*width*height"
                else:
                    area = None
                    if outline is not None:
                        area = _outline_area(outline)
                    if area is not None and area > 0:
                        use_h = h_val if h_val is not None else default_height
                        volume = area * use_h
                        volume_source = "outline_area*height" if h_val is not None else "outline_area*default_height"
                    elif l_val is not None and w_val is not None:
                        use_h = h_val if h_val is not None else default_height
                        volume = l_val * w_val * use_h
                        volume_source = "length*width*height(defaulted)" if h_val is None else "length*width*height"
                    elif (l_val is not None or w_val is not None) and h_val is not None:
                        lp = l_val if l_val is not None else (w_val if w_val is not None else default_height)
                        wp = w_val if w_val is not None else (l_val if l_val is not None else default_height)
                        volume = lp * wp * h_val
                        volume_source = "partial_dims*height"
                    else:
                        volume = 0.0
                        volume_source = "insufficient_attributes"

                obj = {
                    "id": oid,
                    "name": oname,
                    "elem_tag": tag,
                    "type": otype,
                    "subtype": subtype,
                    "s": s_attr,
                    "t": t_attr,
                    "zOffset": zoff,
                    "has_outline": has_outline,
                    "raw_attrs": str(other_attrs),
                    "length": l_val if l_val is not None else "",
                    "width": w_val if w_val is not None else "",
                    "height": h_val if h_val is not None else "",
                    "volume": round(float(volume), 6) if volume is not None else 0.0,
                    "volume_source": volume_source,
                }
                _add_obj_to(rid, obj)
            except Exception:
                continue

    # Second: scan the whole document for any object-like nodes that were not inside road/<objects>
    seen_objs = set()
    for k, lst in objects_map.items():
        for o in lst:
            if o.get("id"):
                seen_objs.add(o.get("id"))

    # iterate all elements and pick those whose stripped tag is in extra_tags
    for elem in [e for e in root.iter() if _strip_ns(e.tag) in extra_tags]:
        try:
            tag = _strip_ns(elem.tag)
            oid = elem.attrib.get("id", "") or ""
            # skip objects we've already recorded (by id)
            if oid and oid in seen_objs:
                continue
            # try to find parent road for nicer placement
            parent_road = None
            for r in road_elems:
                if elem in list(r.iter()):
                    parent_road = r.attrib.get("id", "")
                    break
            road_key = parent_road if parent_road is not None else "_global"
            oname = elem.attrib.get("name", "") or ""
            otype = elem.attrib.get("type", "") or elem.attrib.get("objType", "") or ""
            subtype = elem.attrib.get("subtype", "") or elem.attrib.get("subType", "") or ""
            s_attr = elem.attrib.get("s", "") or ""
            t_attr = elem.attrib.get("t", "") or ""
            zoff = elem.attrib.get("zOffset", "") or elem.attrib.get("zoffset", "") or ""
            outline = _get_child_by_tag(elem, "outline")
            has_outline = "yes" if outline is not None else "no"
            other_attrs = {k: v for k, v in elem.attrib.items() if k not in ("id", "name", "type", "objType", "subtype", "subType", "s", "t", "zOffset", "zoffset")}
            # parse numeric dims if present and compute an estimated volume (m^3)
            def _parse_num(val):
                try:
                    return float(val) if val is not None and val != "" else None
                except Exception:
                    return None

            l_val = _parse_num(elem.attrib.get("length", "") or "")
            w_val = _parse_num(elem.attrib.get("width", "") or "")
            h_val = _parse_num(elem.attrib.get("height", "") or "")

            outline = _get_child_by_tag(elem, "outline")
            def _outline_area(outline_elem):
                if outline_elem is None:
                    return None
                pts = []
                for cc in list(outline_elem):
                    tagc = _strip_ns(cc.tag)
                    if tagc.startswith("corner"):
                        try:
                            x = float(cc.attrib.get("x", cc.attrib.get("X", "0")))
                            y = float(cc.attrib.get("y", cc.attrib.get("Y", "0")))
                            pts.append((x, y))
                        except Exception:
                            continue
                if len(pts) < 3:
                    return None
                area = 0.0
                for i in range(len(pts)):
                    x1, y1 = pts[i]
                    x2, y2 = pts[(i + 1) % len(pts)]
                    area += x1 * y2 - x2 * y1
                return abs(area) / 2.0

            default_height = 1.0
            volume = None
            volume_source = "unknown"

            if l_val is not None and w_val is not None and h_val is not None:
                volume = l_val * w_val * h_val
                volume_source = "length*width*height"
            else:
                area = None
                if outline is not None:
                    area = _outline_area(outline)
                if area is not None and area > 0:
                    use_h = h_val if h_val is not None else default_height
                    volume = area * use_h
                    volume_source = "outline_area*height" if h_val is not None else "outline_area*default_height"
                elif l_val is not None and w_val is not None:
                    use_h = h_val if h_val is not None else default_height
                    volume = l_val * w_val * use_h
                    volume_source = "length*width*height(defaulted)" if h_val is None else "length*width*height"
                elif (l_val is not None or w_val is not None) and h_val is not None:
                    lp = l_val if l_val is not None else (w_val if w_val is not None else default_height)
                    wp = w_val if w_val is not None else (l_val if l_val is not None else default_height)
                    volume = lp * wp * h_val
                    volume_source = "partial_dims*height"
                else:
                    volume = 0.0
                    volume_source = "insufficient_attributes"

            obj = {
                "id": oid,
                "name": oname,
                "elem_tag": tag,
                "type": otype,
                "subtype": subtype,
                "s": s_attr,
                "t": t_attr,
                "zOffset": zoff,
                "has_outline": has_outline,
                "raw_attrs": str(other_attrs),
                "length": l_val if l_val is not None else "",
                "width": w_val if w_val is not None else "",
                "height": h_val if h_val is not None else "",
                "volume": round(float(volume), 6) if volume is not None else 0.0,
                "volume_source": volume_source,
            }
            _add_obj_to(road_key, obj)
        except Exception:
            continue

    return objects_map


def write_visualization_html(out_path: str, roads_geoms: Dict[str, Dict], junction_markers: Dict[str, Tuple[float, float, str]], objects_map: Optional[Dict[str, List[Dict[str, str]]]] = None):
    """
    Produce a standalone, styled HTML file with an SVG showing road areas (polygons), centerlines and junction markers.

    This implementation:
    - Renders the SVG server-side (svg_polys / svg_centers / svg_marks)
    - Injects a small JS helper to handle click highlighting and simple viewBox zoom
    - Avoids embedding any Python f-string style templates; uses string concatenation
    - Enhancements: roads receive distinct colors (consistent per-id), objects of "line" type are rendered as lines
    - Layout improvement: produce a two-column grid: left = viewer+toolbar, right = info panel.
    """
    # collect all points
    all_x = []
    all_y = []
    for r in roads_geoms.values():
        for x, y in r.get("poly", []):
            all_x.append(x)
            all_y.append(y)
    for coord in junction_markers.values():
        if coord and len(coord) >= 2:
            all_x.append(coord[0])
            all_y.append(coord[1])

    if all_x and all_y:
        minx = min(all_x)
        maxx = max(all_x)
        miny = min(all_y)
        maxy = max(all_y)
    else:
        minx = miny = 0.0
        maxx = maxy = 1.0

    # margins and size
    width = 800
    height = 700
    margin = 30
    dx = maxx - minx
    dy = maxy - miny
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0
    scale = min((width - 2 * margin) / dx, (height - 2 * margin) / dy)

    # center the map within the SVG so there's symmetric padding
    map_w = dx * scale
    map_h = dy * scale
    offset_x = (width - map_w) / 2.0
    offset_y = (height - map_h) / 2.0

    def transform(pt: Tuple[float, float]) -> Tuple[float, float]:
        x, y = pt
        tx = (x - minx) * scale + offset_x
        # flip Y and account for offset_y
        ty = height - offset_y - (y - miny) * scale
        return tx, ty

    def compute_offsets(poly: List[Tuple[float, float]], left_w: float, right_w: float):
        """
        Given a centerline poly (list of (x,y)), compute left and right offset point lists.
        """
        if not poly or len(poly) < 2:
            return [], []
        left_pts = []
        right_pts = []
        n = len(poly)
        for i in range(n):
            x, y = poly[i]
            # estimate tangent using neighbors
            if i == 0:
                x2, y2 = poly[i + 1]
                tx = x2 - x
                ty = y2 - y
            elif i == n - 1:
                x1, y1 = poly[i - 1]
                tx = x - x1
                ty = y - y1
            else:
                x1, y1 = poly[i - 1]
                x2, y2 = poly[i + 1]
                tx = x2 - x1
                ty = y2 - y1
            # normalize tangent
            tlen = math.hypot(tx, ty)
            if tlen == 0:
                nx, ny = 0.0, 0.0
            else:
                ux, uy = tx / tlen, ty / tlen
                # normal pointing to the left of the tangent (rotate by +90deg)
                nx, ny = -uy, ux
            # offset points
            lx = x + nx * left_w
            ly = y + ny * left_w
            rx = x - nx * right_w
            ry = y - ny * right_w
            left_pts.append((lx, ly))
            right_pts.append((rx, ry))
        return left_pts, right_pts

    # color palette (pairs of fill, stroke). We'll choose color deterministically by road id hash.
    color_palette = [
        ("#e8f7ff", "#2f7fc1"),
        ("#fff7e6", "#e08b2f"),
        ("#eafaf1", "#2fa86a"),
        ("#fff0f0", "#d94d4d"),
        ("#f3e8ff", "#7b4dff"),
        ("#fff0d6", "#d97a00"),
        ("#f7f7f7", "#6b6b6b"),
    ]

    def _color_for_road(rid: str):
        if not rid:
            idx = 0
        else:
            s = str(rid)
            h = 0
            for ch in s:
                h = ((h << 5) - h) + ord(ch)
                h &= 0xFFFFFFFF
            idx = abs(h) % len(color_palette)
        return color_palette[idx]

    # Prepare containers and a map of roads associated with junctions.
    svg_road_polygons = []
    svg_centerlines = []
    # road_junctions maps incoming road id -> list of junction descriptors
    road_junctions: Dict[str, List[Dict[str, str]]] = {}
    for jid, coord in junction_markers.items():
        if not coord:
            continue
        jname = ""
        incoming_vals: List[str] = []
        # coord structure: (x, y, name, incoming) where incoming may be a list of ids
        if len(coord) >= 4:
            incoming_field = coord[3]
            # accept either a single string or a list of strings
            if isinstance(incoming_field, list):
                incoming_vals = [str(v) for v in incoming_field if v]
            elif incoming_field:
                incoming_vals = [str(incoming_field)]
            jname = coord[2] if coord[2] else ""
        elif len(coord) >= 3:
            jname = coord[2] if coord[2] else ""
        for inc in incoming_vals:
            if inc:
                road_junctions.setdefault(inc, []).append({"id": jid, "name": jname})

    # We'll render non-junction roads first, then junction-associated roads on top with a distinct style.
    normal_polys: List[str] = []
    normal_centers: List[str] = []
    junction_polys: List[str] = []
    junction_centers: List[str] = []

    for rid, rdata in roads_geoms.items():
        poly = rdata.get("poly", [])
        if not poly or len(poly) < 2:
            continue
        wl = float(rdata.get("width_left", 0.0) or 0.0)
        wr = float(rdata.get("width_right", 0.0) or 0.0)
        if wl <= 0 and wr <= 0:
            wl = wr = 2.5
        left_pts, right_pts = compute_offsets(poly, wl, wr)

        # determine colors - revert to original light-blue for all roads
        fill = "#e8f7ff"
        stroke = "#2f7fc1"
        stroke_width_poly = 0.8
        if rid in road_junctions:
            stroke_width_poly = 1.6

        # produce screen coordinates
        if left_pts and right_pts:
            poly_screen = [transform(p) for p in left_pts] + [transform(p) for p in reversed(right_pts)]
            pts_str = " ".join(f"{round(px,2)},{round(py,2)}" for px, py in poly_screen)
            title = f"road {rid} {rdata.get('name','')}"
            # decide target list based on whether this road is associated with a junction
            if rid in road_junctions:
                # junction-associated roads drawn on top with slightly different stroke
                junction_polys.append('<polygon class="road-poly junction-road" data-road-id="{rid}" data-road-name="{rname}" points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"><title>{title}</title></polygon>'.format(rid=rid, rname=rdata.get("name",""), pts=pts_str, title=title, fill=fill, stroke=stroke, sw=stroke_width_poly))
            else:
                normal_polys.append('<polygon class="road-poly" data-road-id="{rid}" data-road-name="{rname}" points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"><title>{title}</title></polygon>'.format(rid=rid, rname=rdata.get("name",""), pts=pts_str, title=title, fill=fill, stroke=stroke, sw=stroke_width_poly))

        pts_center = [transform(p) for p in poly]
        center_str = " ".join(f"{round(px,2)},{round(py,2)}" for px, py in pts_center)
        # centerline stroke darker variant
        center_stroke = "#333"
        if rid in road_junctions:
            center_stroke = "#8b3b00"
        else:
            # try to use stroke color darkened a bit
            center_stroke = stroke if stroke else "#444"
        center_sw = 1 if rid not in road_junctions else 2
        if rid in road_junctions:
            junction_centers.append('<polyline class="road-line junction-line" data-road-id="{rid}" data-road-name="{rname}" points="{pts}" fill="none" stroke="{stroke}" stroke-width="{sw}"><title>center {rid}</title></polyline>'.format(rid=rid, rname=rdata.get("name",""), pts=center_str, stroke=center_stroke, sw=center_sw))
        else:
            normal_centers.append('<polyline class="road-line" data-road-id="{rid}" data-road-name="{rname}" points="{pts}" fill="none" stroke="{stroke}" stroke-width="{sw}"><title>center {rid}</title></polyline>'.format(rid=rid, rname=rdata.get("name",""), pts=center_str, stroke=center_stroke, sw=center_sw))

    # Final render order: normal polygons/centers first, then junction polys/centers so they appear on top.
    svg_road_polygons = normal_polys + junction_polys
    svg_centerlines = normal_centers + junction_centers

    svg_markers = []
    svg_objects = []
    object_elem_tags = set()
    # helper: cumulative lengths and point-at-s along a polyline (map coords)
    def _poly_cumulative(pts):
        if not pts:
            return [0.0]
        cum = [0.0]
        for i in range(len(pts) - 1):
            ax, ay = pts[i]
            bx, by = pts[i + 1]
            cum.append(cum[-1] + math.hypot(bx - ax, by - ay))
        return cum

    def _point_at_s(pts, s_val):
        if not pts:
            return None
        cum = _poly_cumulative(pts)
        total = cum[-1]
        if total == 0:
            return pts[0]
        if s_val <= 0:
            return pts[0]
        if s_val >= total:
            return pts[-1]
        # find segment
        for i in range(len(cum) - 1):
            if cum[i] <= s_val <= cum[i + 1]:
                seg_len = cum[i + 1] - cum[i]
                if seg_len <= 0:
                    return pts[i]
                t = (s_val - cum[i]) / seg_len
                ax, ay = pts[i]
                bx, by = pts[i + 1]
                return (ax + (bx - ax) * t, ay + (by - ay) * t)
        return pts[-1]

    # junction markers (unchanged)
    for jid, coord in junction_markers.items():
        if not coord or len(coord) < 2:
            continue
        x = coord[0]
        y = coord[1]
        name = coord[2] if len(coord) > 2 and coord[2] else f"J{jid}"
        tx, ty = transform((x, y))
        svg_markers.append(f'<circle class="junction" data-junction-id="{jid}" data-junction-name="{name}" cx="{round(tx,2)}" cy="{round(ty,2)}" r="5" fill="#d94d4d" stroke="#7a1f1f" stroke-width="1.2"/>')
        svg_markers.append(f'<text class="jlabel" x="{round(tx+8,2)}" y="{round(ty+4,2)}" font-size="12" fill="#222">{name}</text>')

    # render parsed objects (if provided). We'll approximate position using object's 's' along the road centerline,
    # and 't' as lateral offset (meters) to compute a final location. If no s/t or no geometry, fallback to road centroid.
    if objects_map:
        for road_id, objs in objects_map.items():
            road_poly = roads_geoms.get(road_id, {}).get("poly", []) if roads_geoms else []
            for o in objs:
                oid = o.get("id", "")
                oname = o.get("name", "") or ""
                otype = o.get("type", "") or ""
                elem_tag_raw = o.get("elem_tag", "") or ""
                elem_tag = elem_tag_raw.lower() if isinstance(elem_tag_raw, str) else ""
                subtype = o.get("subtype", "") or ""
                s_str = o.get("s", "") or ""
                t_str = o.get("t", "") or ""
                try:
                    s_val = float(s_str) if s_str != "" else 0.0
                except Exception:
                    s_val = 0.0
                try:
                    t_val = float(t_str) if t_str != "" else 0.0
                except Exception:
                    t_val = 0.0

                # compute base position
                pos = None
                if road_poly and len(road_poly) >= 2:
                    try:
                        pos = _point_at_s(road_poly, s_val)
                    except Exception:
                        pos = None
                if pos is None:
                    # fallback: centroid of road poly
                    if road_poly:
                        sx = sum(p[0] for p in road_poly) / len(road_poly)
                        sy = sum(p[1] for p in road_poly) / len(road_poly)
                        pos = (sx, sy)
                    else:
                        # final fallback: place at map center
                        pos = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

                # apply lateral offset t (positive to left of road direction)
                tangent = (1.0, 0.0)
                if road_poly and len(road_poly) >= 2 and t_val != 0.0:
                    # find nearest segment to s_val for tangent estimation
                    cum = _poly_cumulative(road_poly)
                    seg_idx = 0
                    for i in range(len(cum) - 1):
                        if cum[i] <= s_val <= cum[i + 1]:
                            seg_idx = i
                            break
                    # clamp idx
                    if seg_idx >= len(road_poly) - 1:
                        seg_idx = len(road_poly) - 2
                    ax, ay = road_poly[seg_idx]
                    bx, by = road_poly[seg_idx + 1]
                    txv = bx - ax
                    tyv = by - ay
                    tlen = math.hypot(txv, tyv)
                    if tlen > 1e-9:
                        ux, uy = txv / tlen, tyv / tlen
                        # left normal
                        nx, ny = -uy, ux
                        pos = (pos[0] + nx * t_val, pos[1] + ny * t_val)
                        tangent = (ux, uy)
                else:
                    # attempt to get a tangent near the point using first segment if available
                    if road_poly and len(road_poly) >= 2:
                        ax, ay = road_poly[0]
                        bx, by = road_poly[1]
                        txv = bx - ax
                        tyv = by - ay
                        tlen = math.hypot(txv, tyv)
                        if tlen > 1e-9:
                            tangent = (txv / tlen, tyv / tlen)

                # transform to screen coordinates
                sx, sy = transform(pos)
                # serialize object metadata for frontend (JSON). normalize unknown numeric 'type' placeholders (e.g. -1)
                vol = 0.0
                try:
                    obj_for_json = dict(o) if isinstance(o, dict) else {"value": str(o)}
                    # normalize numeric-type placeholder
                    tval = obj_for_json.get("type", "")
                    try:
                        tnum = int(float(tval)) if tval != "" else None
                        if tnum is not None and tnum < 0:
                            obj_for_json["type"] = f"unspecified({tnum})"
                    except Exception:
                        pass
                    try:
                        vol = float(obj_for_json.get("volume", 0.0) or 0.0)
                    except Exception:
                        vol = 0.0
                    _obj_json = json.dumps(obj_for_json, ensure_ascii=False)
                except Exception:
                    _obj_json = str(o)
                    vol = 0.0
                # escape double-quotes for safe HTML attribute embedding
                _obj_attr = _obj_json.replace('"', '"')
                object_elem_tags.add(elem_tag)
                # choose marker size based on volume (cube-root scale) with caps for readability
                r_px = 4
                if vol and vol > 0:
                    r_px = 3 + min(36, (vol ** (1.0/3.0)) * 0.9)

                title_txt = oname if oname else (f"object {oid}" if oid else "object")
                vol_src = ""
                try:
                    if isinstance(obj_for_json, dict):
                        vol_src = obj_for_json.get("volume_source", "") or ""
                except Exception:
                    vol_src = ""

                # detect "line-like" objects: by elem_tag or type/subtype containing 'line'
                is_line_like = False
                if isinstance(elem_tag, str) and "line" in elem_tag:
                    is_line_like = True
                if isinstance(otype, str) and "line" in otype.lower():
                    is_line_like = True
                if isinstance(subtype, str) and "line" in subtype.lower():
                    is_line_like = True

                # We will render line-like objects as short oriented lines along the road tangent.
                if is_line_like and road_poly and len(road_poly) >= 2:
                    # determine a small map-length for the line (meters)
                    map_len = 1.0  # 1 meter line by default
                    try:
                        # if object has explicit length attribute, prefer that (safe fallback)
                        if isinstance(o.get("length", ""), (int, float)) and float(o.get("length", 0)) > 0:
                            map_len = float(o.get("length"))
                    except Exception:
                        pass
                    # line endpoints in map coords
                    ux, uy = tangent
                    half_vec = (ux * (map_len / 2.0), uy * (map_len / 2.0))
                    p1_map = (pos[0] - half_vec[0], pos[1] - half_vec[1])
                    p2_map = (pos[0] + half_vec[0], pos[1] + half_vec[1])
                    p1_px = transform(p1_map)
                    p2_px = transform(p2_map)
                    # line style based on elem_tag / object type — 默认黑色（用户要求）
                    line_color = "#000000"
                    stroke_w = max(1.0, min(3.0, r_px / 2.0))
                    try:
                        if elem_tag:
                            # 尽量为特殊标记选择颜色变体（但默认保持黑色）
                            if "stop" in elem_tag:
                                line_color = "#d94d4d"
                            elif "cross" in elem_tag or "zebra" in elem_tag:
                                line_color = "#000000"
                    except Exception:
                        pass
                    svg_objects.append(f'<g class="xodr-object-group" data-road-id="{road_id}" data-obj-id="{oid}" data-obj-name="{oname}" data-obj-type="{otype}" data-elem-tag="{elem_tag_raw}" data-raw="{_obj_attr}">')
                    svg_objects.append(f'<line class="xodr-object-line" x1="{round(p1_px[0],2)}" y1="{round(p1_px[1],2)}" x2="{round(p2_px[0],2)}" y2="{round(p2_px[1],2)}" stroke="{line_color}" stroke-width="{stroke_w}" stroke-linecap="round" data-obj-id="{oid}" data-obj-name="{oname}" data-obj-type="{otype}" data-road-id="{road_id}" data-elem-tag="{elem_tag_raw}" data-raw="{_obj_attr}" />')
                    svg_objects.append(f'<title>{title_txt}</title></g>')
                else:
                    # default rectangular marker for other objects
                    # choose fill color by elem_tag (simple mapping)
                    fill_col = "#2b9df4"
                    stroke_col = "#083a73"
                    try:
                        if elem_tag:
                            if "signal" in elem_tag:
                                fill_col = "#ffd24d"; stroke_col = "#b06f00"
                            elif "pole" in elem_tag or "controller" in elem_tag:
                                fill_col = "#93c47d"; stroke_col = "#2e7d32"
                            elif "station" in elem_tag:
                                fill_col = "#f3e8ff"; stroke_col = "#7b4dff"
                    except Exception:
                        pass
                    svg_objects.append('<g class="xodr-object-group" data-road-id="{road}" data-obj-id="{oid}" data-obj-name="{oname}" data-obj-type="{otype}" data-elem-tag="{etag}" data-raw="{raw}">'.format(road=road_id, oid=oid, oname=oname.replace('"', '"'), otype=otype, etag=elem_tag_raw, raw=_obj_attr))
                    x_px = round(sx - r_px, 2)
                    y_px = round(sy - r_px, 2)
                    w_px = round(r_px * 2, 2)
                    h_px = round(r_px * 2, 2)
                    rx_px = round(max(0.0, r_px * 0.15), 2)
                    svg_objects.append(f'<rect class="xodr-object" x="{x_px}" y="{y_px}" width="{w_px}" height="{h_px}" rx="{rx_px}" ry="{rx_px}" fill="{fill_col}" stroke="{stroke_col}" stroke-width="0.8" data-obj-id="{oid}" data-obj-name="{oname}" data-obj-type="{otype}" data-road-id="{road_id}" data-elem-tag="{elem_tag_raw}" data-raw="{_obj_attr}" data-volume="{vol if vol is not None else ""}" data-volume-source="{vol_src}"></rect>')
                    svg_objects.append(f'<title>{title_txt}</title></g>')

    # small JS mapping for road->junctions
    def _esc_js(s: str) -> str:
        return (str(s).replace("\\", "\\\\").replace('"', '\\"'))
    js_items = []
    for rid, jlist in road_junctions.items():
        items = ",".join(f'{{"id":"{_esc_js(j.get("id",""))}","name":"{_esc_js(j.get("name",""))}"}}' for j in jlist)
        js_items.append(f'"{_esc_js(rid)}":[{items}]')
    js_map = "{" + ",".join(js_items) + "}"

    svg_polys = "".join(svg_road_polygons)
    svg_centers = "".join(svg_centerlines)
    svg_marks = "".join(svg_markers)

    # Build final HTML using concatenation to avoid f-string/format pitfalls
    # Use a two-column responsive grid: left = viewer+toolbar, right = info panel.
    html = (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        "<meta charset=\"utf-8\"/>\n"
        "<title>OpenDRIVE visualization</title>\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"/>\n"
        "<style>\n"
        "  :root { --bg: #f6f8fb; --card:#fff; --gap:28px; --radius:10px; --info-width:360px; --divider-width:12px; --max-width:1200px; }\n"
        "  *{box-sizing:border-box}\n"
        "  body { font-family: Inter, Arial, Helvetica, sans-serif; padding:20px; margin:0; background:var(--bg); display:flex; justify-content:center; }\n"
        "  .page { width:100%; max-width:var(--max-width); }\n"
"  .container { display:grid; grid-template-columns: 1fr var(--info-width); gap:var(--gap); align-items:start; width:100%; }\n"
        "  @media (max-width:980px){ .container { grid-template-columns:1fr; } .info { order:2; } .canvas-area { order:1; } }\n"
"  .canvas-area { background: transparent; display:flex; flex-direction:column; gap:12px; }\n"
        "  .card { background:var(--card); border-radius:var(--radius); box-shadow:0 8px 22px rgba(16,24,40,0.06); padding:12px; }\n"
        "  .toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }\n"
        "  button.tool { padding:8px 12px; border-radius:8px; border:1px solid #d7e0f2; background:#fff; cursor:pointer; font-size:13px }\n"
        "  .viewer { padding:8px; }\n"
        "  .svg-wrap { width:100%; background:#fff; border-radius:8px; overflow:hidden; }\n"
        "  svg { display:block; width:100%; height:auto; }\n"
"  .info { width:100%; max-width:var(--info-width); padding:14px; border-radius:var(--radius); background:var(--card); box-shadow:0 8px 22px rgba(16,24,40,0.04); font-size:13px; }\n"
        "  .info h4 { margin:0 0 8px 0; font-size:15px }\n"
        "  .info .section { margin-bottom:12px }\n"
        "  .road-poly { transition: fill 160ms ease, stroke 160ms ease; }\n"
        "  .road-poly:hover { filter:brightness(1.04); }\n"
        "  .road-poly.selected { stroke:#222 !important; stroke-width:2 !important; fill:#fff3bf !important; }\n"
        "  .xodr-object.selected { stroke:#000 !important; stroke-width:1.6 !important; }\n"
        "  .road-line { pointer-events:none; }\n"
        "  .junction { cursor:default }\n"
        "  .jlabel { font-family: Arial, sans-serif; pointer-events:none }\n"
        "  .xodr-object-line { cursor:pointer; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<div class=\"page\">\n"
        "  <div class=\"container\">\n"
        "    <div class=\"canvas-area\">\n"
        "      <div class=\"card toolbar\" role=\"toolbar\" aria-label=\"Map tools\">\n"
        "        <button class=\"tool\" id=\"zoom-in\">Zoom In</button>\n        <button class=\"tool\" id=\"zoom-out\">Zoom Out</button>\n        <button class=\"tool\" id=\"reset-view\">Reset</button>\n      </div>\n"
        "      <div class=\"card viewer\">\n"
        "        <div class=\"svg-wrap\" role=\"img\" aria-label=\"OpenDRIVE map\">\n"
f"          <svg id=\"map-svg\" width=\"100%\" height=\"auto\" viewBox=\"0 0 {width} {height}\" preserveAspectRatio=\"xMidYMid meet\" xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\">\n"
        "            <g id=\"map-layer\">\n"
        + svg_polys + svg_centers + svg_marks + "".join(svg_objects) +
        "            </g>\n"
        "          </svg>\n"
        "        </div>\n"
        "      </div>\n"
        "    </div>\n"
"    <aside class=\"info card\" id=\"info-panel\" aria-live=\"polite\">\n"
"      <h4>Map Info</h4>\n"
"      <div id=\"info-content\" class=\"section\">点击道路或转弯口查看信息。</div>\n"
"      <div id=\"object-filters-container\" class=\"section\">\n"
"        <div id=\"object-filters-card\" class=\"card\" style=\"padding:8px; box-shadow: none;\">\n"
"          <h5 style=\"margin:0 0 8px 0;font-size:14px\">Objects</h5>\n"
"          <div id=\"object-filters\" class=\"section\"></div>\n"
"        </div>\n"
"      </div>\n"
"      <div class=\"section\" style=\"color:#666;font-size:12px\">Radius = 3 + min(36, 0.9 * cube_root(volume[m^3])) — default height when missing: 1.0 m<br/>Volume source shown when available.</div>\n"
"      <div class=\"section\" style=\"color:#666;font-size:12px\">Generated by opendrive_parser.py</div>\n"
"    </aside>\n"
        "  </div>\n"
        "</div>\n"
        "<script>\n"
        "  // small, safe interactive helpers (preserve earlier behavior)\n"
        "  var roadJunctions = " + js_map + ";\n"
        "  var svg = document.getElementById('map-svg');\n"
        "  var mapLayer = document.getElementById('map-layer');\n"
        "  var infoContent = document.getElementById('info-content');\n"
        "  function clearSelection(){\n"
        "    document.querySelectorAll('.road-poly.selected').forEach(function(el){ el.classList.remove('selected'); });\n"
        "  }\n"
        "  function showRoadInfo(id,name){\n"
        "    var html = '<b>Road ID:</b> ' + id + '<br/>' + '<b>Name:</b> ' + (name||'<无名称>') + '<br/>';\n"
        "    var juncs = roadJunctions[id] || [];\n"
        "    if (juncs.length === 0) html += '<b>关联 junction:</b> 无'; else { html += '<b>关联 junction:</b><ul>'; for(var i=0;i<juncs.length;i++){ html += '<li>' + (juncs[i].name||('J'+juncs[i].id)) + ' (id=' + juncs[i].id + ')</li>'; } html += '</ul>'; }\n"
        "    infoContent.innerHTML = html;\n"
        "  }\n"
        "  // attach click handlers\n"
        "  if (mapLayer) {\n"
        "    mapLayer.querySelectorAll('.road-poly').forEach(function(el){\n"
        "      el.style.cursor = 'pointer';\n"
        "      el.addEventListener('click', function(evt){ evt.stopPropagation(); clearSelection(); el.classList.add('selected'); showRoadInfo(el.getAttribute('data-road-id'), el.getAttribute('data-road-name')); });\n"
        "    });\n"
        "    mapLayer.querySelectorAll('.junction').forEach(function(el){\n"
        "      el.addEventListener('click', function(evt){ evt.stopPropagation(); var jid = el.getAttribute('data-junction-id'); var jn = el.getAttribute('data-junction-name'); infoContent.innerHTML = '<b>Junction ID:</b> ' + jid + '<br/><b>Name:</b> ' + (jn||'<无名称>'); });\n"
        "    });\n"
        "    mapLayer.querySelectorAll('.xodr-object, .xodr-object-line').forEach(function(el){\n"
        "      el.style.cursor = 'pointer';\n        el.addEventListener('click', function(evt){\n"
        "        evt.stopPropagation();\n"
        "        document.querySelectorAll('.xodr-object.selected').forEach(function(x){ x.classList.remove('selected'); });\n"
        "        document.querySelectorAll('.xodr-object-line.selected').forEach(function(x){ x.classList.remove('selected'); });\n"
        "        el.classList.add('selected');\n"
        "        var parent = el.parentElement || el;\n"
        "        var raw = parent.getAttribute('data-raw') || el.getAttribute('data-raw') || '{}';\n"
        "        var parsed = null;\n"
        "        try { parsed = JSON.parse(raw); } catch(e) { parsed = raw; }\n"
        "        var html = '<b>Object ID:</b> ' + (el.getAttribute('data-obj-id')||'<no-id>') + '<br/>' +\n"
        "                   '<b>Name:</b> ' + (el.getAttribute('data-obj-name')||'<无名称>') + '<br/>' +\n"
        "                   '<b>Type:</b> ' + (el.getAttribute('data-obj-type')||'') + '<br/>' +\n"
        "                   '<b>Elem Tag:</b> ' + (el.getAttribute('data-elem-tag')||'') + '<br/>' +\n"
        "                   '<b>Road:</b> ' + (el.getAttribute('data-road-id')||'') + '<br/>' +\n"
        "                   '<pre style=\"white-space:pre-wrap\">' + (typeof parsed === 'string' ? parsed : JSON.stringify(parsed, null, 2)) + '</pre>';\n"
        "        infoContent.innerHTML = html;\n"
        "      });\n"
        "    });\n"
        "  }\n"
        "  // simple viewBox zoom helpers\n"
        "  var vb = {x:0,y:0,w:" + str(width) + ",h:" + str(height) + "};\n"
        "  function setViewBox(){ if (svg) svg.setAttribute('viewBox', vb.x + ' ' + vb.y + ' ' + vb.w + ' ' + vb.h); }\n"
        "  var zi = document.getElementById('zoom-in'); if (zi) zi.addEventListener('click', function(){ vb.w *= 0.8; vb.h *= 0.8; vb.x += vb.w*0.1; vb.y += vb.h*0.1; setViewBox(); });\n"
        "  var zo = document.getElementById('zoom-out'); if (zo) zo.addEventListener('click', function(){ vb.x -= vb.w*0.1; vb.y -= vb.h*0.1; vb.w /= 0.8; vb.h /= 0.8; setViewBox(); });\n"
        "  var rz = document.getElementById('reset-view'); if (rz) rz.addEventListener('click', function(){ fitToContent(); });\n"
        "  function fitToContent(padPx){\n"
        "    padPx = (padPx === undefined) ? 20 : padPx;\n"
        "    try {\n"
        "      var bbox = mapLayer.getBBox();\n"
        "      var bw = bbox.width || 1;\n"
        "      var bh = bbox.height || 1;\n"
        "      var pad = padPx;\n"
        "      var targetW = bw + pad * 2;\n"
        "      var targetH = bh + pad * 2;\n"
        "      var rect = svg.getBoundingClientRect();\n"
        "      var svgRatio = rect.width / rect.height || (800/700);\n"
        "      var contentRatio = targetW / targetH;\n"
        "      var finalW = targetW, finalH = targetH;\n"
        "      if (contentRatio > svgRatio) finalH = targetW / svgRatio; else finalW = targetH * svgRatio;\n"
        "      var baseW = 800;\n"
        "      var ratio = finalH / finalW || 1.0;\n"
        "      var maxFactor = 6.0;\n"
        "      var minFactor = 0.25;\n"
        "      var factor = finalW / baseW;\n"
        "      if (factor > maxFactor) { finalW = baseW * maxFactor; finalH = finalW * ratio; }\n"
        "      if (factor < minFactor) { finalW = baseW * minFactor; finalH = finalW * ratio; }\n"
        "      vb.x = bbox.x - (finalW - bw)/2 - pad;\n"
        "      vb.y = bbox.y - (finalH - bh)/2 - pad;\n"
        "      vb.w = finalW; vb.h = finalH; setViewBox();\n"
        "    } catch (e) { vb = {x:0,y:0,w:800,h:700}; setViewBox(); }\n"
        "  }\n"
        "  window.addEventListener('load', function(){ setTimeout(fitToContent, 50); });\n"
        "  window.addEventListener('resize', function(){ setTimeout(fitToContent, 150); });\n"
        "  svg.addEventListener('click', function(){ clearSelection(); if(infoContent) infoContent.innerHTML = '点击道路或转弯口查看信息。'; });\n"
        "  // Wheel zoom + drag pan for the existing viewBox-based map (preserve behavior)\n"
        "  (function(){\n"
        "    var svgEl = svg; if (!svgEl) return;\n"
        "    var isDragging = false;\n"
        "    var dragStart = {x:0,y:0};\n"
        "    var vbStart = {x:0,y:0};\n"
        "    function clientToView(x, y, rect) {\n"
        "      return {\n"
        "        vx: vb.x + (x - rect.left) * vb.w / rect.width,\n"
        "        vy: vb.y + (y - rect.top) * vb.h / rect.height\n"
        "      };\n"
        "    }\n"
        "    svgEl.addEventListener('wheel', function(e){\n"
        "      e.preventDefault();\n"
        "      var rect = svgEl.getBoundingClientRect();\n"
        "      var mouse = clientToView(e.clientX, e.clientY, rect);\n"
        "      var zoomFactor = e.deltaY < 0 ? 0.85 : (1/0.85);\n"
        "      var newW = vb.w * zoomFactor;\n"
        "      var newH = vb.h * zoomFactor;\n"
        "      var minW = 40, maxW = 10000;\n"
        "      if (newW < minW) newW = minW;\n"
        "      if (newW > maxW) newW = maxW;\n"
        "      newH = newW * (vb.h / vb.w);\n"
        "      vb.x = mouse.vx - ((e.clientX - rect.left) * newW / rect.width);\n"
        "      vb.y = mouse.vy - ((e.clientY - rect.top) * newH / rect.height);\n"
        "      vb.w = newW; vb.h = newH; setViewBox();\n"
        "    }, { passive: false });\n"
        "    svgEl.addEventListener('mousedown', function(e){ if (e.button !== 0) return; isDragging = true; dragStart.x = e.clientX; dragStart.y = e.clientY; vbStart.x = vb.x; vbStart.y = vb.y; svgEl.style.cursor = 'grabbing'; });\n"
        "    window.addEventListener('mousemove', function(e){ if (!isDragging) return; var rect = svgEl.getBoundingClientRect(); var dxPx = e.clientX - dragStart.x; var dyPx = e.clientY - dragStart.y; var dx = -dxPx * vb.w / rect.width; var dy = -dyPx * vb.h / rect.height; vb.x = vbStart.x + dx; vb.y = vbStart.y + dy; setViewBox(); });\n"
        "    window.addEventListener('mouseup', function(e){ if (!isDragging) return; isDragging = false; svgEl.style.cursor = 'default'; });\n"
        "    var lastTouchDist = null; var lastTouchMid = null;\n"
        "    svgEl.addEventListener('touchstart', function(e){ if (e.touches.length === 1) { var t = e.touches[0]; isDragging = true; dragStart.x = t.clientX; dragStart.y = t.clientY; vbStart.x = vb.x; vbStart.y = vb.y; } else if (e.touches.length === 2) { lastTouchDist = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); lastTouchMid = { x: (e.touches[0].clientX + e.touches[1].clientX) / 2, y: (e.touches[0].clientY + e.touches[1].clientY) / 2 }; } }, { passive: true });\n"
        "    svgEl.addEventListener('touchmove', function(e){ if (e.touches.length === 1 && isDragging) { var t = e.touches[0]; var rect = svgEl.getBoundingClientRect(); var dxPx = t.clientX - dragStart.x; var dyPx = t.clientY - dragStart.y; vb.x = vbStart.x - dxPx * vb.w / rect.width; vb.y = vbStart.y - dyPx * vb.h / rect.height; setViewBox(); } else if (e.touches.length === 2 && lastTouchDist !== null) { e.preventDefault(); var d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); var rect = svgEl.getBoundingClientRect(); var mid = { x: (e.touches[0].clientX + e.touches[1].clientX) / 2, y: (e.touches[0].clientY + e.touches[1].clientY) / 2 }; var scale = d / lastTouchDist; var mouse = clientToView(mid.x, mid.y, rect); var newW = vb.w / scale; var newH = vb.h / scale; var minW = 40, maxW = 10000; if (newW < minW) { newW = minW; newH = newW * (vb.h / vb.w); } if (newW > maxW) { newW = maxW; newH = newW * (vb.h / vb.w); } vb.x = mouse.vx - ((mid.x - rect.left) * newW / rect.width); vb.y = mouse.vy - ((mid.y - rect.top) * newH / rect.height); vb.w = newW; vb.h = newH; lastTouchDist = d; lastTouchMid = mid; setViewBox(); } }, { passive: false });\n"
        "    svgEl.addEventListener('touchend', function(e){ if (e.touches.length === 0) { isDragging = false; lastTouchDist = null; lastTouchMid = null; } }, { passive: true });\n"
        "  })();\n"
        "  (function(){\n"
        "    // Object type filter UI - generate filter checkboxes from rendered .xodr-object elements\n"
"    function initObjectFilters(){\n"
"      // Prefer the object-filters slot inside the info panel; fall back to toolbar for backwards compatibility\n"
"      var target = document.getElementById('object-filters');\n"
"      var container = target || document.querySelector('.toolbar');\n"
"      if (!container || !mapLayer) return;\n"
"      // Avoid creating the generated UI twice\n"
"      if (document.getElementById('object-filters-generated')) return;\n"
"      var wrap = document.createElement('div');\n"
"      wrap.id = 'object-filters-generated';\n"
"      wrap.style.display = 'flex';\n"
        "      wrap.style.gap = '8px';\n"
        "      wrap.style.alignItems = 'center';\n"
        "      wrap.style.marginLeft = '8px';\n"
        "      wrap.style.flexWrap = 'wrap';\n"
"      wrap.style.maxWidth = '48%';\n"
"      wrap.style.fontSize = '13px';\n"
"      wrap.innerHTML = '<span style=\"color:#333;margin-right:6px;\">Objects:</span>';\n"
        "      var objs = Array.from(mapLayer.querySelectorAll('.xodr-object, .xodr-object-line'));\n"
        "      var types = {};\n"
        "      objs.forEach(function(el){ var t = el.getAttribute('data-obj-type') || el.getAttribute('data-elem-tag') || 'unknown'; types[t] = (types[t] || 0) + 1; });\n"
        "      var allTypes = Object.keys(types).sort();\n"
        "      if (allTypes.length === 0) {\n"
        "        var noObj = document.createElement('span'); noObj.style.color = '#666'; noObj.textContent = ' 无对象'; wrap.appendChild(noObj);\n"
        "      } else {\n"
"        var selectAll = document.createElement('button'); selectAll.className = 'tool'; selectAll.style.padding = '4px 8px'; selectAll.textContent = '全选'; selectAll.addEventListener('click', function(){ document.querySelectorAll('#object-filters input[type=checkbox]').forEach(function(cb){ cb.checked = true; cb.dispatchEvent(new Event('change')); }); });\n"
"        var clearAll = document.createElement('button'); clearAll.className = 'tool'; clearAll.style.padding = '4px 8px'; clearAll.textContent = '清除'; clearAll.addEventListener('click', function(){ document.querySelectorAll('#object-filters input[type=checkbox]').forEach(function(cb){ cb.checked = false; cb.dispatchEvent(new Event('change')); }); });\n"
"        wrap.appendChild(selectAll); wrap.appendChild(clearAll);\n"
        "        allTypes.forEach(function(t){ var label = document.createElement('label'); label.style.display = 'inline-flex'; label.style.alignItems = 'center'; label.style.gap = '6px'; label.style.marginLeft = '6px'; label.style.cursor = 'pointer'; var cb = document.createElement('input'); cb.type = 'checkbox'; cb.checked = true; cb.dataset.type = t; cb.addEventListener('change', function(){ var checked = cb.checked; objs.forEach(function(el){ var et = el.getAttribute('data-obj-type') || el.getAttribute('data-elem-tag') || 'unknown'; if (et === t) { el.style.display = checked ? '' : 'none'; } }); }); var span = document.createElement('span'); span.style.fontSize = '12px'; span.style.color = '#333'; span.textContent = t + ' (' + types[t] + ')'; label.appendChild(cb); label.appendChild(span); wrap.appendChild(label); });\n"
        "      }\n"
        "      container.appendChild(wrap);\n"
        "    }\n"
        "    window.addEventListener('load', function(){ setTimeout(initObjectFilters, 50); });\n"
        "    window.refreshObjectFilters = initObjectFilters;\n"
        "  })();\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )

    # write file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Minimal OpenDRIVE (.xodr) parser.")
    parser.add_argument("xodr", help="Path to the .xodr (OpenDRIVE) file")
    parser.add_argument("--list", action="store_true", help="Also list each road's id/name/length")
    parser.add_argument("--junctions", action="store_true", help="Count junctions (转弯口)")
    parser.add_argument("--list-junctions", action="store_true", help="List junctions with basic details")
    parser.add_argument("--visualize", nargs="?", const="visualization.html", help="Export simple HTML/SVG visualization (optional path)")
    parser.add_argument("--resample-road", help="Force higher sampling density for a specific road id (debug/fix)")
    args = parser.parse_args(argv)

    try:
        n = count_roads(args.xodr)
    except ET.ParseError as e:
        print(f"XML 解析错误: {e}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        print(f"文件未找到: {args.xodr}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(4)

    print(f"这个地图有{n}条路")

    if args.list:
        try:
            roads = list_roads(args.xodr)
            if roads:
                print()
                print("道路清单:")
                for idx, r in enumerate(roads, start=1):
                    rid = r.get("id", "")
                    name = r.get("name", "") or "<无名称>"
                    length = r.get("length", "") or "<无长度>"
                    print(f"{idx}. id={rid}  name={name}  length={length}")
            else:
                print("未找到任何道路元素以列出详细信息。")
        except Exception as e:
            print(f"列出道路时出错: {e}", file=sys.stderr)
            sys.exit(5)

    if args.junctions:
        try:
            jn = count_junctions(args.xodr)
            print(f"这个地图有{jn}个转弯口")
        except ET.ParseError as e:
            print(f"XML 解析错误（计算转弯口）: {e}", file=sys.stderr)
            sys.exit(6)
        except Exception as e:
            print(f"计算转弯口时出错: {e}", file=sys.stderr)
            sys.exit(7)

    if args.list_junctions:
        try:
            junctions = list_junctions(args.xodr)
            if junctions:
                print()
                print("转弯口（junction）清单:")
                for idx, j in enumerate(junctions, start=1):
                    jid = j.get("id", "")
                    name = j.get("name", "") or "<无名称>"
                    conn_count = j.get("connection_count", "0")
                    print(f"{idx}. id={jid}  name={name}  connections={conn_count}")
                    conns = j.get("connections", [])
                    if conns:
                        for cidx, cs in enumerate(conns, start=1):
                            print(f"    {cidx}) {cs}")
            else:
                print("未找到任何转弯口（junction）以列出详细信息。")
        except Exception as e:
            print(f"列出转弯口时出错: {e}", file=sys.stderr)
            sys.exit(8)

    if args.visualize is not None:
        out_path = args.visualize or "visualization.html"
        try:
            roads_geoms = extract_road_geometries(args.xodr, getattr(args, "resample_road", None))
            root = _load_root(args.xodr)
            objects_map = extract_objects_from_root(root)
            # 不计算 junction 标注，也不用于渲染 — 传入空字典
            write_visualization_html(out_path, roads_geoms, {}, objects_map)
            abs_path = os.path.abspath(out_path)
            print(f"已生成可视化文件: {abs_path}")
        except Exception as e:
            print(f"生成可视化时出错: {e}", file=sys.stderr)
            sys.exit(9)


if __name__ == "__main__":
    main()
