"""
junctions.py

转弯口（junction）标注位置计算。
提供：
- junction_marker_positions(root: ET.Element, roads_geoms: Dict[str, Dict]) -> Dict[str, Tuple[float, float, str, List[str]]]
"""
from __future__ import annotations
from typing import List, Tuple, Dict
import math
import xml.etree.ElementTree as ET

from .utils import _strip_ns, _find_elements_by_tag


def _proj_point_to_seg(px, py, ax, ay, bx, by):
    """Project point P onto segment AB; return (cx,cy,dist) where C is closest point on segment."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    if denom == 0:
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
    # iterate segments of pa and project endpoints of pb
    for i in range(len(pa) - 1):
        ax, ay = pa[i]
        bx, by = pa[i + 1]
        # project endpoints of pb onto segment AB
        for (px, py) in (pb[0],):
            cx, cy, d = _proj_point_to_seg(px, py, ax, ay, bx, by)
            if d < best[2]:
                best = (cx, cy, d)
        for j in range(len(pb) - 1):
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


def junction_marker_positions(root: ET.Element, roads_geoms: Dict[str, Dict]) -> Dict[str, Tuple[float, float, str, List[str]]]:
    """
    For each junction element, compute a robust representative coordinate, name and list of associated incoming road ids.

    Returns a dict mapping junction id -> (x, y, name, [incoming_ids...])
    """
    markers: Dict[str, Tuple[float, float, str, List[str]]] = {}
    j_elems = _find_elements_by_tag(root, "junction")
    for j in j_elems:
        jid = j.attrib.get("id", "")
        jname = j.attrib.get("name", "") or ""
        endpoint_pts = []
        incoming_list: List[str] = []
        for c in list(j):
            if _strip_ns(c.tag) != "connection":
                continue
            inc = c.attrib.get("incomingRoad")
            contact = c.attrib.get("contactPoint", "start")
            if not inc:
                continue
            incoming_list.append(inc)
            if inc in roads_geoms:
                poly = roads_geoms[inc].get("poly", [])
                if not poly:
                    continue
                if contact == "end":
                    endpoint_pts.append(poly[-1])
                else:
                    endpoint_pts.append(poly[0])

        # remove duplicates while preserving order
        seen = set()
        incoming_unique: List[str] = []
        for r in incoming_list:
            if r not in seen:
                incoming_unique.append(r)
                seen.add(r)

        # collect pairwise closest points between incoming polylines
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
            avg_x = sum(p[0] for p in endpoint_pts) / len(endpoint_pts)
            avg_y = sum(p[1] for p in endpoint_pts) / len(endpoint_pts)
            chosen_point = (avg_x, avg_y)
        else:
            for inc in incoming_unique:
                if inc in roads_geoms:
                    p = roads_geoms[inc].get("poly", [])
                    if p:
                        chosen_point = p[0]
                        break

        if chosen_point is None:
            continue

        markers[jid] = (chosen_point[0], chosen_point[1], jname, incoming_unique)
    return markers
