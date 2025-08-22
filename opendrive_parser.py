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


def extract_road_geometries(xodr_path: str) -> Dict[str, Dict]:
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

            # choose sampling resolution: ~1m or at most 200 samples
            samples = max(2, min(int(math.ceil(length / 1.0)) + 1, 200))

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
                # unsupported type (spiral etc) -> approximate as straight line
                for i in range(samples):
                    s = (i / (samples - 1)) * length
                    xi = x0 + s * math.cos(hdg)
                    yi = y0 + s * math.sin(hdg)
                    poly.append((xi, yi))

        # Optionally, collapse consecutive duplicate points
        collapsed: List[Tuple[float, float]] = []
        last = None
        for p in poly:
            if last is None or (abs(p[0] - last[0]) > 1e-6 or abs(p[1] - last[1]) > 1e-6):
                collapsed.append(p)
                last = p
        wl, wr = _compute_road_widths(r)
        roads_out[rid] = {"name": rname, "poly": collapsed, "width_left": wl, "width_right": wr}

    return roads_out


def _junction_marker_positions(root: ET.Element, roads_geoms: Dict[str, Dict]) -> Dict[str, Tuple[float, float, str, str]]:
    """
    For each junction element, pick a representative coordinate, name and incoming road id to mark it on the map.

    Strategy:
    - For each <junction>, look at its first <connection> child.
    - Use the connection's 'incomingRoad' and 'contactPoint' attributes:
        - if contactPoint == 'start' -> use the first point of that incoming road
        - if contactPoint == 'end'   -> use the last point of that incoming road
    - If road geometry isn't available, skip that junction.
    Returns dict mapping junction id -> (x, y, name, incomingRoad)
    """
    markers: Dict[str, Tuple[float, float, str, str]] = {}
    j_elems = _find_elements_by_tag(root, "junction")
    for j in j_elems:
        jid = j.attrib.get("id", "")
        jname = j.attrib.get("name", "") or ""
        # find first connection child
        con = None
        for c in list(j):
            if _strip_ns(c.tag) == "connection":
                con = c
                break
        if con is None:
            continue
        incoming = con.attrib.get("incomingRoad")
        contact = con.attrib.get("contactPoint", "start")
        if incoming and incoming in roads_geoms:
            poly = roads_geoms[incoming]["poly"]
            if not poly:
                continue
            if contact == "end":
                x, y = poly[-1]
            else:
                x, y = poly[0]
            markers[jid] = (x, y, jname, incoming)
    return markers


def write_visualization_html(out_path: str, roads_geoms: Dict[str, Dict], junction_markers: Dict[str, Tuple[float, float, str]]):
    """
    Produce a standalone HTML file with an SVG showing road areas (polygons), centerlines and junction markers.

    Uses per-road 'width_left' and 'width_right' (meters) when available to build a simple road polygon by
    offsetting the centerline left/right. Falls back to default width when missing.
    """
    # collect all points
    all_x = []
    all_y = []
    for r in roads_geoms.values():
        for x, y in r.get("poly", []):
            all_x.append(x)
            all_y.append(y)
        # if width fields included, they won't affect bounds directly
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
    width = 600
    height = 600
    margin = 20
    dx = maxx - minx
    dy = maxy - miny
    if dx == 0:
        dx = 1.0
    if dy == 0:
        dy = 1.0
    scale = min((width - 2 * margin) / dx, (height - 2 * margin) / dy)

    def transform(pt: Tuple[float, float]) -> Tuple[float, float]:
        x, y = pt
        tx = (x - minx) * scale + margin
        ty = height - margin - (y - miny) * scale  # flip Y for screen coords
        return tx, ty

    def compute_offsets(poly: List[Tuple[float, float]], left_w: float, right_w: float):
        """
        Given a centerline poly (list of (x,y)), compute left and right offset point lists.
        left_w/right_w are distances in meters to offset; function returns (left_pts, right_pts)
        where left_pts corresponds to left side following the poly order, and right_pts corresponds
        to right side following the poly order.
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
                x2, y2 = poly[i+1]
                tx = x2 - x
                ty = y2 - y
            elif i == n-1:
                x1, y1 = poly[i-1]
                tx = x - x1
                ty = y - y1
            else:
                x1, y1 = poly[i-1]
                x2, y2 = poly[i+1]
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

    svg_road_polygons = []
    svg_centerlines = []
    # build mapping road_id -> list of junctions (from junction_markers if they include incoming road id)
    road_junctions: Dict[str, List[Dict[str, str]]] = {}
    for jid, coord in junction_markers.items():
        if not coord:
            continue
        # coord may be (x,y,name) or (x,y,name,incoming)
        incoming = None
        jname = ""
        if len(coord) >= 4:
            incoming = coord[3]
            jname = coord[2] if coord[2] else ""
        elif len(coord) >= 3:
            jname = coord[2] if coord[2] else ""
        if incoming:
            road_junctions.setdefault(incoming, []).append({"id": jid, "name": jname})
    # draw roads as filled polygons using left/right widths
    for rid, rdata in roads_geoms.items():
        poly = rdata.get("poly", [])
        if not poly or len(poly) < 2:
            continue
        wl = float(rdata.get("width_left", 0.0) or 0.0)
        wr = float(rdata.get("width_right", 0.0) or 0.0)
        # if widths are zero, use a reasonable fallback (half-width 2.5m on each side)
        if wl <= 0 and wr <= 0:
            wl = wr = 2.5
        # compute offsets
        left_pts, right_pts = compute_offsets(poly, wl, wr)
        if left_pts and right_pts:
            # build polygon: left side forward, right side reversed to close
            poly_screen = [transform(p) for p in left_pts] + [transform(p) for p in reversed(right_pts)]
            pts_str = " ".join(f"{round(px,2)},{round(py,2)}" for px, py in poly_screen)
            title = f"road {rid} {rdata.get('name','')}"
            # lightly filled road with subtle stroke; include data attributes for interactivity
            svg_road_polygons.append(f'<polygon class="road-poly" data-road-id="{rid}" data-road-name="{rdata.get("name","")}" points="{pts_str}" fill="#ddd" stroke="#999" stroke-width="0.8"><title>{title}</title></polygon>')
        # also add centerline on top
        pts_center = [transform(p) for p in poly]
        center_str = " ".join(f"{round(px,2)},{round(py,2)}" for px, py in pts_center)
        svg_centerlines.append(f'<polyline class="road-line" data-road-id="{rid}" data-road-name="{rdata.get("name","")}" points="{center_str}" fill="none" stroke="#444" stroke-width="1"><title>center {rid}</title></polyline>')
        # optional thin lighter overlay to indicate lane boundaries could be added later

    svg_markers = []
    for jid, coord in junction_markers.items():
        if not coord or len(coord) < 2:
            continue
        x = coord[0]
        y = coord[1]
        name = coord[2] if len(coord) > 2 and coord[2] else f"J{jid}"
        tx, ty = transform((x, y))
        svg_markers.append(f'<circle class="junction" data-junction-id="{jid}" data-junction-name="{name}" cx="{round(tx,2)}" cy="{round(ty,2)}" r="4" fill="#c22" stroke="#800" stroke-width="1"/>')
        svg_markers.append(f'<text x="{round(tx+6,2)}" y="{round(ty+4,2)}" font-size="12" fill="#200">{name}</text>')

    # assemble HTML
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>OpenDRIVE visualization</title>
<style>
  body {{ font-family: sans-serif; }}
  svg {{ border: 1px solid #ccc; background: #fafafa; }}
</style>
</head>
<body>
<h3>OpenDRIVE visualization</h3>
<p>Roads shown as filled polygons (approximated from lane widths). Junctions marked in red.</p>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" version="1.1">
  <g>
    {"".join(svg_road_polygons)}
    {"".join(svg_centerlines)}
    {"".join(svg_markers)}
  </g>
</svg>
</body>
</html>
"""
    # write file
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Minimal OpenDRIVE (.xodr) parser.")
    parser.add_argument("xodr", help="Path to the .xodr (OpenDRIVE) file")
    parser.add_argument("--list", action="store_true", help="Also list each road's id/name/length")
    parser.add_argument("--junctions", action="store_true", help="Count junctions (转弯口)")
    parser.add_argument("--list-junctions", action="store_true", help="List junctions with basic details")
    parser.add_argument("--visualize", nargs="?", const="visualization.html", help="Export simple HTML/SVG visualization (optional path)")
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
            roads_geoms = extract_road_geometries(args.xodr)
            root = _load_root(args.xodr)
            markers = _junction_marker_positions(root, roads_geoms)
            write_visualization_html(out_path, roads_geoms, markers)
            abs_path = os.path.abspath(out_path)
            print(f"已生成可视化文件: {abs_path}")
        except Exception as e:
            print(f"生成可视化时出错: {e}", file=sys.stderr)
            sys.exit(9)


if __name__ == "__main__":
    main()
