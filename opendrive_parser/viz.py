"""
viz.py

将原始单文件脚本中的 write_visualization_html 提取到独立模块。
提供：
- write_visualization_html(out_path, roads_geoms, junction_markers, objects_map=None)
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import json
import os

# write_visualization_html 的实现基于原始单文件版本，保留了原有输出结构、样式与交互逻辑。
def write_visualization_html(out_path: str, roads_geoms: Dict[str, Dict], junction_markers: Dict[str, Tuple[float, float, str]], objects_map: Optional[Dict[str, List[Dict[str, str]]]] = None):
    """
    Produce a standalone, styled HTML file with an SVG showing road areas (polygons), centerlines and junction markers.

    Signature maintained from original script.
    """
    # User request: disable junction rendering and any junction-based styling.
    # We still accept the parameter for API compatibility, but ignore its contents.
    junction_markers = {}
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
            road_length = rdata.get("length", "N/A")
            if rid in road_junctions:
                junction_polys.append('<polygon class="road-poly junction-road" data-road-id="{rid}" data-road-name="{rname}" data-road-length="{length}" points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"><title>{title}</title></polygon>'.format(rid=rid, rname=rdata.get("name",""), length=road_length, pts=pts_str, title=title, fill=fill, stroke=stroke, sw=stroke_width_poly))
            else:
                normal_polys.append('<polygon class="road-poly" data-road-id="{rid}" data-road-name="{rname}" data-road-length="{length}" points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"><title>{title}</title></polygon>'.format(rid=rid, rname=rdata.get("name",""), length=road_length, pts=pts_str, title=title, fill=fill, stroke=stroke, sw=stroke_width_poly))

        pts_center = [transform(p) for p in poly]
        center_str = " ".join(f"{round(px,2)},{round(py,2)}" for px, py in pts_center)
        # centerline stroke darker variant
        center_stroke = "#333"
        if rid in road_junctions:
            center_stroke = "#8b3b00"
        else:
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

                    # Use screen-space vector math to avoid sign/flip issues introduced by the map->screen Y flip.
                    # We'll compute a small pair of nearby map points along the tangent, transform them to screen,
                    # derive the screen-space unit tangent, then construct the oriented line in screen pixels.
                    ux, uy = tangent
                    # center point in screen space (sx,sy) already computed above via transform(pos)
                    # compute two nearby map points (half-length along tangent), transform to screen
                    eps = map_len / 2.0
                    try:
                        pa_px = transform((pos[0] - ux * eps, pos[1] - uy * eps))
                        pb_px = transform((pos[0] + ux * eps, pos[1] + uy * eps))
                    except Exception:
                        # fallback: use center only
                        pa_px = (sx - eps * scale, sy)
                        pb_px = (sx + eps * scale, sy)

                    tx_px = pb_px[0] - pa_px[0]
                    ty_px = pb_px[1] - pa_px[1]
                    tlen_px = math.hypot(tx_px, ty_px)
                    if tlen_px > 1e-9:
                        ux_px, uy_px = tx_px / tlen_px, ty_px / tlen_px
                    else:
                        ux_px, uy_px = 1.0, 0.0

                    # half length in pixels for the final drawn line
                    half_len_px = (map_len / 2.0) * scale

                    # For regular line-like objects: draw along the tangent in screen space
                    p1_px = (sx - ux_px * half_len_px, sy - uy_px * half_len_px)
                    p2_px = (sx + ux_px * half_len_px, sy + uy_px * half_len_px)

                    # For stop-like elements, draw perpendicular to the road centerline (across the road)
                    try:
                        if elem_tag and "stop" in elem_tag:
                            # compute screen-space normal (rotate tangent by +90deg -> left normal)
                            nx_px, ny_px = -uy_px, ux_px
                            p1_px = (sx - nx_px * half_len_px, sy - ny_px * half_len_px)
                            p2_px = (sx + nx_px * half_len_px, sy + ny_px * half_len_px)
                    except Exception:
                        # keep tangent-based p1_px/p2_px on error
                        pass

                    # line style based on elem_tag / object type — 默认黑色（用户要求）
                    line_color = "#000000"
                    stroke_w = max(1.0, min(3.0, r_px / 2.0))
                    try:
                        if elem_tag:
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
"      <div id=\"search-controls\" class=\"section\">\n"
"        <div style=\"display:flex;flex-direction:column;gap:8px;\">\n"
"          <div style=\"display:flex;gap:8px;align-items:center;\">\n"
"            <input id=\"search-road-id\" placeholder=\"Road ID\" style=\"flex:1;padding:6px;border:1px solid #ddd;border-radius:6px;\" />\n"
"            <button id=\"search-road-btn\" class=\"tool\" style=\"padding:6px 10px;\">查询道路</button>\n"
"          </div>\n"
"          <div style=\"display:flex;gap:8px;align-items:center;\">\n"
"            <input id=\"search-obj-id\" placeholder=\"Object ID\" style=\"flex:1;padding:6px;border:1px solid #ddd;border-radius:6px;\" />\n"
"            <button id=\"search-obj-btn\" class=\"tool\" style=\"padding:6px 10px;\">查询物体</button>\n"
"          </div>\n"
"          <div id=\"search-status\" style=\"color:#666;font-size:12px;\"></div>\n"
"        </div>\n"
"      </div>\n"
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
        "  function showRoadInfo(id, name, length){\n"
        "    var length_val = parseFloat(length);\n"
        "    var length_str = !isNaN(length_val) ? '<b>Length:</b> ' + length_val.toFixed(2) + 'm<br/>' : '';\n"
        "    var html = '<b>Road ID:</b> ' + id + '<br/>' + '<b>Name:</b> ' + (name||'<无名称>') + '<br/>' + length_str;\n"
        "    var juncs = roadJunctions[id] || [];\n"
        "    if (juncs.length === 0) html += '<b>关联 junction:</b> 无'; else { html += '<b>关联 junction:</b><ul>'; for(var i=0;i<juncs.length;i++){ html += '<li>' + (juncs[i].name||('J'+juncs[i].id)) + ' (id=' + juncs[i].id + ')</li>'; } html += '</ul>'; }\n"
        "    infoContent.innerHTML = html;\n"
        "  }\n"
        "  // attach click handlers\n"
        "  if (mapLayer) {\n"
        "    mapLayer.querySelectorAll('.road-poly').forEach(function(el){\n"
        "      el.style.cursor = 'pointer';\n"
        "      el.addEventListener('click', function(evt){ evt.stopPropagation(); clearSelection(); el.classList.add('selected'); showRoadInfo(el.getAttribute('data-road-id'), el.getAttribute('data-road-name'), el.getAttribute('data-road-length')); });\n"
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
        "    window.addEventListener('load', function(){ setTimeout(initObjectFilters, 50); setTimeout(initSearchControls, 60); });\n"
        "    window.refreshObjectFilters = initObjectFilters;\n"
"    // initialize search controls for Road ID / Object ID\n"
"    function initSearchControls(){\n"
"      var sr = document.getElementById('search-road-id');\n"
"      var sb = document.getElementById('search-road-btn');\n"
"      var so = document.getElementById('search-obj-id');\n"
"      var sob = document.getElementById('search-obj-btn');\n"
"      var st = document.getElementById('search-status');\n"
"      if (!mapLayer) return;\n"
"      if (sb){\n"
"        sb.addEventListener('click', function(){\n"
"          var id = (sr && sr.value) ? sr.value.trim() : '';\n"
"          if (!id){ if (st) st.textContent = '请输入 road id'; return; }\n"
"          var el = mapLayer.querySelector('.road-poly[data-road-id=\"' + id + '\"]');\n"
"          var found = false;\n"
"          if (el){\n"
"            el.dispatchEvent(new MouseEvent('click', { bubbles: true }));\n"
"            found = true;\n"
"          } else {\n"
"            var el2 = mapLayer.querySelector('.road-line[data-road-id=\"' + id + '\"]');\n"
"            if (el2){ el2.dispatchEvent(new MouseEvent('click', { bubbles: true })); found = true; }\n"
"          }\n"
"          if (st) st.textContent = found ? ('已定位道路 id=' + id) : ('未找到道路 id=' + id);\n"
"        });\n"
"      }\n"
"      if (sr){ sr.addEventListener('keydown', function(e){ if (e.key === 'Enter') { if (sb) sb.click(); } }); }\n"
"      if (sob){\n"
"        sob.addEventListener('click', function(){\n"
"          var id = (so && so.value) ? so.value.trim() : '';\n"
"          if (!id){ if (st) st.textContent = '请输入 object id'; return; }\n"
"          var el = mapLayer.querySelector('.xodr-object[data-obj-id=\"' + id + '\"], .xodr-object-line[data-obj-id=\"' + id + '\"]');\n"
"          var found = false;\n"
"          if (el){ el.dispatchEvent(new MouseEvent('click', { bubbles: true })); found = true; }\n"
"          if (st) st.textContent = found ? ('已定位对象 id=' + id) : ('未找到对象 id=' + id);\n"
"        });\n"
"      }\n"
"      if (so){ so.addEventListener('keydown', function(e){ if (e.key === 'Enter') { if (sob) sob.click(); } }); }\n"
"    }\n"
"    window.initSearchControls = initSearchControls;\n"
"  })();\n"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )

    # write file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
