"""
objects.py

从 OpenDRIVE 根元素中提取场景对象（如 object, signal, pole 等）。
提供：
- extract_objects_from_root(root: ET.Element) -> Dict[str, List[Dict[str, str]]]
"""
from __future__ import annotations
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

from .utils import _strip_ns, _get_child_by_tag, _find_elements_by_tag


def extract_objects_from_root(root: ET.Element) -> Dict[str, List[Dict[str, str]]]:
    """
    Parse scene objects in the OpenDRIVE and return a mapping:
      road_id -> [ { id, name, elem_tag, type, subtype, s, t, zOffset, has_outline, raw_attrs }, ... ]
    """
    objects_map: Dict[str, List[Dict[str, str]]] = {}

    def _add_obj_to(road_key: str, obj_dict: Dict[str, str]):
        objects_map.setdefault(road_key, []).append(obj_dict)

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

                def _parse_num(val):
                    try:
                        return float(val) if val is not None and val != "" else None
                    except Exception:
                        return None

                l_val = _parse_num(child.attrib.get("length", "") or "")
                w_val = _parse_num(child.attrib.get("width", "") or "")
                h_val = _parse_num(child.attrib.get("height", "") or "")

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

    for elem in [e for e in root.iter() if _strip_ns(e.tag) in extra_tags]:
        try:
            tag = _strip_ns(elem.tag)
            oid = elem.attrib.get("id", "") or ""
            if oid and oid in seen_objs:
                continue
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

            def _parse_num(val):
                try:
                    return float(val) if val is not None and val != "" else None
                except Exception:
                    return None

            l_val = _parse_num(elem.attrib.get("length", "") or "")
            w_val = _parse_num(elem.attrib.get("width", "") or "")
            h_val = _parse_num(elem.attrib.get("height", "") or "")

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
