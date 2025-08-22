"""
parser.py

High-level parsing utilities extracted from the original single-file implementation.
Provides:
- load_root(xodr_path) -> ET.Element
- count_roads(xodr_path) -> int
- list_roads(xodr_path) -> List[Dict[str, str]]
- count_junctions(xodr_path) -> int
- list_junctions(xodr_path) -> List[Dict[str, str]]
"""
from __future__ import annotations
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

from .utils import _strip_ns, _load_root, _find_elements_by_tag


def load_root(xodr_path: str) -> ET.Element:
    """Parse XML and return root element. Raises ET.ParseError on bad XML."""
    return _load_root(xodr_path)


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
