"""
utils.py

小的工具函数，从原始单文件中抽取出来，供其他模块复用：
- _strip_ns(tag)
- _load_root(xodr_path)
- _find_elements_by_tag(root, tag)
- _get_child_by_tag(parent, tag)
"""
from __future__ import annotations
from typing import List, Optional
import xml.etree.ElementTree as ET


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


def _get_child_by_tag(parent: ET.Element, tag: str) -> Optional[ET.Element]:
    """Return first child whose stripped tag equals tag, else None."""
    for c in list(parent):
        if _strip_ns(c.tag) == tag:
            return c
    return None
