# Package entrypoint for the refactored opendrive parser.
# Exposes a compact API that mirrors the original single-file script.
from .parser import count_roads, list_roads, count_junctions, list_junctions, load_root
from .geometry import extract_road_geometries
from .objects import extract_objects_from_root
from .junctions import junction_marker_positions
from .viz import write_visualization_html

__all__ = [
    "count_roads",
    "list_roads",
    "count_junctions",
    "list_junctions",
    "load_root",
    "extract_road_geometries",
    "extract_objects_from_root",
    "junction_marker_positions",
    "write_visualization_html",
]
