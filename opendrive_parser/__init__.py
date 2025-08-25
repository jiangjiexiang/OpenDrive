"""
opendrive_parser
包入口与公共 API 导出模块。

此文件在包被导入时暴露一组与原始单文件脚本兼容的便捷函数，
将拆分到各模块（parser, geometry, objects, junctions, viz）的实现统一导出，方便外部直接使用。

对外导出（示例）：
- count_roads, list_roads, count_junctions, list_junctions, load_root
- extract_road_geometries
- extract_objects_from_root
- junction_marker_positions
- write_visualization_html

说明：保持与原脚本的函数签名兼容，以便平滑迁移与回退。
"""
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
