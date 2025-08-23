#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package CLI entrypoint for the refactored opendrive_parser package.

使用方法:
    python -m opendrive_parser path/to/file.xodr [--list] [--junctions] [--list-junctions] [--visualize [out.html]] [--resample-road ROAD_ID]

该模块调用拆分后的包内函数（geometry, parser, objects, junctions, viz），
从而不再依赖顶层的 opendrive_parser.py 单文件脚本。
"""
from __future__ import annotations
import sys
import argparse
import os
import xml.etree.ElementTree as ET

from . import (
    count_roads,
    list_roads,
    count_junctions,
    list_junctions,
    load_root,
    extract_road_geometries,
    extract_objects_from_root,
    junction_marker_positions,
    write_visualization_html,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Minimal OpenDRIVE (.xodr) parser (package entrypoint).")
    parser.add_argument("xodr", help="Path to the .xodr (OpenDRIVE) file")
    parser.add_argument("--list", action="store_true", help="Also list each road's id/name/length")
    parser.add_argument("--junctions", action="store_true", help="Count junctions (转弯口)")
    parser.add_argument("--list-junctions", action="store_true", help="List junctions with basic details")
    parser.add_argument("--visualize", nargs="?", const="visualization.html", help="Export simple HTML/SVG visualization (optional path)")
    parser.add_argument("--resample-road", help="Force higher sampling density for a specific road id (debug/fix)")
    args = parser.parse_args(argv)

    # Basic validation/parsing errors handled similarly to the original script
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
            # 使用拆分后的 geometry 提取道路几何、objects 提取对象、junctions 计算标注
            roads_geoms = extract_road_geometries(args.xodr, getattr(args, "resample_road", None))
            root = load_root(args.xodr)
            objects_map = extract_objects_from_root(root)
            junctions = junction_marker_positions(root, roads_geoms)
            write_visualization_html(out_path, roads_geoms, junctions, objects_map)
            abs_path = os.path.abspath(out_path)
            print(f"已生成可视化文件: {abs_path}")
        except Exception as e:
            print(f"生成可视化时出错: {e}", file=sys.stderr)
            sys.exit(9)


if __name__ == "__main__":
    main()
