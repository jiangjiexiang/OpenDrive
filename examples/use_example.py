"""
示例脚本：演示如何以编程方式使用 opendrive_parser 包来加载 .xodr 文件，
提取道路几何与对象，并生成可视化 HTML 文件。

用法示例:
    python examples/use_example.py map/test.xodr out_example.html

该脚本演示了包公共 API 的最小调用流程，便于在外部代码中复用。
"""

import argparse
from opendrive_parser import (
    load_root,
    extract_road_geometries,
    extract_objects_from_root,
    junction_marker_positions,
    write_visualization_html,
)


def main():
    parser = argparse.ArgumentParser(description="示例：使用 opendrive_parser API 生成可视化 HTML")
    parser.add_argument("xodr", help="输入的 OpenDRIVE (.xodr) 文件路径")
    parser.add_argument("out_html", help="输出的 HTML 可视化文件路径")
    parser.add_argument("--resample-road", help="可选：指定需要高分辨率重采样的 road id", default=None)
    args = parser.parse_args()

    print(f"加载并处理：{args.xodr}")

    # 提取道路几何（包含中心线、多边形等）
    roads_geoms = extract_road_geometries(args.xodr, getattr(args, "resample_road", None))
    print(f"提取到 {len(roads_geoms)} 条道路几何（roads_geoms 字典）")

    # 读取 XML 根节点（某些 API 需要）
    root = load_root(args.xodr)

    # 提取场景对象（如 sign, signal, object 等，按 road 分组）
    objects_map = extract_objects_from_root(root)
    print(f"提取到对象分组，示例 keys: {list(objects_map.keys())[:5]}")

    # 计算 junction 标注（本示例会传入给 viz；如果不希望渲染 junction，可传空 dict）
    junctions = junction_marker_positions(root, roads_geoms)

    # 生成可视化 HTML（write_visualization_html 保持函数签名兼容）
    write_visualization_html(args.out_html, roads_geoms, junctions, objects_map)
    print(f"已生成可视化文件: {args.out_html}")


if __name__ == "__main__":
    main()
