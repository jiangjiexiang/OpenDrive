# OpenDrive Parser (重构版)

简要说明
- 这是对原单文件 `opendrive_parser.py` 的重构，逻辑被拆分为包 `opendrive_parser/` 中的多个模块（`parser`, `geometry`, `junctions`, `objects`, `viz`, `utils` 等）。
- 目标：保留原有功能（统计道路/转弯口、导出可视化），并提高模块化、可维护性与复用性。

快速开始（无需安装）
1. 列出道路数量与清单
   ```
   python -m opendrive_parser map/test.xodr --list
   ```
2. 统计或列出转弯口
   ```
   python -m opendrive_parser map/test.xodr --junctions
   python -m opendrive_parser map/test.xodr --list-junctions
   ```
3. 导出可视化（生成 HTML）
   ```
   python -m opendrive_parser map/test.xodr --visualize out.html
   ```
   说明：
   - 可视化会生成一个 standalone 的 HTML 文件（内嵌 SVG + JS），默认路径 `visualization.html` 或自定义路径 `out.html`。
   - 按用户要求，当前 `opendrive_parser/viz.py` 已忽略 `junction` 标注参数（即默认不渲染转弯口标记）。如果需要恢复 junction 渲染，可编辑 `opendrive_parser/viz.py`（或在调用端传入非空值并移除内部忽略逻辑）。

包 API（可被其他脚本 import 并复用）
- 在 Python 中直接使用：
  ```py
  import opendrive_parser as od
  root = od.load_root("map/test.xodr")
  roads = od.extract_road_geometries("map/test.xodr")
  od.write_visualization_html("out.html", roads, {}, objects_map=None)
  ```
- 暴露的函数（示例）：
  - count_roads(xodr_path)
  - list_roads(xodr_path)
  - count_junctions(xodr_path)
  - list_junctions(xodr_path)
  - load_root(xodr_path)
  - extract_road_geometries(xodr_path, force_high_res_road=None)
  - extract_objects_from_root(root)
  - junction_marker_positions(root, roads_geoms)
  - write_visualization_html(out_path, roads_geoms, junction_markers, objects_map=None)

主要文件说明（快速索引）
- opendrive_parser/
  - __main__.py — 包级 CLI 入口（使用 `python -m opendrive_parser`）
  - __init__.py — 包的公共 API 导出
  - parser.py — 高层解析工具（load_root、count/list）
  - geometry.py — 道路几何提取与采样（line/arc/spiral），包含后处理（去尖点/插点/平滑）
  - junctions.py — 计算转弯口标注位置（可选用于渲染）
  - objects.py — 提取场景对象并估算体积/面积
  - viz.py — 生成 standalone HTML+SVG 的可视化（当前实现忽略 junction 标注以满足用户要求）
  - utils.py — 小工具函数（namespace 处理、XML 查询等）
- 根目录
  - map/ — 示例/测试地图文件（.xodr）
  - out.html — 由 CLI 生成的可视化示例文件（如果已生成）
  - opendrive_parserdebug.py — 保留的单文件脚本备份（历史兼容）

注意事项与建议
- 若希望在系统其他位置全局可用，可添加 packaging（pyproject.toml / setup.cfg）并使用 `pip install -e .` 以进行可编辑安装。
- 若想恢复或启用 junction 渲染：
  - 方法一：编辑 `opendrive_parser/viz.py`，移除忽略 `junction_markers` 的那行（当前实现内部覆盖为 `{}`）。
  - 方法二：在调用端传入合理的 `junction_markers` 并确保 viz.py 不覆盖该参数。
- 如果需要，我可以：
  - 将根目录的旧脚本备份并删除（避免混淆），或重命名为 `opendrive_parser_legacy.py`（需你确认）。
  - 创建示例脚本 `use_example.py` 展示如何以编程方式使用包 API。
  - 增加简单的单元测试（geometry / junctions / objects）。
