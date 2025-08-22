已为 opendrive_parser.py 添加地图可视化功能并生成了可视化文件。

完成的工作（概述）
- 在脚本中新增道路几何提取（支持 planView 下的 geometry -> line / arc）。
- 采样每段几何生成道路中心线折线（约 1m 分辨率，最多 200 样点）。
- 为每个 junction 选择代表点（使用第一个 connection 的 incomingRoad 的起点/终点）作为标记位置。
- 生成独立的 HTML 文件（内嵌 SVG），绘制道路灰色折线并在转弯口处用红点标注，文本标签显示 junction id。
- 增加 CLI 参数 --visualize [out.html]（默认为 visualization.html）。

验证
- 在 d:/OpenDrive 下对 test.xodr 执行了可视化：
  python opendrive_parser.py test.xodr --visualize visualization.html
  输出显示已生成: D:\OpenDrive\visualization.html
- 生成文件路径: D:\OpenDrive\visualization.html

如何查看可视化（在 Windows 11 上）
- 在命令行中打开（在当前工作目录或任意目录均可）:
  start "" "D:\OpenDrive\visualization.html"
- 或者在资源管理器中双击 D:\OpenDrive\visualization.html
- 也可以用浏览器打开该文件（File -> Open File -> 选择 visualization.html）。

脚本使用示例
- 统计道路:
    python opendrive_parser.py test.xodr
- 列出道路:
    python opendrive_parser.py test.xodr --list
- 统计转弯口:
    python opendrive_parser.py test.xodr --junctions
- 列出转弯口详情:
    python opendrive_parser.py test.xodr --list-junctions
- 生成可视化:
    python opendrive_parser.py test.xodr --visualize visualization.html

注意与扩展建议
- 当前可视化基于道路中心线的几何信息（planView.geometry 的 x,y,hdg,length 与 line/arc）。对于 spiral 或复杂几何，脚本会近似为直线。
- junction 标记采用每个 junction 第一个 connection 的 incomingRoad 起/终点；若需要更精确的交叉点位置，可基于 lane/link 或 road contactPoint 的更复杂逻辑进行计算并绘制。
- 可按需将 SVG 样式、颜色、缩放交互（JavaScript）、或导出为 GeoJSON 的功能加入。

文件已写入并测试成功：D:\OpenDrive\opendrive_parser.py 与 D:\OpenDrive\visualization.html