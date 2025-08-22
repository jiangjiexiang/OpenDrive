# OpenDrive 工具 (opendrive_parser.py)

一个轻量级的 OpenDRIVE (.xodr) 解析与可视化辅助脚本，便于快速浏览道路、转弯口和场景对象信息，并导出简单的 HTML/SVG 可视化结果。

## 主要功能
- 统计道路数量：count_roads
- 列出道路基本信息（id / name / length）
- 统计并列出转弯口（junctions）
- 提取道路中心线几何（支持 `line`、`arc`、简单 `spiral` 近似）
- 从场景中提取对象（object / signal / pole / trafficLight 等）并估算体积
- 导出独立的 HTML 可视化（SVG + 少量交互脚本）

> 当前可视化实现的几点说明
- 可视化时不会计算或渲染 junction 标注（默认向 `write_visualization_html` 传入空字典）。
- 渲染为“线状”的对象（例如车道标线类 object）默认使用黑色线条显示；某些特殊标签（例如 stop）仍使用红色以便区分。

## 依赖
- Python 3.7 及以上（测试通过版本：3.8+）
- 标准库：`xml.etree.ElementTree`, `math`, `argparse`, `json` 等（无需额外安装第三方包）

## 快速使用

仓库内有若干示例地图文件：`map/` 目录（例如 `map/test.xodr`）。

在项目根目录运行脚本示例：

- 统计道路：
```
python opendrive_parser.py map/test.xodr
# 输出示例：这个地图有<N>条路
```

- 列出道路信息：
```
python opendrive_parser.py map/test.xodr --list
```

- 统计转弯口数量：
```
python opendrive_parser.py map/test.xodr --junctions
```

- 列出转弯口详情：
```
python opendrive_parser.py map/test.xodr --list-junctions
```

- 导出可视化 HTML（生成 `out.html`）并在 Windows 上直接打开：
```
python opendrive_parser.py map/test.xodr --visualize out.html && start "" out.html
```
如果不提供输出文件名，默认会写 `visualization.html`。

- 针对某条道路强制更高采样（调试用）：
```
python opendrive_parser.py map/test.xodr --visualize out.html --resample-road 3
```

## 可视化行为说明
- 道路中心线与道路面（基于估算车道宽度）会在 SVG 中渲染，并支持点击查看道路信息。
- 对象（object/signal/pole 等）会根据其 `s`（纵向）和 `t`（横向）属性投影到道路中心线上显示。
- 线状对象（如车道线、停止线等）现在默认以黑色绘制（用户要求），特殊类型如 `stop` 使用红色。
- junction（转弯口）标注不会自动计算或渲染。如果需要恢复，请在源码中将 `_junction_marker_positions` 的调用恢复到 `--visualize` 分支。

## 常见问题与排查
- XML 解析错误：确认输入 `.xodr` 文件是有效 XML（脚本会捕获并报告 `ET.ParseError`）。
- 未生成 HTML：确认脚本有写入权限，并留意异常输出（脚本在遇到错误会以非零状态码退出）。
- 对象位置异常：部分 object 节点缺少 `s` 或 `t` 属性，脚本会尝试使用道路质心或地图中心作为回退位置。

## 开发与贡献
脚本为单文件实现（`opendrive_parser.py`），便于阅读和修改。欢迎提交 issue / PR 以改进：
- 恢复或改进 junction 标注策略
- 增强 spiral（clothoid）更精确的数值积分
- 增加更多 object 标签与样式规则
- 提高可视化的交互与图层控制

## 示例文件
包含若干示例地图：
- `map/test.xodr`
- `map/cross.xodr`
- `map/AITown.xodr`
- `map/Zhejiang.xodr`
- `map/chang.xodr`

## 许可
默认未指定专门许可证 —— 如需将本仓库用于开源/商业项目，请在仓库中添加合适的 LICENSE 文件或联系作者。

## 联系
仓库： https://github.com/jiangjiexiang/OpenDrive

