为什么看起来还是用 "python -m opendrive_parser ..."？

简短回答：
- 使用 `python -m opendrive_parser` 并不是在调用根目录的单文件 `opendrive_parser.py`，而是让 Python 以包的方式运行包内的 `__main__.py`（即 `opendrive_parser/__main__.py`）。这个入口文件正是我为重构后的包写的 CLI 适配层，它会调用拆分后的模块（geometry, parser, objects, junctions, viz 等）。所以你看到的命令其实是在运行“拆分后”的代码，而不是旧的单文件脚本。

细节说明：
- 两种不同的调用方式
  1. 包级运行（推荐，用于 CLI）：
     - 命令：`python -m opendrive_parser map/test.xodr --visualize out.html`
     - 含义：Python 在包级别查找 `opendrive_parser` 包并执行其 `__main__.py`。`__main__.py` 会导入并调用拆分模块的函数（例如 `extract_road_geometries`、`write_visualization_html` 等）。
     - 优点：不依赖文件系统中某个单独脚本位置，适合发布、安装和模块化维护。
  2. 直接导入模块（用于在其它 Python 代码中复用）：
     - 示例：
       ```python
       from opendrive_parser import extract_road_geometries, write_visualization_html, load_root

       roads_geoms = extract_road_geometries("map/test.xodr")
       root = load_root("map/test.xodr")
       # ... 处理数据 ...
       write_visualization_html("out.html", roads_geoms, junctions, objects_map)
       ```
     - 优点：可以把解析功能嵌入到其他程序或脚本中，更灵活。

关于根目录的 opendrive_parser.py（单文件脚本）：
- 当前仓库根下仍保留 `opendrive_parser.py`，它是旧版本/兼容拷贝。保留它不会影响包的运行（包运行使用的是 `opendrive_parser/__main__.py`）。
- 如果你希望只保留新的包结构以避免混淆，可以备份或删除该文件，例如：
  - 备份：在 Windows 命令行中执行 `rename opendrive_parser.py opendrive_parser.py.bak`
  - 删除：`del opendrive_parser.py`
  我可以为你执行备份/删除（会先备份再删除为建议的安全流程），但在做任何破坏性修改前会先征求你的确认。

结论与建议：
- 你看到的 `python -m opendrive_parser` 正是在运行被拆分后的包（不是旧单文件脚本）。
- 若要在代码中直接使用模块，请用 import（示例已给出）。
- 若要消除混淆，我可以备份并移除根目录的旧脚本，或把它保留作为兼容/历史文件。请告诉我你想如何处理旧脚本，或如果你希望我演示如何在另一个脚本中直接导入并调用包内函数，我可以创建一个示例脚本。
