下面是把 OpenDRIVE 的 (s, t) 投影到全局平面坐标 (x, y) 和航向 hdg 的完整算法与数学原理（中文，简洁版）：

总体思路（Frenet-frame）
1. 定位 geometry 段：在 road 的 planView 中找到 geometry 段 g 使 g.s <= s < g.s + g.length。设 ds = s - g.s（段内偏移）。
2. 在该 geometry 段上计算参考线点 P_ref(ds) = (x_ref, y_ref) 和切向航向 hdg_ref(ds)：
   - 对不同 geometry.type（line、arc、spiral）分别计算。
3. 依据 hdg_ref 计算左侧单位法向量 n̂ = (-sin(hdg_ref), cos(hdg_ref))（OpenDRIVE 中 t 通常为正向左侧；若你的数据定义相反，使用 -t）。
4. 最终坐标：
   x = x_ref + t * n̂.x
   y = y_ref + t * n̂.y
   hdg = hdg_ref

闭式公式（line 与 arc）
A. 直线（line）
- 已知 g.x, g.y, g.hdg, ds：
  x_ref = g.x + ds * cos(g.hdg)
  y_ref = g.y + ds * sin(g.hdg)
  hdg_ref = g.hdg

B. 圆弧（arc），常曲率 k（k = curvature，单位 1/m）
- hdg_ref = g.hdg + k * ds
- 若 k != 0：
  x_ref = g.x + ( sin(g.hdg + k*ds) - sin(g.hdg) ) / k
  y_ref = g.y + ( -cos(g.hdg + k*ds) + cos(g.hdg) ) / k
- 若 |k| 很小时（如 |k| < 1e-9）退化为直线公式以避免除零

C. 螺线（spiral / clothoid）
- 曲率 k(s) 通常线性变化（k(s)=k0 + a*s），对 x,y 没有简单初等闭式解（涉及 Fresnel 积分）。
- 推荐数值积分（RK4 或小步长累积）：
  微分方程： dx/ds = cos(hdg), dy/ds = sin(hdg), dhdg/ds = k(s)
  从 ds=0 数值积分到 ds 用足够小步长（例如 0.01–0.2m，按精度需求）累积 x,y,hdg。

数值与实现要点
- 单位：hdg 和 curvature 用弧度制；长度用米（与 XODR 保持一致）。
- t 方向：常见约定为正值朝左，确认你的 XODR 是否一致。
- k → 0 处理：当 |k| 很小使用直线近似以避免数值不稳定。
- 段边界：s 恰等于段端时清晰定义归属（可归入当前段末或下一段首，需一致）。
- 性能：若要大量查询，预先为每条 road 建索引（按 geometry.s 排序并用二分查找）可以加速定位段。
- 精度：spiral 用自适应或固定小步长积分；arc 用闭式公式是既快又精确。

伪代码（关键部分）
- find geometry g where g.s <= s < g.s+g.length
- ds = s - g.s
- switch g.type:
  - line: x_ref = g.x + ds*cos(g.hdg); y_ref = g.y + ds*sin(g.hdg); hdg_ref = g.hdg
  - arc: hdg_ref = g.hdg + k*ds; if |k|<EPS use line else use arc closed-form above
  - spiral: (x_ref,y_ref,hdg_ref) = numeric_integrate(g, ds)
- nx = -sin(hdg_ref); ny = cos(hdg_ref)
- x = x_ref + t * nx; y = y_ref + t * ny
- return (x, y, hdg_ref)

简短数值示例（arc）
- g.x=0, g.y=0, g.hdg=0, k=0.1, ds=10, t=2（左侧）
  hdg_ref = 1.0 rad
  x_ref ≈ (sin(1.0)-0)/0.1 ≈ 8.41470985
  y_ref ≈ (-cos(1.0)+1)/0.1 ≈ 4.59697694
  n̂ ≈ (-0.84147, 0.54030)
  x ≈ 8.41471 - 1.68294 = 6.73177
  y ≈ 4.59698 + 1.08060 = 5.67758
  hdg = 1.0 rad

测试与验证建议
- 编写单元测试覆盖：line, arc, spiral（包括 k→0 退化），以及不同 t 符号约定的情形。
- 可对照已知简单地图或制造已知参考点的数据验证实现。

结论
- 核心是先在参考曲线（geometry）上求出 P_ref 和 hdg_ref，再沿法向量施加横向偏移 t。
- line/arc 有解析公式，spiral 通常用数值积分（或 Fresnel 积分实现更精确）。
- 若你之后需要，我可以在仓库中实现并把 (x,y,hdg) 写入 stopLine 的解析结果（例如新增 fields "x","y","hdg"），并加入 unit tests；这一步是可选的，需你确认我可以修改代码并运行测试。