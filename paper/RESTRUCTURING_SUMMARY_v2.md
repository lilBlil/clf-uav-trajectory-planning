# 实验评估部分重构 — 最终版本

**日期：** 2026-07-06  
**修改文件：** `paper/CL_paper_IEEE_English_CL_draft_v2.tex`  
**状态：** ✅ 编译成功（6页）

---

## 图表重组方案（最终确认）

### 原有Fig 3（三联图）→ 拆分为两张独立图：

**新Fig 3: 代表性轨迹图（3D场景）**
- 单张3D可视化，展示所有4个方法的轨迹
- 包含：建筑物、基站位置、起点/终点标记、定位退化区域/CLF场
- 不同颜色/线型区分4种方法
- 当前资源：`outputs/method_comparison.png`（仅3个方法，需重新生成）
- **标签：** `fig:trajectories`

**新Fig 4: 定量对比（双子图）**
- **(a) 训练return曲线**：3个RL方法（CLF-MAPPO, GeoAgg-style RL, MAPPO-without-CLF）
  - ⚠️ TODO：目前为占位符，需要实现MAPPO-without-CLF后生成
  - 注意：需要处理不同奖励函数的可比性问题
  
- **(b) 物理指标柱状图**：4个方法的 L̄, C̄, arrival, collision 对比
  - ⚠️ TODO：目前为占位符，需要所有4个方法数据后生成

- **标签：** `fig:quant`

### 保持不变的图表：

**Fig 5: 训练动态（算法对比 + 分辨率扫描）**
- 子图(a)：MAPPO vs. IPPO vs. MAA2C 算法对比
- 子图(b)：体素分辨率扫描（Δr=20/15/10 m）
- 资源：`outputs/algo_return_only.png` + `outputs/resolution_return_only.png`
- **标签：** `fig:training`
- **状态：** ✅ 真实数据，保持原样

**Fig 6: Pareto前沿**
- w_L 权重扫描展示通信-定位权衡
- 资源：`outputs/pareto_clf.png`
- **标签：** `fig:pareto`
- **状态：** ✅ 真实数据，保持原样

---

## 四方法对比框架

| 方法 | 作用 | 代码 | 数据 |
|------|------|------|------|
| **CLF-MAPPO（提出方法）** | 离线CLF地图 + 相邻体素前瞻 + CTDE | ✅ | ✅ L=11.14m, C=20.74 Mbit/s |
| **CLF-weighted Dijkstra** | CLF表示 + 传统图搜索（非学习） | ❌ | ⚠️ TODO |
| **GeoAgg-style RL baseline** | 原始物理指标奖励（无CLF指纹） | ✅ | ✅ L=19.34m, C=20.69 Mbit/s |
| **MAPPO without CLF** | 消融：移除观测中的CLF通道 | ❌ | ⚠️ TODO |

**关键对比逻辑：**
- GeoAgg-style 测试：原始奖励是否足够？
- CLF-Dijkstra 测试：CLF表示本身能否通过图搜索解决任务？
- MAPPO-no-CLF 测试：多智能体强化学习能否在没有指纹先验的情况下学习？
- CLF-MAPPO：结合CLF前瞻与CTDE学习

**评估：** 所有方法在中性物理指标（C, L=√CRLB）上打分，不偏向任何设计

---

## 主要修改内容

### 1. 摘要（Abstract）
```latex
旧：71% over geodesic, 42% over baseline
新：~40% localization improvement in representative scenario
```
- 原因：单次运行数据，无多种子统计；测地线基线从核心对比中移除

### 2. 引言贡献3
明确量化CLF先验的作用，通过3个基线隔离：
- 原始指标RL（无CLF）
- CLF加权Dijkstra（无学习）  
- MAPPO无CLF（无指纹先验）

### 3. 实验设置 — 基线描述（详细）

#### (i) GeoAgg-style 原始指标RL基线
- 相同的MAPPO/CTDE训练设置
- 优化**原始**物理指标（C, 1/L）的对数势奖励，无CLF指纹
- 仅观测当前体素原始特征，无相邻体素前瞻
- **测试：** 直接物理奖励是否足够
- **真实数据：** ✅ `outputs/method_comparison.csv`

#### (ii) CLF-weighted Dijkstra
- 非学习图搜索，CLF感知边代价（Eq. \ref{eq:dijkstra_cost}）
- 多无人机使用优先级规划 + 预约表避碰
- **测试：** CLF表示本身能否通过传统搜索解决
- **真实数据：** ❌ TODO — 规划器未实现

#### (iii) MAPPO-without-CLF 消融
- 相同MAPPO/CTDE训练，相同到达/能量/碰撞奖励
- **移除** 观测中的指纹通道 F(v_n), F(V_n^adj)
- 保留位置、目标、动作掩码、局部占用
- **测试：** 多智能体RL能否在无指纹先验下学习
- **真实数据：** ❌ TODO — 观测侧消融未实现

#### (iv) CLF-MAPPO（提出）
- 结合CLF前瞻与CTDE学习
- **真实数据：** ✅ L=11.14m（比基线降低42%）

### 4. 参数表（Table I）更新
```latex
通信/定位发射功率：-10 dBm / 20 dBm（解耦）
噪声 PSD/系数：-174 dBm/Hz / 5 dB
```
- 匹配 `compare_three_methods.py` 实际运行参数

### 5. 表格 II：物理服务指标

| 方法 | 到达 | L̄ (m)↓ | C̄ (Mbit/s)↑ | L<10m | 碰撞 |
|------|------|---------|-------------|-------|------|
| CLF-MAPPO (提出) | 3/3 | **11.14** | 20.74 | **0.74** | 0 |
| CLF-weighted Dijkstra | -- | -- | -- | -- | -- |
| GeoAgg-style RL | 3/3 | 19.34 | 20.69 | 0.66 | 0 |
| MAPPO without CLF | -- | -- | -- | -- | -- |

- 真实数据：前2行从 `outputs/method_comparison.csv`
- TODO：填充剩余2行，添加多种子均值±标准差

### 6. 叙述重组

**轨迹行为（Fig. 3）**
- 解释死区回避机制：CLF感知方法绕行，原始指标/无CLF方法穿越
- 真实支持：仅CLF-MAPPO和GeoAgg-style RL
- TODO：验证Dijkstra/无CLF轨迹描述

**物理指标（Table II + Fig. 4b）**
- 报告42% L降低（11.1 vs 19.3 m），C匹配（~20.7 Mbit/s）
- 单次运行限制，无置信声明
- TODO：填充缺失行，生成柱状图

**训练稳定性（Fig. 4a）**
- 描述3个RL方法的收敛对比（TODO：生成实际曲线）
- 需要处理奖励可比性警告

**算法对比（Fig. 5a）保持不变**
- MAPPO > IPPO/MAA2C，确认集中式训练重要性

**分辨率扫描（Fig. 5b）保持不变**
- 粗网格收敛最快，体素分辨率权衡空间精度与样本复杂度

### 7. 结论
```latex
旧：geodesic + independent-learning baselines
新：~40% improvement relative to raw-metric baseline; 
    graph-search + no-CLF comparisons designed to isolate roles
```
- 移除"显著"/"稳健"声明（单场景，无多种子统计）

---

## 待实现工作

### 高优先级（完成4方法对比）

1. **CLF-weighted Dijkstra 规划器**
   - 实现 Eq. \ref{eq:dijkstra_cost} 边代价函数
   - 添加优先级规划 + 预约表（多无人机协调）
   - 集成到对比脚本
   - 估计时间：~2-3小时（确定性，无需训练）

2. **MAPPO-without-CLF 观测消融**
   - 在 `src/marl_env.py` 添加 `observation_mode` 配置标志
   - 在 `_self_features()` 和 `get_obs()` 中，当 mode="no_clf" 时归零 F 通道
   - 保持位置、目标、动作掩码、局部占用
   - 使用相同MAPPO设置训练
   - 估计时间：~1-2小时实现 + 30-60分钟训练/种子

3. **重新生成 Fig 3**（4方法轨迹）
   - 扩展 `scripts/compare_three_methods.py:plot_three_way()`
   - 输出单个3D场景到 `outputs/trajectories_4way.png`
   - 估计时间：~30分钟（方法1-2存在后）

4. **生成 Fig 4(a)**（3方法训练曲线）
   - 绘制CLF-MAPPO, GeoAgg-style RL, MAPPO-no-CLF 的team return
   - 警告：需处理不同奖励函数的可比性
   - 估计时间：~30分钟（无CLF数据存在后）

5. **生成 Fig 4(b)**（4方法指标柱状图）
   - 分组柱状图：4方法 × 4指标（L̄, C̄, arrival, collision）
   - 估计时间：~30分钟（所有数据存在后）

### 中优先级（严谨性）

6. **多种子训练**（所有4个方法，5+种子）
   - 报告均值±标准差，允许"显著"声明（需p值支持）
   - 估计时间：~2-4小时训练 × 4方法 × 5种子

7. **随机化城市地图**
   - 变化建筑物/基站位置，验证CLF优势的泛化性
   - 估计时间：~1天

---

## 技术注意事项

### 奖励可比性警告（Fig 4a）

三个RL方法优化**不同的奖励函数**：
- **CLF-MAPPO:** `r = w_C(f_comm-1) + w_L(f_loc-1) + arrival + energy + collision`
- **GeoAgg-style RL:** `r = log(C/γ_r) + (1/L)/scale + arrival + energy + collision`
- **MAPPO-no-CLF:** `r = arrival + energy + collision`（如果消融移除观测则无通信/定位项）

原始team return曲线不可直接对比。选项：
1. **绘制共享导航return**（仅arrival + energy + collision 分量）
2. **绘制中性物理指标**（如 arrival fraction vs. episodes）
3. **归一化最终return** 对比收敛速度
4. **接受不可比性** 框架为"收敛稳定性"而非"性能排名"

当前TODO已在图注中标记，实现时决定解决方案。

### 文件完整性

所有编辑保留**真实实验数据**：
- 未插入伪造数字
- 有数据支持的现有图表/表格保持不变
- 每个空缺透明标记TODO
- 编译验证通过（6页，无错误）

原始草稿备份为 `CL_paper_IEEE_English_CL_draft.tex`（仍在仓库中）

---

## 实施建议

### 路径A：仅论文重构（✅ 今日完成）
今日交付内容：
- 4方法实验框架，诚实TODO占位符
- 2/4方法真实数据（CLF-MAPPO, GeoAgg-style RL）
- LaTeX干净编译，结构就绪等待图表/数据填充
- 摘要/结论适当对冲（单场景，~40%改进）

**用例：** 提交给导师审阅实验逻辑，再运行昂贵计算

---

### 路径B：最小完整对比（推荐下一步）
实现上述项1-5，用单次运行数据填充所有4个方法：
- **时间估计：** ~1天（主要是Dijkstra + 观测消融实现/训练）
- **交付物：** 完整4方法对比，真实轨迹、指标、训练曲线
- **限制：** 仍为单次运行/方法（无多种子统计）

**用例：** 内部审阅完整论文草稿，再承诺多种子运行

---

### 路径C：完全严谨（如果准备投稿）
添加项6-7（多种子，随机化地图）：
- **时间估计：** +2-3天计算
- **交付物：** 统计稳健声明、置信区间、泛化证据
- **需要用于：** 将摘要声明从"~40%"加强为"显著降低"（带p值）

**用例：** 路径B结果内部接受后的最终投稿版本

---

## 关键文件

如果实施路径B/C，需修改的关键文件：
- `src/marl_env.py` — 添加 `observation_mode` 标志（镜像 `reward_mode` 模式）
- `src/environment.py` — Dijkstra基线可重用 `geodesic_field()` 逻辑，交换边权重
- `scripts/compare_three_methods.py` — 扩展到4方法，添加Dijkstra优先级规划
- `scripts/plotting.py` 或内联 — 分组柱状图 + 3方法训练曲线函数

所有TODO注释在 `.tex` 文件中带 `% TODO:` 前缀，可搜索。

---

**总结完毕。** 论文已准备好供导师审阅或实现缺失的基线。
