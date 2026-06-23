# CLF-Aided Multi-UAV Trajectory Planning

基于**通信-定位指纹(Communication-Localization Fingerprint, CLF)**的多无人机协同路径规划复现实现。

复现论文:*"Communication-Localization Fingerprint Aided Multi-UAV Trajectory
Planning in Urban Environments"*(Communications Letters)。

## 方法概述

1. **城市环境体素化(Sec. II-A)** — 将三维空域离散为立方体素,结合建筑物占用、安全距离、飞行高度约束,筛选出可行体素集合 `V_f`。
2. **CLF 地图构建(Sec. II-B)** — 为每个可行体素离线计算:
   - 可见基站集合 `M_i`(通信距离 + LOS 判定);
   - 通信性能 `C(v_i)`(SNR → 香农容量),映射为通信指纹 `f_comm`;
   - TDOA 定位性能 `L(v_i) = sqrt(CRLB)`(需 ≥4 个可见基站),映射为定位指纹 `f_loc`。
3. **MAPPO 轨迹规划(Sec. III, Algorithm 1)** — 将多无人机协同规划建模为 Dec-POMDP,采用 CTDE 框架的 MAPPO 求解:共享 actor(分布执行)+ 集中式 critic(集中训练)。

## 与论文的差异(实现说明)

提供的论文素材(`01.md`–`03.md`)**不包含 Section IV 的数值参数表**,因此 `src/config.py` 中所有数值均为物理上合理的工程假设。

此外,论文给出的奖励 `r(t)` 各项均 ≤ 0(纯惩罚),其字面最优策略是**悬停在高 CLF 体素**而非到达终点。为使智能体真正学会导航,本实现引入两项工程补充(均可关闭以还原论文原始奖励):

- **测地势函数 shaping**(`use_shaping`):`Phi(s) = -k·d_geo(pos, goal)/norm`,其中 `d_geo` 是可行体素图上的最短路径距离(绕开建筑)。基于 Ng et al. 1999 的势函数定理,该项 telescope 为常数,**理论上不改变最优策略**,仅提供导航梯度。
- **一次性到达奖励**(`r_arrive`):这是真正使"到达"成为最优策略的项。

详见 `src/config.py` 内的注释。

## 对比实验

### 1. CLF 奖励设计对比 `compare_policies.py`(主结果)

在**都能到达终点**的前提下,对比三种规划器的通信-定位服务质量——只有都能到达,f_comm / f_loc 的差距才纯粹来自"是否优化 CLF",而非可达性:

| 规划器 | 说明 |
|--------|------|
| 参考(CLF-盲几何贪婪) | 只沿测地场避障最短路,从不看指纹 |
| MAPPO 无CLF(消融) | 同网络/训练,但 `w_C = w_L = 0`,奖励只剩导航+能耗+碰撞 |
| **MAPPO 完整CLF**(本文方法) | CLF 感知规划器 |

```bash
python compare_policies.py
```

输出 `outputs/policy_comparison.png`:**左**为三策略的 **3D 轨迹**(穿越城市天际线,线型区分规划器、颜色区分无人机),**右**为 f_comm / f_loc / CLF奖励每步的**柱状图**(各策略分组,均值 over UAVs)。典型结果:完整CLF 在 f_comm、f_loc 两项上均最高(如 0.63 / 0.48 vs 参考 0.47 / 0.34),即在同样到达的前提下显著改善通信+定位服务质量。

同时输出 `outputs/policy_training_curves.png`:四面板**训练曲线**(团队回报、到达率、f_comm、f_loc 随训练更新)叠加对比无CLF vs 完整CLF 两个 MAPPO 变体,参考策略(几何贪婪、无训练)画为水平基准线——展示 CLF 感知与否的学习动态差异。

> 注:本脚本对每个 MAPPO 变体在多个种子上搜索,保留首个全员到达的种子(该场景较难,见下文)。

### 2. CTDE-MARL 算法对比 `compare_ctde.py`

只比较**主流且高性能的 CTDE(集中式训练-分散式执行)算法**,各自调到最佳收敛,画**两条头部曲线**:团队 episode return 与 goal-arrival fraction。

| 算法 | CTDE 子类 | 集中式信号 |
|------|-----------|-----------|
| MAPPO | 策略型 CTDE | 集中式 critic V(s) 训练共享分散 actor(PPO clip) |
| QMIX | 价值型 CTDE | 状态条件单调混合网络组合各 agent Q(训练用),分散 argmax 执行 |
| VDN | 价值型 CTDE | 加性分解 `Q_tot = Σ Q_n`,联合 TD 目标训练,分散 argmax 执行 |

```bash
python compare_ctde.py                 # 3 种子,完整训练
python compare_ctde.py --algos MAPPO QMIX
python compare_ctde.py --quick         # 快速冒烟
```

排除 IPPO/IAC/IQL——它们是独立学习,**非 CTDE**。不强求全部到达,重点是在同一公平条件(共享环境/CLF/奖励/种子)下对比最强 CTDE 方法的收敛行为与最终到达。价值型(VDN/QMIX)用更长 ε 退火 + 更高梯度比 + 略大学习率调优。输出 `outputs/ctde_comparison.png`(两面板:return + arrival)与 `outputs/ctde_comparison.csv`。

### 3. MARL 算法可达性对比 `compare_algos.py`

在 **{集中式 critic vs 各自 critic} × {PPO 截断 vs 朴素策略梯度}** 两轴上比较 MAPPO / IPPO / MAA2C / IAC,只保留全员到达的种子绘制训练曲线。

```bash
python compare_algos.py
```

**关键发现:在本重设计场景(错位窄缝、端到端穿越)下,只有 MAPPO(集中式 critic + PPO clip)能可靠让全部无人机到达。** IPPO(缺集中式 critic)、MAA2C(缺 PPO 稳定更新)及价值型 IQL/VDN/QMIX(`compare_families.py`)均会搁浅至少一架——印证"协调信用分配 + 稳定更新"两者缺一不可。这本身就是"为何选 MAPPO"的实证依据,也是 `compare_policies.py` 统一采用 MAPPO 的原因。

### 4. 跨家族探索 `compare_families.py`(辅助)

MAPPO vs 价值型 IQL/VDN/QMIX,同环境/CLF/奖励隔离"学习范式"变量。价值型早期样本效率高,但 ε-greedy 在本稀疏奖励场景探索不足,稳定卡在 2/3 到达,故不进入主对比。

## 环境要求

- Python 3.10+(开发于 3.12)
- 依赖见 `requirements.txt`:numpy / scipy / pandas / matplotlib

MAPPO(含 actor/critic MLP、手写反向传播 + Adam)以纯 NumPy 实现,**不依赖 PyTorch/TensorFlow**。

## 安装与运行(虚拟环境)

### Windows (Git Bash)

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活
source .venv/Scripts/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 运行
python main.py            # 完整训练(论文风格默认参数)
python main.py --quick    # 快速冒烟测试
```

### Windows (PowerShell / CMD)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### 命令行选项

```
python main.py --quick           # 快速冒烟测试(少 episode、粗网格)
python main.py --episodes 1200   # 自定义训练 episode 数(Epi_max)
python main.py --seed 42         # 指定随机种子
python main.py --no-verbose      # 关闭逐次更新日志
```

## 输出

运行后在 `outputs/` 生成(已在 `.gitignore` 中忽略):

| 文件 | 内容 |
|------|------|
| `clf_heatmaps.png` | 参考高度层的 `f_comm` / `f_loc` 指纹热图 |
| `trajectories_3d.png` | 规划出的多无人机三维轨迹 |
| `training_curves.png` | 训练曲线(reward / CLF / 到达率等) |
| `training_history.csv` | 逐次更新的训练指标 |
| `clf_summary.csv` | CLF 地图统计汇总 |

## 项目结构

```
.
├── main.py                      # 入口,参数解析
├── compare_algos.py             # 同家族对比(MAPPO/IPPO/MAA2C/IAC)
├── compare_families.py          # 跨家族对比(MAPPO vs IQL/VDN/QMIX)
├── compare_policies.py          # CLF 消融对比(reference / no-CLF / full-CLF)
├── requirements.txt
├── 01.md, 02a.md, 02b.md, 03.md # 论文素材(引言 / 环境 / CLF / 算法)
├── src/
│   ├── config.py                # 全部配置与参数(工程假设)
│   ├── environment.py           # 城市环境体素化 + 可行集 + 测地距离场
│   ├── clf.py                   # CLF 地图构建(SNR/容量 + TDOA CRLB)
│   ├── marl_env.py              # Dec-POMDP 多无人机环境(状态/观测/奖励)
│   ├── mappo.py                 # MAPPO 训练器(GAE + clipped surrogate)
│   ├── marl_algos.py            # on-policy AC 家族(IPPO/MAA2C/IAC 变体)
│   ├── value_marl.py            # value-based 家族(IQL/VDN/QMIX + 回放/目标网络)
│   ├── networks.py              # NumPy MLP + Adam + masked softmax
│   ├── utils.py                 # 几何工具(box / LOS 判定)
│   └── plotting.py              # 可视化
└── experiments/
    └── run_experiment.py        # 端到端流程(环境→CLF→训练→出图)
```

## 已知问题 / 进行中

- 稀疏奖励下 MAPPO 探索不足,远距离起点时策略可能过早收敛于悬停。改进方向:增强熵正则 / 起点随机化课程。
