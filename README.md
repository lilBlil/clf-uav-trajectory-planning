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
├── requirements.txt
├── 01.md, 02a.md, 02b.md, 03.md # 论文素材(引言 / 环境 / CLF / 算法)
├── src/
│   ├── config.py                # 全部配置与参数(工程假设)
│   ├── environment.py           # 城市环境体素化 + 可行集 + 测地距离场
│   ├── clf.py                   # CLF 地图构建(SNR/容量 + TDOA CRLB)
│   ├── marl_env.py              # Dec-POMDP 多无人机环境(状态/观测/奖励)
│   ├── mappo.py                 # MAPPO 训练器(GAE + clipped surrogate)
│   ├── networks.py              # NumPy MLP + Adam + masked softmax
│   ├── utils.py                 # 几何工具(box / LOS 判定)
│   └── plotting.py              # 可视化
└── experiments/
    └── run_experiment.py        # 端到端流程(环境→CLF→训练→出图)
```

## 已知问题 / 进行中

- 稀疏奖励下 MAPPO 探索不足,远距离起点时策略可能过早收敛于悬停。改进方向:增强熵正则 / 起点随机化课程。
