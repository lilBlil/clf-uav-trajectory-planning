# IEEE Communications Letters 写作风格指南

本指南基于 `paper/writing_refs` 中参考论文的写作方式整理，重点关注论文“如何写”，不总结论文技术内容。

## 0. 分析依据

### 参考论文中实际观察到的规律

已分析以下论文：

1. `Trajectory_Design_and_Link_Selection_in_UAV-Assisted_Hybrid_Satellite-Terrestrial_Network.pdf`
2. `Enhanced_RSS-Based_UAV_Localization_Via_Trajectory_and_Multi-Base_Stations.pdf`
3. `Joint_3-D_Deployment_and_Resource_Allocation_for_UAV-Assisted_Integrated_Communication_and_Localization.pdf`
4. `Energy-Efficient_Decision-Making_With_Belief_Structures_for_Dynamic_Path_Planning_in_UAV-Enabled_6G_Networks.pdf`
5. `Deployment_Optimization_of_Dual-Functional_UAVs_for_Integrated_Localization_and_Communication.pdf`
6. `Three-Dimensional-Map-Based_Trajectory_Design_in_UAV-Aided_Wireless_Localization_Systems.pdf`
7. `Jointly_Optimize_Throughput_and_Localization_Accuracy_UAV_Trajectory_Design_for_Multiuser_Integrated_Communication_and_Sensing.pdf`
8. `ISAC_From_the_Sky_UAV_Trajectory_Design_for_Joint_Communication_and_Target_Localization.pdf`
9. `GeoAgg-HSAC_an_RL-based_Framework_for_Trajectory_and_Resource_Optimization_in_Mountainous_UAV_Integrated_Localization_and_Communication_Networks.pdf`

其中 IEEE Communications Letters 风格的论文更强调篇幅压缩、贡献直接、问题定义清楚和仿真解释简洁。TWC、IoT Journal 等长文虽然章节更细、推导更多，但其 System Model、Algorithm 和 Results 的局部写法仍有参考价值。

### 我的写作推断

对于 IEEE Communications Letters，最稳妥的风格是“短而明确”：每一段只承担一个技术功能，每个 claim 都应能对应到模型、约束、指标、算法步骤或仿真结果。避免大背景铺陈、泛泛综述和宣传式表达。

## 1. Introduction

### 参考论文中实际观察到的规律

Introduction 通常遵循四步逻辑：

1. 交代应用场景和研究必要性。
   开头通常从具体无线网络场景切入，例如 UAV-assisted communication、localization、ISAC/ILAC、emergency coverage 或 satellite-terrestrial networking。先说明该场景的价值，再立即指出实际限制。

2. 按技术类别压缩相关工作。
   相关工作不是逐篇罗列，而是按方法或系统类型组织，例如 deterministic optimization、UAV trajectory design、localization metrics、map-aided modeling、GNN/RL methods 或 resource allocation。

3. 明确现有工作的不足。
   研究空白通常写成“现有假设”和“目标场景需求”之间的技术不匹配。例如，已有方法假设完整信息、简单障碍物、单功能 UAV、固定部署、单用户/单目标，或依赖高复杂度求解器。

4. 给出本文方案和贡献。
   最后一段通常先承接研究空白，然后说明本文考虑的系统、优化目标、核心方法和主要结果。Letters 中贡献列表通常很短；若使用 bullet，每条贡献一般以 `formulate`、`derive`、`propose`、`design`、`show` 等精确动词开头。

贡献表述通常包含：

- 模型贡献：考虑了什么系统、信道、不确定性、定位指标或资源耦合关系。
- 问题贡献：联合优化了什么目标和约束。
- 方法贡献：提出了什么变换、近似、学习框架或求解器。
- 结果贡献：在哪个指标上优于哪个 baseline。

### 我的写作推断

IEEE Communications Letters 的 Introduction 可控制在 3 到 5 段：

1. 背景和实际需求：3 到 5 句。
2. 相关工作和局限：4 到 6 句。
3. 研究空白和技术挑战：2 到 4 句。
4. 本文方法和贡献：4 到 7 句，或 2 到 3 条紧凑贡献。

不要写成领域综述。每一句都应把读者进一步引向本文实际解决的问题。

## 2. System Model

### 参考论文中实际观察到的规律

System Model 通常以直接的场景句开头，例如 “Consider a ... system/network, where ...”。随后尽早引用系统图，并依次定义集合、坐标、时间槽、用户、UAV、BS、目标和信道。

常见组织顺序为：

1. 场景和拓扑。
   定义节点、坐标、时间范围、slot index 和服务关系。

2. 假设。
   只写会影响公式的假设，例如固定高度、已知用户位置、移动模型、信道模型、LoS/NLoS 条件、带宽分配、能耗模型或定位测量类型。

3. 指标定义。
   定义通信速率、定位精度、CRB/D-optimality/GDOP、能耗或能效。每个指标前通常有一句说明为什么采用该指标。

4. 问题建模。
   在所有变量和指标定义完成后给出优化问题。优化问题之后立即用简短文字解释各约束含义。

常见写法：

- 符号尽量在第一次使用附近定义。
- 公式前有一句建模说明，公式后解释尚未定义的变量。
- 约束不会只靠公式表达，通常会在问题后用一句话解释。
- `For simplicity`、`Without loss of generality`、`Unless otherwise specified` 使用克制，只在确实消除歧义时使用。
- Letters 中不展开过长推导，除非推导直接服务于后续求解。

### 我的写作推断

应严格保持符号一致：

- 一个概念只用一个名称。不要在无区别的情况下交替使用 `UE`、`user`、`terminal`、`GT`。
- 一次性定义 `i in I`、`k in K`、`t in T` 后全文一致使用。
- 不要在 Algorithm 或 Results 中重复通信速率、定位指标或 reward 公式，引用已有公式编号即可。
- 每个优化问题后应给出紧凑的约束解释，例如 “constraint (x) enforces the UAV mobility constraint, (y) imposes the transmit-power budget, and (z) guarantees the localization requirement.”

## 3. Algorithm / MARL

### 参考论文中实际观察到的规律

Algorithm 部分通常先说明原问题为什么难解：mixed-integer variables、nonconvex constraints、high-dimensional state space、partial observability、dynamic topology 或 computational cost。随后先给主思路，再进入细节。

RL/MARL 部分常见顺序为：

1. 将问题转化为 MDP 或学习问题。
2. 定义 tuple，通常包括 state、action、reward 和 discount factor 或 transition setting。
3. state 只定义 agent 可获得的信息，而不是所有系统变量。
4. action 明确对应 agent 控制的决策变量。
5. reward 与原优化目标和约束挂钩。
6. 用 algorithm box 描述训练或执行流程。
7. 篇幅允许时补充复杂度、收敛条件或实现细节。

state/action/reward 的写法特点：

- state 定义简洁，通常采用枚举式表述。
- action 明确区分连续动作和离散动作。
- reward 不是泛泛的 performance score，而是与通信、定位、能耗、惩罚项等目标直接对应。
- Algorithm 段落避免重复推导前文公式，只说明 reward、payoff 或约束按前文公式计算。
- 训练细节具体，例如 episodes、learning rate、batch size、network layers、exploration policy 或 stopping rule。

### 我的写作推断

IEEE Communications Letters 的 MARL 部分建议：

- 第一段写清楚：“Problem P is difficult because ... Therefore, we use ...”
- 用 bullet 或紧凑段落一次性定义 `state`、`action`、`reward`。
- 如果是 multi-agent，明确区分 local observation、shared/global information，以及是否采用 centralized training decentralized execution。
- reward 只定义一次。后文 algorithm step 写 “compute the reward using (x)” 即可。
- Algorithm box 用来写流程，不用来承担长解释。解释放在 algorithm 前后。
- 除非有证明，不要把 learning-based 方法称为全局最优。更稳妥的动词是 `learns`、`approximates`、`obtains`、`updates`。

## 4. Experimental Results

### 参考论文中实际观察到的规律

Results 部分通常先说明仿真目的，再列 simulation environment、parameters 和 baselines。密集参数用表格呈现，重复设置用 `Unless otherwise specified` 处理。

图表讨论常采用以下顺序：

1. 结果现象。
   先说明图中显示了什么：收敛、上升/下降趋势、tradeoff、轨迹行为或 baseline 对比。

2. 定量比较。
   给出具体数值、百分比、相对增益、运行时间下降或方法排序。

3. 原因解释。
   用模型或算法机制解释现象，例如 LoS probability、resource reallocation、energy budget、trajectory proximity、state aggregation、uncertainty handling 或 constraint relaxation。

结果段落语气保守：

- 常用 `shows`、`demonstrates`、`indicates`、`can be attributed to`、`this is because` 连接现象和解释。
- baseline 首次出现时明确命名并定义缩写。
- 强结论会限定在具体仿真条件下，而不是泛化到所有场景。
- 部分论文也会承认 tradeoff，例如性能提升伴随复杂度增加，或约束变严格导致性能下降。

### 我的写作推断

建议每个 figure 或 subfigure group 用一个段落：

1. 第一句：指出图号和主要现象。
2. 第二句：给出关键定量比较。
3. 第三句：解释趋势原因。
4. 可选第四句：说明该结果在当前设置下的含义。

避免只写“proposed method is better”。读者需要知道在哪个条件下、好多少、为什么更好。

## 5. Conclusion

### 参考论文中实际观察到的规律

Conclusion 通常很短，偏回顾性：

1. 一句话重述研究问题。
2. 一到三句话总结模型或方法。
3. 一句话总结 numerical findings。
4. 长文中有时补充未来扩展方向。

语气事实化，不扩展到仿真未支持的 claim。

### 我的写作推断

IEEE Communications Letters 的 Conclusion 建议控制为一个紧凑段落。不要引入新背景、新符号、新 baseline 或未经验证的应用意义。好的 Conclusion 应像一段压缩记录：做了什么，用了什么方法，观察到什么结果。

## 6. 常见句式、连接方式和术语习惯

### 参考论文中实际观察到的功能句式

以下为概括后的功能句式，不是参考论文原句：

- 背景：`Recent advances in X enable Y in Z scenarios.`
- 问题压力：`However, achieving Y is challenging due to A and B.`
- 相关工作归纳：`Existing studies have investigated A, B, and C.`
- 研究空白：`These methods typically assume A, and therefore do not capture B.`
- 本文定位：`In this letter, we consider a ... system where ...`
- 优化目标：`We aim to maximize/minimize ... subject to ...`
- 问题建模：`The problem is formulated as ...`
- 难点说明：`The formulated problem is difficult to solve because ...`
- 方法引出：`To address this issue, we propose/design/derive ...`
- 算法概述：`The main idea is to ...`
- 图表描述：`Fig. X shows ...`
- 原因解释：`This is because ...` 或 `This can be attributed to ...`
- 约束解释：`Constraint (x) ensures/imposes/limits ...`
- 保守结论：`Numerical results show that ... under the considered setup.`

### 技术术语使用习惯

- 缩写首次出现时定义，之后统一使用：UAV、BS、LoS、NLoS、ISAC、ILAC、CRB、GDOP、MDP、MARL。
- 每个实体保持一个固定术语。技术一致性优先于同义词替换。
- 优先使用精确技术名词，而不是宽泛形容词，例如 `nonconvex constraint`、`mixed-integer variable`、`partial observability`、`LoS blockage`、`resource allocation`、`trajectory planning`、`localization accuracy`。
- `proposed method/scheme/algorithm/framework` 选择一种主要表述后尽量保持一致。
- `baseline`、`benchmark`、`existing method` 只有在比较角色明确时使用。

## 7. 应避免的表达

### 参考论文中通过对比观察到的规律

较强的参考段落通常避免宣传式 claim。它们很少使用没有支撑的 `revolutionary`、`groundbreaking`、`remarkable`、`perfect` 等表达。性能提升一般绑定具体指标、图表和 baseline。

### 我的写作推断

应避免 AI 化、低信息密度或过度宣传的表达：

- `This paper provides a comprehensive and robust solution`，除非确实验证了 comprehensive 和 robust。
- `Significantly improves performance`，但没有指标、baseline 和条件。
- `Greatly enhances system efficiency`，但没有数值证据。
- `It is obvious that`，尤其当需要模型解释时。
- `Due to the rapid development of technology` 这类泛泛开头。
- 在 Introduction、System Model 和 Algorithm 中反复重复同一研究动机。
- 用文字重复公式本身，却没有增加解释。
- 为避免重复而随意替换核心术语。
- 将 heuristic、approximate 或 finite-simulation training 的 learning 方法过度声称为 globally optimal。

## 8. IEEE CL 写作检查表

投稿式改写前检查：

- Introduction 是否形成清晰链条：scenario -> related work -> gap -> contribution。
- 每条 contribution 是否是一个具体技术动作和对象。
- System Model 是否先定义集合、变量、假设和指标，再给优化问题。
- 每个公式前是否有目的句，公式后是否解释符号。
- MARL 部分是否只定义一次 state、action、reward 和 training flow。
- Results 段落是否遵循现象 -> 定量比较 -> 原因解释。
- Conclusion 是否没有引入新 claim。
- 全文术语是否一致。
- 所有 claim 是否保守，并且能对应证据。
