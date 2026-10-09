# 论文风格诊断

范围：`paper/CL_paper_IEEE_English.tex`

本诊断只检查写作风格，不改写正文。

## 总体判断

这篇稿子整体已经比较接近 IEEE Communications Letters 的风格：结构紧凑、System Model 清楚、MARL 形式化也比较明确。剩下的问题主要是局部层面，不是结构性问题。

主要问题：

1. 少数表述比证据更大，尤其在摘要、引言和结论中。
2. 有一段结果讨论偏“评论化”，需要拆开或压缩。
3. 少量段落存在同义重复，主要集中在 LCF map、reward design 和结果讨论。
4. 核心术语总体一致，但 `accuracy / error / performance / quality / reliability / continuity` 之间的边界还不够严格。

## 分段诊断

| 位置 | 结论 | 问题判断 | 建议 |
|---|---|---|---|
| 摘要（L72） | Minor Edit | 简洁且符合 CL 风格，但最后的 `trajectory reliability` 偏虚，没有直接在结果中被操作化。 | 保持摘要紧凑，把收尾表述对齐到可测量结果，如路径长度、return 或无碰撞到达。 |
| 引言第 1 段（L80） | Keep | 场景交代清楚，技术背景也比较扎实。 | 基本不用改。 |
| 引言第 2 段（L82） | Keep | 问题切入明确，相关工作归纳也比较紧凑。 | 基本不用改。 |
| 引言第 3 段（L84） | Keep | RL 相关工作按功能分组，而不是机械罗列。 | 基本不用改。 |
| 引言第 4 段（L86） | Minor Edit | 研究空白清楚，但句子稍长，对 map-aided 方法的收益描述略泛。 | 删掉一两个从句，只保留“空白 -> 方案”的桥接。 |
| 引言第 5 段（L88） | Minor Edit | 收尾不错，但重复了摘要中的部分内容，而且 `trajectory reliability` 仍然偏泛。 | 收紧总结，尽量使用可测量术语。 |
| System Model 第 1 段（L94） | Keep | buildings、BS、坐标定义清晰。 | 基本不用改。 |
| System Model 第 2 段（L96-L112） | Keep | 从 voxel 定义到 feasibility 约束的推进顺畅。 | 基本不用改。 |
| System Model 第 3 段（L114-L118） | Keep | 轨迹定义简洁，易读。 | 基本不用改。 |
| LCF Map 第 1 段（L129-L135） | Keep | 从系统模型自然过渡到地图构建。 | 基本不用改。 |
| 通信模型（L138-L145） | Keep | 公式和变量解释都比较紧凑、明确。 | 基本不用改。 |
| TDOA 定位段（L148-L153） | Minor Edit | 技术上正确，但一个段落塞入的内容较多，接近 CL 的篇幅上限。 | 如果篇幅紧，可以再压缩；否则也可以保留。 |
| 归一化 fingerprint 段（L156-L175） | Minor Edit | 映射逻辑清楚，但阈值解释略重复，逆误差的说明偏长。 | 压缩阈值说明，每个指标保留一句干净解释即可。 |
| UAV Mobility Model（L179-L181） | Keep | 简短、功能明确。 | 基本不用改。 |
| Multi-UAV Planning Problem（L185-L213） | Minor Edit | 优化问题本身不错，但最后一句包含了实现细节，放在这里略显拥挤。 | 数学问题留在这里，mask/penalty 的实现说明可考虑后移。 |
| 算法引导句（L217） | Keep | 从问题到 MAPPO 的过渡自然。 | 基本不用改。 |
| State Space（L227-L236） | Keep | state 和 observation 的定义顺序合理。 | 基本不用改。 |
| Action Space（L239-L243） | Keep | 动作定义简洁，并且和状态设计一致。 | 基本不用改。 |
| Reward Design 引导+公式（L246-L255） | Minor Edit | reward 逻辑是清楚的，但 `Here, ...` 这类解释稍显重复。 | 合并对 `r(t)` 和 `X_n(t)` 的说明，减少重复句。 |
| 到达指示变量（L256-L264） | Keep | 必要且清晰的定义。 | 基本不用改。 |
| 能量归一化（L266-L268） | Keep | 简短且必要。 | 基本不用改。 |
| 碰撞惩罚（L270-L285） | Keep | 定义精确，也有技术依据。 | 基本不用改。 |
| MAPPO 过渡 + 算法 1（L287-L305） | Keep | 按照 guide 的顺序写，结构清楚。 | 基本不用改。 |
| 实验设置（L311-L343） | Keep | 符合 CL 风格，参数表也很紧凑。 | 基本不用改。 |
| 结果段 1：方法对比（L369） | Rewrite | 这是最明显的问题段。它把图像描述、baseline 定义、结果解释和元评论塞进一个长段落里，信息层级不清楚。`may facilitate` 也偏保守且空。最后一句关于实验“主要报告什么”没有必要。 | 拆成“现象句 + 依据句 + 解释句”，删掉多余评论，只保留一个机制解释。 |
| 结果段 2：训练动态（L381） | Keep | 很符合 CL 风格：现象 -> 对比 -> 原因。 | 基本不用改。 |
| 结果段 3：权衡曲线（L390） | Minor Edit | 简洁清楚，但最后一句稍微泛。 | 保留数值权衡，必要时可以删掉最后一句。 |
| 结论（L394） | Minor Edit | 篇幅合适，但 `trajectory reliability` 仍然偏泛。 | 收尾尽量映射到可观察结果。 |

## 英文补充

这篇稿子的整体结构已经符合 IEEE CL 的组织方式：

- 引言：是，已经形成了场景 -> 空白 -> 方法 -> 贡献的链条。
- System Model：是，且先定义再建模。
- Algorithm：是，state / action / reward 和算法框架都齐全。
- Results：大体是，但有一段需要压缩并明确区分现象和解释。
- Conclusion：是，但措辞略宽。

术语总体稳定。需要特别注意的漂移词是：

- `capacity`、`performance`、`quality`、`continuity`、`reliability`、`accuracy`、`error`

这些词不是不能用，而是不要随意互换，除非它们确实对应不同量。

## 总结

这篇稿子已经接近 IEEE Communications Letters 风格。最需要改的是局部减法：收紧几处表述、压缩一段结果、严格区分术语边界。
