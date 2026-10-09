# Manuscript Style Audit

Scope: `paper/CL_paper_IEEE_English.tex`

This audit checks style only. It does not rewrite the manuscript.

## Executive Summary

Overall, the manuscript is already close to IEEE Communications Letters style: the structure is compact, the system model is explicit, and the MARL formulation is technically clear. The remaining issues are mostly local rather than structural.

Main issues:

1. A few claims are broader than the evidence shown, especially in the abstract, introduction, and conclusion.
2. One results paragraph is too commentary-heavy and should be split or trimmed.
3. Some paragraphs repeat the same meaning in slightly different words, especially around the LCF map, reward design, and result discussion.
4. Core terminology is mostly consistent, but `accuracy / error / performance / quality / reliability / continuity` sometimes overlap without a strict boundary.

## Section-by-Section Audit

| Location | Label | Diagnosis | Suggested direction |
|---|---|---|---|
| Abstract (L72) | Minor Edit | Concise and aligned with CL, but the final claim `trajectory reliability` is vague and not directly operationalized in the results. | Keep the abstract tight and tie the closing claim to measurable outputs such as path length, return, or collision-free arrival. |
| Introduction P1 (L80) | Keep | Clear scene-setting and technically grounded. | No major change needed. |
| Introduction P2 (L82) | Keep | Good problem framing and compact related-work grouping. | No major change needed. |
| Introduction P3 (L84) | Keep | The RL-related literature is grouped by function rather than listed mechanically. | No major change needed. |
| Introduction P4 (L86) | Minor Edit | The gap is clear, but the paragraph is slightly long and the benefit of map-aided methods is described in a generic way. | Trim one or two clauses and keep only the gap-to-solution bridge. |
| Introduction P5 (L88) | Minor Edit | Good closing paragraph, but it repeats some abstract content and ends with a broad claim about `trajectory reliability`. | Keep the summary sharper and prefer measurable terms. |
| System Model P1 (L94) | Keep | Dense but clear definitions of buildings, BSs, and coordinates. | No major change needed. |
| System Model P2 (L96-L112) | Keep | Good progression from voxel definition to feasibility constraints. | No major change needed. |
| System Model P3 (L114-L118) | Keep | Trajectory definition is compact and easy to follow. | No major change needed. |
| LCF Map P1 (L129-L135) | Keep | Clean transition into the map construction idea. | No major change needed. |
| Communication Model (L138-L145) | Keep | Formula plus variable explanation is concise and technically explicit. | No major change needed. |
| TDOA Localization P1 (L148-L153) | Minor Edit | Technically correct but dense; the long formula explanation is close to the CL limit for one paragraph. | Split or shorten only if page budget is tight; otherwise keep. |
| Normalized Fingerprints P1 (L156-L175) | Minor Edit | The mapping is clear, but the threshold explanation is slightly repetitive and the inverse-error explanation is a bit wordy. | Tighten the explanation of the normalization threshold and keep one clean sentence per metric. |
| UAV Mobility Model (L179-L181) | Keep | Short and functional. | No major change needed. |
| Multi-UAV Planning Problem (L185-L213) | Minor Edit | The optimization statement is strong, but the last sentence adds implementation detail that could be moved to the algorithm section. | Keep the mathematical problem here and move masking/penalty implementation detail later if space is tight. |
| Algorithm lead-in (L217) | Keep | Good one-sentence bridge from problem to MAPPO. | No major change needed. |
| State Space (L227-L236) | Keep | State and observation are defined cleanly and in the right order. | No major change needed. |
| Action Space (L239-L243) | Keep | Action definition is concise and consistent with the state design. | No major change needed. |
| Reward Design intro + formula (L246-L255) | Minor Edit | The reward logic is clear, but the paragraph repeats `Here, ...` style explanations and could be slightly tighter. | Merge the repeated explanation of `r(t)` and `X_n(t)` into fewer sentences. |
| Arrival indicator (L256-L264) | Keep | Clear and necessary definition. | No major change needed. |
| Energy normalization (L266-L268) | Keep | Short, exact, and useful. | No major change needed. |
| Collision penalty (L270-L285) | Keep | The penalty definition is precise and technically motivated. | No major change needed. |
| MAPPO transition + Algorithm 1 (L287-L305) | Keep | The flow is concise and follows the guide: state/action/reward -> algorithm box. | No major change needed. |
| Experimental setup (L311-L343) | Keep | Good CL-style setup paragraph with a compact parameter table. | No major change needed. |
| Results paragraph 1: method comparison (L369) | Rewrite | This is the main style weak point. It mixes figure description, baseline definitions, result interpretation, and meta-commentary in one long paragraph. The phrase `may facilitate` is tentative, and the final sentence about what the experiments `primarily report` is unnecessary. | Split into a result sentence, a grounded explanation sentence, and cut the meta-commentary. Keep only one mechanism-based explanation. |
| Results paragraph 2: training dynamics (L381) | Keep | Strong CL-style result paragraph: phenomenon -> comparison -> explanation. | No major change needed. |
| Results paragraph 3: tradeoff curve (L390) | Minor Edit | The paragraph is short and clear, but the last sentence is slightly generic. | Keep the quantitative tradeoff and consider dropping the final sentence if space is tight. |
| Conclusion (L394) | Minor Edit | Compact and correctly scoped, but `trajectory reliability` is again too broad. | Rephrase the ending so it maps to observable results only. |

## English Notes

The manuscript mostly satisfies the IEEE CL organization pattern:

- Introduction: yes, with a clear scenario -> gap -> method -> contribution chain.
- System Model: yes, with definitions before optimization.
- Algorithm: yes, with state/action/reward and an algorithm box.
- Results: mostly yes, but one paragraph needs compression and clearer separation of observation and explanation.
- Conclusion: yes, but slightly broad in wording.

Terminology is mostly stable. The only repeated drift is around the final outcome words:

- `capacity`, `performance`, `quality`, `continuity`, `reliability`, `accuracy`, `error`

These are not wrong, but they should not be used interchangeably unless they refer to different quantities.

## Summary

The manuscript is close to IEEE Communications Letters style. The main fixes are local: tighten a few claims, compress one results paragraph, and keep terminology boundaries strict.
