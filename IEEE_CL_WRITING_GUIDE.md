# IEEE Communications Letters Writing Guide

This guide summarizes academic writing patterns observed from the reference papers in `paper/writing_refs`. It focuses on writing style and section organization, not on the technical content of the papers.

## 0. Source Basis

### Actually Observed in the Reference Papers

The following papers were analyzed:

1. `Trajectory_Design_and_Link_Selection_in_UAV-Assisted_Hybrid_Satellite-Terrestrial_Network.pdf`
2. `Enhanced_RSS-Based_UAV_Localization_Via_Trajectory_and_Multi-Base_Stations.pdf`
3. `Joint_3-D_Deployment_and_Resource_Allocation_for_UAV-Assisted_Integrated_Communication_and_Localization.pdf`
4. `Energy-Efficient_Decision-Making_With_Belief_Structures_for_Dynamic_Path_Planning_in_UAV-Enabled_6G_Networks.pdf`
5. `Deployment_Optimization_of_Dual-Functional_UAVs_for_Integrated_Localization_and_Communication.pdf`
6. `Three-Dimensional-Map-Based_Trajectory_Design_in_UAV-Aided_Wireless_Localization_Systems.pdf`
7. `Jointly_Optimize_Throughput_and_Localization_Accuracy_UAV_Trajectory_Design_for_Multiuser_Integrated_Communication_and_Sensing.pdf`
8. `ISAC_From_the_Sky_UAV_Trajectory_Design_for_Joint_Communication_and_Target_Localization.pdf`
9. `GeoAgg-HSAC_an_RL-based_Framework_for_Trajectory_and_Resource_Optimization_in_Mountainous_UAV_Integrated_Localization_and_Communication_Networks.pdf`

The Communications Letters-style papers use compact sections, short contribution statements, direct problem formulation, and concise simulation interpretation. The longer TWC/IoT-style papers use more subsections and richer derivations, but their local writing patterns around models, algorithms, and figures remain useful.

### My Inference for an IEEE Communications Letters Manuscript

For IEEE Communications Letters, the safest style is compressed but explicit: each paragraph should have one technical function; every claim should connect to a model, constraint, metric, algorithmic step, or numerical result. Avoid broad narrative, extensive survey, and promotional phrasing.

## 1. Introduction

### Actually Observed

The introduction usually follows a four-step logic:

1. Establish the application setting and why it matters.
   The opening paragraph starts from a concrete wireless-network scenario, such as UAV-assisted communication, localization, ISAC/ILAC, emergency coverage, or satellite-terrestrial networking. It names the key system benefit and then immediately introduces the practical limitation.

2. Compress related work by technical category.
   Prior work is grouped by method or system type, not listed paper by paper without synthesis. Typical groupings include deterministic optimization, UAV trajectory design, localization metrics, map-aided modeling, GNN/RL methods, or resource allocation.

3. State the limitation or research gap.
   The gap is written as a technical mismatch between existing assumptions and the target scenario. Common forms are: existing methods assume complete information, static/simple obstacles, single-function UAVs, fixed deployment, single user/target, or high-complexity solvers.

4. Present the paper's response and contributions.
   The final introduction paragraph often starts with a motivation phrase, then states the considered system, objective, key method, and main result. Short Letters often avoid a long bullet list unless space allows; when bullets are used, each contribution begins with a precise verb such as "formulate", "derive", "propose", "design", or "show".

Observed contribution statements are technical and bounded:

- Model contribution: what system, uncertainty, channel, localization, or resource-coupling model is considered.
- Problem contribution: what objective and constraints are jointly optimized.
- Method contribution: what transformation, approximation, learning framework, or solver is introduced.
- Result contribution: what metric improves and against which baseline.

### My Inference

For a Communications Letters submission, use 3 to 5 introduction paragraphs:

1. Background and practical need: 3-5 sentences.
2. Related work and limitation: 4-6 sentences.
3. Research gap and challenge: 2-4 sentences.
4. Proposed approach and contributions: 4-7 sentences or 2-3 compact bullets.

Do not write a broad tutorial. Every sentence should narrow the path toward the exact problem solved in the letter.

## 2. System Model

### Actually Observed

System Model sections usually open with a direct scenario sentence: "Consider a ... system/network, where ...". The figure is referenced early, then sets, coordinates, time slots, users, UAVs, BSs, targets, and channels are defined before equations become dense.

The common order is:

1. Scenario and topology.
   Define nodes, coordinates, time horizon, slot index, and service relationships.

2. Assumptions.
   State only assumptions that affect formulas: fixed altitude, known user locations, mobility model, channel model, LoS/NLoS condition, bandwidth sharing, energy model, or localization measurement type.

3. Metric definitions.
   Introduce communication rate, localization accuracy, CRB/D-optimality/GDOP, energy consumption, or energy efficiency. Each metric is introduced by a sentence that explains why it is used.

4. Problem formulation.
   Present the optimization problem after all variables and metrics have been defined. Immediately after the problem, explain each constraint in one compact sentence or phrase.

Observed writing habits:

- Symbols are introduced close to their first use.
- Equations are preceded by a short modeling sentence and followed by definitions of variables that were not already introduced.
- Constraints are interpreted after the optimization problem rather than left unexplained.
- "For simplicity", "Without loss of generality", and "Unless otherwise specified" are used sparingly and only when they remove ambiguity.
- Long derivations are avoided in Letters unless they directly enable the solver.

### My Inference

Use a strict symbol discipline:

- Keep one name per concept. Do not alternate between "UE", "user", "terminal", and "GT" unless the distinction is real.
- Prefer definitions such as `i in I`, `k in K`, `t in T` once, then reuse them consistently.
- Do not repeat the same channel/rate/localization formula in Algorithm or Results. Refer back to the equation number.
- After every optimization problem, include a compact constraint explanation, e.g., "(x) enforces the UAV mobility constraint, (y) imposes the transmit-power budget, and (z) guarantees the localization requirement."

## 3. Algorithm / MARL Section

### Actually Observed

Algorithm sections first state why the formulated problem is hard: mixed-integer variables, nonconvex constraints, high-dimensional state space, partial observability, dynamic topology, or computational cost. Then they give the main solution idea before details.

For RL/MARL-style sections, the common sequence is:

1. State the transformation to an MDP or learning problem.
2. Define the tuple, typically state, action, reward, and discount factor or transition setting.
3. Define state with the information available to the agent, not all system variables.
4. Define action by decision variables the agent controls.
5. Define reward by linking it to the original objective and constraints.
6. Present training or execution in an algorithm box.
7. Add complexity, convergence condition, or implementation note only if space permits.

Observed style for state/action/reward:

- State definitions are concise and enumerative.
- Action definitions specify continuous and/or discrete components explicitly.
- Reward is tied to communication/localization/energy objectives and penalties, instead of being described as a vague performance score.
- Algorithm text avoids re-deriving earlier equations. It says that payoffs, rewards, or constraints are evaluated according to previously defined expressions.
- Training details are concrete: number of episodes, learning rate, batch size, network layers, exploration policy, or stopping rule.

### My Inference

For a MARL section in an IEEE Communications Letters manuscript:

- Start with one paragraph: "Problem P is difficult because ... Therefore, we use ...".
- Define `state`, `action`, and `reward` in bullets or a compact paragraph, not in scattered prose.
- If multi-agent, explicitly distinguish local observation, shared/global information, and centralized training/decentralized execution if applicable.
- Define reward once. Later algorithm steps should say "compute the reward using (x)" rather than restating the reward formula.
- Use the algorithm box for sequence, not explanation. Put rationale in the paragraph before or after the algorithm.
- Avoid claiming "optimal" for learning-based methods unless the proof supports it. Use "learns", "approximates", "obtains", or "updates".

## 4. Experimental Results

### Actually Observed

The Results section usually begins by stating the purpose of simulations and listing the simulation environment, parameters, and baselines. Tables are used for dense parameter settings; repeated conditions are handled by "Unless otherwise specified".

Figure discussion often follows this order:

1. Result phenomenon.
   Begin with what the figure shows: convergence, increasing/decreasing trend, tradeoff, trajectory behavior, or baseline comparison.

2. Quantitative comparison.
   Give concrete values, percentages, relative gains, runtime reductions, or ranking among methods when available.

3. Reason explanation.
   Explain the mechanism behind the observed result using the model or algorithm design: LoS probability, resource reallocation, energy budget, trajectory proximity, state aggregation, uncertainty handling, or constraint relaxation.

Observed result paragraphs are conservative:

- They use "shows", "demonstrates", "indicates", "can be attributed to", and "this is because" to connect evidence and explanation.
- Baselines are named explicitly and abbreviated after first definition.
- Strong results are tied to specific conditions, not generalized to all scenarios.
- Some papers also acknowledge tradeoffs, such as improved performance at the cost of complexity or reduced performance under stricter constraints.

### My Inference

Use one paragraph per figure or subfigure group:

1. Sentence 1: identify the figure and primary observation.
2. Sentence 2: report the key numerical comparison.
3. Sentence 3: explain why the trend occurs.
4. Optional sentence 4: state the implication under the studied setting.

Avoid result paragraphs that only say the proposed method is better. The reader should learn the condition, magnitude, and cause of the improvement.

## 5. Conclusion

### Actually Observed

Conclusions are short and retrospective. They usually:

1. Restate the studied problem in one sentence.
2. Summarize the proposed model/method in one to three sentences.
3. Summarize numerical findings in one sentence.
4. Optionally mention a future extension, especially in longer papers.

The tone is factual. Claims are not expanded beyond what the simulations support.

### My Inference

For IEEE Communications Letters, keep the conclusion to one compact paragraph unless the journal format strongly favors two. Do not introduce new background, new symbols, new baselines, or unverified implications. A good conclusion reads like a compressed record of what was done and what was observed.

## 6. Common Sentence Patterns and Technical Style

### Actually Observed Patterns

Useful functional patterns, rewritten generically:

- Background: "Recent advances in X enable Y in Z scenarios."
- Problem pressure: "However, achieving Y is challenging due to A and B."
- Related work grouping: "Existing studies have investigated A, B, and C."
- Gap: "These methods typically assume A, and therefore do not capture B."
- Paper positioning: "In this letter, we consider a ... system where ..."
- Objective: "We aim to maximize/minimize ... subject to ..."
- Formulation: "The problem is formulated as ..."
- Difficulty: "The formulated problem is difficult to solve because ..."
- Method: "To address this issue, we propose/design/derive ..."
- Algorithm summary: "The main idea is to ..."
- Figure result: "Fig. X shows ..."
- Explanation: "This is because ..." or "This can be attributed to ..."
- Constraint explanation: "Constraint (x) ensures/imposes/limits ..."
- Conservative result claim: "Numerical results show that ... under the considered setup."

### Terminology Habits

- Use abbreviations after first definition: UAV, BS, LoS, NLoS, ISAC, ILAC, CRB, GDOP, MDP, MARL.
- Keep one preferred term for each entity. Technical consistency is more important than synonym variation.
- Prefer exact technical nouns over broad adjectives: "nonconvex constraint", "mixed-integer variable", "partial observability", "LoS blockage", "resource allocation", "trajectory planning", "localization accuracy".
- Use "proposed method/scheme/algorithm/framework" consistently; do not rotate among many synonyms.
- Use "baseline", "benchmark", and "existing method" only when the comparison role is clear.

## 7. Expressions to Avoid

### Actually Observed by Contrast

The stronger reference passages avoid long promotional claims. They rarely use unsupported phrases such as "revolutionary", "groundbreaking", "remarkable", or "perfect". Improvements are tied to metrics, figures, and baselines.

### My Inference

Avoid AI-like or low-information wording:

- "This paper provides a comprehensive and robust solution" unless comprehensiveness and robustness are evaluated.
- "Significantly improves performance" without metric, baseline, and condition.
- "Greatly enhances system efficiency" without numerical evidence.
- "It is obvious that" when a model-based explanation is needed.
- "Due to the rapid development of technology" as a generic opening.
- Repeated restatement of the same motivation in Introduction, System Model, and Algorithm.
- Formula paraphrases that duplicate the mathematical expression without adding interpretation.
- Excessive synonym replacement for key terms.
- Overclaiming learning results as globally optimal when the algorithm is heuristic, approximate, or trained under finite simulations.

## 8. Compact IEEE CL Checklist

Before submission-style rewriting, check:

- Introduction has a clear chain: scenario -> related work -> gap -> contribution.
- Each contribution is a technical action with a concrete object.
- System Model defines all sets, variables, assumptions, and metrics before optimization.
- Every formula has a purpose sentence before it and symbol clarification after it.
- The MARL section defines state, action, reward, and training flow once.
- Results paragraphs follow phenomenon -> quantitative comparison -> cause.
- Conclusion does not introduce new claims.
- Terminology is consistent across sections.
- Claims are conservative and tied to evidence.
