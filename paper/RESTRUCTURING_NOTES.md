# Experimental Evaluation Restructuring — Summary

**Date:** 2026-07-06  
**File edited:** `paper/CL_paper_IEEE_English_CL_draft_v2.tex`  
**Status:** ✅ Compiled successfully (6 pages)

---

## Changes Applied

### 1. **Abstract** (line ~29)
- **Old:** Claimed 71% improvement over geodesic and 42% over independent baseline
- **New:** Focuses on ~40% localization improvement over raw-metric RL baseline in a "representative urban scenario"
- **Rationale:** Single-run data without multi-seed stats; geodesic dropped from headline comparison; honest hedging per user guidance

### 2. **Introduction Contribution #3** (line ~44)
- **Old:** Generic "baseline and ablation comparisons"
- **New:** Explicitly quantifies CLF prior's role through 3 baselines (raw-metric RL, CLF-weighted Dijkstra, MAPPO-without-CLF) that isolate fingerprint vs. CTDE contributions
- **Rationale:** Clearer reviewer logic for the 4-method structure

### 3. **Experimental Setup — Baseline Descriptions** (lines ~244–280)
Replaced the old 3-method paragraph with a detailed 4-method comparison framework:

#### **(i) GeoAgg-style raw-metric RL baseline**
- Identical MAPPO/CTDE training
- Optimizes log-potential reward on **raw** physical metrics (C, 1/L), no CLF fingerprint
- Observes only current-voxel raw features, no neighbor look-ahead
- **Tests:** whether direct physical reward suffices
- **Real data:** ✅ exists in `outputs/method_comparison.csv` (L=19.34 m, C=20.69 Mbit/s)

#### **(ii) CLF-weighted Dijkstra**
- Non-learning graph search with CLF-aware edge cost (Eq. \ref{eq:dijkstra_cost})
- Uses prioritized planning + reservation table for multi-UAV collision avoidance
- **Tests:** whether CLF representation alone solves the task via conventional search
- **Real data:** ❌ **TODO** — planner not yet implemented

#### **(iii) MAPPO-without-CLF ablation**
- Same MAPPO/CTDE training, same arrival/energy/collision reward
- **Removes** fingerprint channels F(v_n) and F(V_n^adj) from observation
- Keeps position, goal, action masks, local occupancy
- **Tests:** whether MARL learns without the fingerprint prior
- **Real data:** ❌ **TODO** — observation-side ablation not yet implemented (existing `compare_policies.py` zeros reward weights, different ablation)

#### **(iv) CLF-MAPPO (proposed)**
- Combines CLF look-ahead with CTDE learning
- **Real data:** ✅ exists (L=11.14 m, C=20.74 Mbit/s, 42% L improvement vs. baseline)

**All methods scored on neutral physical metrics** (C, L=√CRLB) that favor no particular design.

---

### 4. **Parameter Table** (Table I, line ~230)
- **Updated:** Decoupled transmit powers: `Comm / loc transmit power: -10 dBm / 20 dBm`
- **Updated:** Noise specification: `Noise PSD / figure: -174 dBm/Hz / 5 dB`
- **Rationale:** Matches actual parameters used in `compare_three_methods.py` headline run

---

### 5. **Figure 3: Representative Trajectories** (line ~249)
- **Old label:** `fig:method_comparison` (3-panel layout: paths + C/L curves + bars)
- **New label:** `fig:trajectories` (single 3-D scene, all 4 methods)
- **Current asset:** `outputs/method_comparison.png` (shows only 3 methods: geodesic/GeoAgg/CLF)
- **TODO in caption:** Regenerate as standalone 4-method 3-D visualization once Dijkstra and no-CLF ablation exist
- **Caption rewrite:** Explains dead-zone avoidance mechanism — CLF-aware methods detour around it, raw-metric/no-CLF methods cross it

---

### 6. **Table II: Physical Service Metrics** (line ~259)
**Old structure:** 3 rows (Geodesic greedy, GeoAgg, CLF proposed)  
**New structure:** 4 rows:

| Method                          | Arr. | L̄ (m)↓ | C̄ (Mbit/s)↑ | L<10m | coll. |
|---------------------------------|------|---------|--------------|-------|-------|
| CLF-MAPPO (proposed)            | 3/3  | **11.14** | 20.74      | **0.74** | 0   |
| CLF-weighted Dijkstra           | --   | --      | --         | --    | --    |
| GeoAgg-style RL [ref11]         | 3/3  | 19.34   | 20.69      | 0.66  | 0     |
| MAPPO without CLF               | --   | --      | --         | --    | --    |

- **Real data:** CLF-MAPPO and GeoAgg-style RL rows from `outputs/method_comparison.csv`
- **TODO in caption:** Fill Dijkstra and no-CLF rows once implemented; add mean±std over seeds/maps

---

### 7. **Figure 4: Quantitative Comparison** (new, line ~299)
**Replaces old Fig. 4 (algorithm/resolution training)**

**Structure:** Two subfigures
- **(a) Training return curves** (3 RL methods: CLF-MAPPO, GeoAgg-style, MAPPO-no-CLF)
  - **Current asset:** `algo_return_only.png` (stand-in; shows MAPPO/IPPO/MAA2C)
  - **TODO in caption:** Replace with actual 3-method training curves; flag reward-comparability caveat (different reward functions → returns not directly comparable; suggest neutral metric or shared nav-only return)
  
- **(b) Physical metrics bar chart** (all 4 methods: L̄, C̄, arrival, collision)
  - **Current asset:** Placeholder framebox (no real figure yet)
  - **TODO in caption:** Generate grouped bar chart once all 4 methods exist

**New label:** `fig:quant`

---

### 8. **Figure 5: Resolution Sweep** (line ~321, renumbered from old Fig. 4b)
- **Asset:** `outputs/resolution_return_only.png` (real data, unchanged)
- **Label:** `fig:resolution`
- **Caption:** Preserved as-is (coarser grids converge faster)

---

### 9. **Figure 6: Pareto Frontier** (line ~327, renumbered from old Fig. 5)
- **Asset:** `outputs/pareto_clf.png` (real data, unchanged)
- **Label:** `fig:pareto`
- **Caption:** Preserved as-is (tunable w_L sweep)

---

### 10. **Narrative Sections** (lines ~296–314)
Restructured into 3 subsections with honest data disclosure:

#### **Trajectory behavior (Fig. 3)**
- Explains dead-zone avoidance mechanism for all 4 methods
- **Real backing:** CLF-MAPPO and GeoAgg-style RL only
- **TODO note:** Verify Dijkstra/no-CLF trajectory descriptions once implemented

#### **Physical metrics (Table II)**
- Reports 42% L reduction (11.1 vs 19.3 m) at matched C (~20.7 Mbit/s)
- Notes single-run limitation, no confidence claim
- **TODO note:** Fill missing rows, add multi-seed stats

#### **Training stability (Fig. 4a)**
- Describes what the panel *should* show (3-method convergence comparison)
- Honest disclosure: current panel is algorithm-family comparison stand-in
- **TODO note:** Replace with 3-method curves, address reward comparability

---

### 11. **Conclusion** (line ~330)
- **Old:** Mentioned geodesic + independent-learning baselines
- **New:** "~40% localization improvement relative to raw-metric learning baseline"; graph-search and no-CLF comparisons "designed to isolate" roles (future tense → honest about incomplete data)
- **Dropped:** "significantly"/"robustly" claims (single scenario, no multi-seed stats)

---

## Data Availability Summary

| Method                  | Code Exists? | Data Exists? | Source File                        |
|-------------------------|--------------|--------------|-------------------------------------|
| **CLF-MAPPO**           | ✅ Yes       | ✅ Yes       | `outputs/method_comparison.csv`     |
| **GeoAgg-style RL**     | ✅ Yes       | ✅ Yes       | `outputs/method_comparison.csv`     |
| **CLF-weighted Dijkstra** | ❌ No      | ❌ No        | TODO: implement planner + prioritized planning |
| **MAPPO-without-CLF**   | ❌ No*       | ❌ No        | TODO: implement obs-side ablation   |

\* Note: `scripts/compare_policies.py` has a "no-CLF" run but it zeros **reward** weights (w_C=w_L=0), not **observation** channels — different ablation.

---

## What Still Needs Implementation

### **High Priority (headline 4-method comparison)**

1. **CLF-weighted Dijkstra planner** (`scripts/clf_dijkstra.py` or extend `compare_three_methods.py`)
   - Implement Eq. \ref{eq:dijkstra_cost} edge-cost function
   - Add prioritized planning with reservation table for multi-UAV coordination
   - Integrate into comparison script, output to `outputs/method_comparison_4way.{csv,png,npz}`
   - Estimated time: ~2–3 hours (deterministic, no training)

2. **MAPPO-without-CLF-observation ablation**
   - Add `observation_mode` config flag (like `reward_mode`) to `src/marl_env.py`
   - In `_self_features()` and `get_obs()`, zero out F(v_n) and F(V_n^adj) channels when mode="no_clf"
   - Keep position, goal, action masks, local occupancy
   - Train with same MAPPO settings as headline (entropy_coef=0.03, shaping_coef=70, epi_max=800+)
   - Estimated time: ~1–2 hours implementation + 30–60 min training per seed

3. **Regenerate Figure 3** (4-method trajectories)
   - Extend `scripts/compare_three_methods.py:plot_three_way()` to handle 4 methods
   - Output single 3-D scene (not 3-panel) to `outputs/trajectories_4way.png`
   - Estimated time: ~30 min once methods 1–2 exist

4. **Generate Figure 4(b)** (4-method physical metrics bar chart)
   - Grouped bar chart: 4 methods × 4 metrics (L̄, C̄, arrival, collision)
   - Use `matplotlib` grouped bars with error bars if multi-seed
   - Output to `outputs/quant_metrics.png`
   - Estimated time: ~30 min once data exists

5. **Generate Figure 4(a)** (3-method training return curves)
   - Plot team episode return vs. episodes for CLF-MAPPO, GeoAgg-style RL, MAPPO-no-CLF
   - **Caveat:** Returns aren't directly comparable across different reward functions — consider normalizing by shared nav component or plot neutral metric (arrival fraction) instead
   - Output to `outputs/quant_training.png`
   - Estimated time: ~30 min once no-CLF ablation data exists

### **Medium Priority (rigor)**

6. **Multi-seed training** for all 4 methods
   - Run seeds 0–4 (or 0–7) per method, report mean±std
   - Allows "significantly" claims if p-values support it
   - Estimated time: ~2–4 hours training × 4 methods × 5 seeds = substantial compute

7. **Randomized urban maps**
   - Vary building positions, BS positions (keeping feasible-voxel density similar)
   - Verify CLF advantage holds across map instances
   - Estimated time: ~1 day (map generation + retraining)

---

## Recommendations

### **Path A: Paper-only restructure (DONE ✅)**
What was delivered today:
- 4-method experimental framework with honest TODO placeholders
- Real data for 2/4 methods (CLF-MAPPO, GeoAgg-style RL)
- LaTeX compiles cleanly, structure ready for figures/data to drop in
- Abstract/conclusion hedged appropriately (single scenario, ~40% improvement)

**Use case:** Submit draft to advisor for review of experimental logic before running expensive compute.

---

### **Path B: Minimal complete comparison (recommended next step)**
Implement items 1–5 above to fill all 4 methods with single-run data:
- **Time estimate:** ~1 day (mostly Dijkstra + obs-ablation implementation/training)
- **Deliverable:** Full 4-method comparison with real trajectories, metrics, training curves
- **Limitation:** Still single-run per method (no multi-seed stats)

**Use case:** Complete paper draft for internal review before committing to multi-seed runs.

---

### **Path C: Full rigor (if submission-ready)**
Add items 6–7 (multi-seed, randomized maps):
- **Time estimate:** +2–3 days compute
- **Deliverable:** Statistically robust claims, confidence intervals, generalization evidence
- **Required for:** Strengthening abstract claims from "~40%" to "significantly reduces" with p-values

**Use case:** Final submission version after internal acceptance of Path B results.

---

## Technical Notes

### **Reward-comparability caveat (Fig. 4a)**
The three RL methods optimize **different reward functions**:
- **CLF-MAPPO:** `r = w_C(f_comm-1) + w_L(f_loc-1) + arrival + energy + collision`
- **GeoAgg-style RL:** `r = log(C/γ_r) + (1/L)/scale + arrival + energy + collision`
- **MAPPO-no-CLF:** `r = arrival + energy + collision` (no comm/loc term if ablation removes observation)

Raw team return curves aren't directly comparable. Options:
1. **Plot shared navigation-only return** (arrival + energy + collision components only)
2. **Plot neutral physical metric** (e.g., arrival fraction vs. episodes)
3. **Normalize by final return** and compare convergence speed
4. **Accept incomparability** and frame as "convergence stability" rather than "performance ranking"

Current TODO in caption flags this; decide resolution when implementing.

---

### **Geodesic baseline**
The old 3-method comparison included a geodesic greedy baseline (L=38.72 m, C=18.69 Mbit/s). It was **dropped** from the headline 4-method comparison per user spec, but the data still exists in `outputs/method_comparison.csv` if you want to restore it as a 5th reference (though it's weaker than CLF-Dijkstra and clutters the story).

---

### **File integrity**
All edits preserved **real experimental data**:
- No fabricated numbers inserted
- Existing figures/tables unchanged where data backs them
- TODOs mark every gap transparently
- Compilation verified (6 pages, no errors)

Original draft backed up as `CL_paper_IEEE_English_CL_draft.tex` (still in repo).

---

## Contact / Handoff

If implementing Path B/C, key files to modify:
- `src/marl_env.py` — add `observation_mode` flag (mirrors `reward_mode` pattern)
- `src/environment.py` — Dijkstra baseline can reuse `geodesic_field()` logic, swap edge weights
- `scripts/compare_three_methods.py` — extend to 4 methods, add prioritized planning for Dijkstra
- `scripts/plotting.py` or inline — grouped bar chart + 3-method training curve functions

All TODO comments in the `.tex` file are prefixed `% TODO:` and searchable.

---

**End of summary.** Paper is ready for advisor review or for implementing the missing baselines.
