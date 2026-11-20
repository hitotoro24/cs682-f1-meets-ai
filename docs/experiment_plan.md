# Experiment plan and acceptance criteria (unassigned)

This is a shared research plan, not a division of people. Claim work only after reviewing the dependencies together. The immediate checkpoint is evidence from one race, not a finished neural network.

## E0 — Reproduce the historical replay and verify a controlled intervention

**Question:** Does the simulator approximately reproduce Silverstone 2019 when fed historical strategies, and does changing VER's strategy change the outcome?
- Existing evidence: saved replay CSV, 20 drivers, mean absolute finishing-position error 1.90, Spearman rank correlation 0.901 (calculated from the committed CSV). VER simulated P2 vs real P5. This is not a newly reproduced run.
- Reproduce the existing replay in the Python 3.11 environment. Verify the VER plan (parameter file lists laps 13 and 20) and compare actual output.
- Run the notebook or `python -m f1_ai.pit_pipeline race` from the root. Historical replay vs fixed lap-24 A2 stop, only VER changes; random effects disabled. The test policy is not optimized. Confirm exact executed stops from `driver.strategy_info`, full lap count, and unchanged strategy policy for other drivers. Report a failure or timeout explicitly.
- Output: `historical_trace.csv`, `fixed_lap_24_trace.csv`, `policy_comparison.csv`, reproducible commands, a plot/table of time and position differences.
- Acceptance: both runs complete, the intervention is confirmed, and baseline output is reconciled with the existing replay. If this fails, no model comparison proceeds.

## E1 — Agree on historical/simulated input schema and audit labels

**Question:** Can we supply an identical decision-time representation when training and during simulation?
- Run `python -m f1_ai.pit_pipeline data` for Silverstone 2019 VER. Record cache/access details and pit-in labels. Do not silently substitute toy samples.
- Join historical observations and baseline trace by lap using the notebook. Explicitly inspect laps 12–21 (historical stops in the TUM config are 13 and 20). Verify prior-lap position, earlier stops, race progress; explain differences.
- Current minimal schema: start of lap t, progress=(t-1)/planned laps, previous-lap position/field size, prior pit count. Lap 1 omitted from historical examples because grid position is not yet joined; do not include lap-1 training/inference until handled consistently.
- Expand only after this passes: tyre age vs degradation age, current compound vs A2/A3/A4 coding, gaps with timing/missingness conventions, SC/VSC, driver identity, used two dry compounds. Keep next compound and current pit indicator as targets only.
- Output: `historical_states.csv`, `feature_comparison.csv`, data dictionary with source, units, timing and missing rules; one test per mapping.
- Acceptance: actual non-pit and pit examples are inspected, differences are identified, no current-lap/future leakage, and shared encoder tests pass.

## E2 — Build a reproducible training dataset and baselines

**Question:** Can a model predict held-out pit laps better than a simple rule?
- Begin with several dry races, all drivers; record counts and pit prevalence per race and split. Extend to proposal's 2019–2025 scope only once the pipeline is stable.
- Split by whole race before preprocessing or creating sequence windows. Fit encoders/scalers on training races only, use validation for thresholds and hyperparameters. Reserve Silverstone evaluation race (or a separate explicitly named track holdout) before fitting.
- Baselines: never-pit (offline sanity only), fixed tyre-age rule (once verified tyre age exists), logistic regression, one small MLP in PyTorch. Use BCEWithLogitsLoss; consider training-only positive class weight. Report threshold and actual positive counts.
- Output: split manifest, dataset summary, reproducible training script/checkpoint, loss curves, held-out metrics table, confusion matrix.
- Acceptance: held-out evaluation with common rows, no duplicate race across splits, no training on test data, documented seeds and validation-selected threshold. Report precision, recall, F1 and average precision (identify PR-AUC convention); optional pit-event matching within ±2 laps with one-to-one matching.

## E3 — Compare current-state and causal sequence models

**Question:** Does a history of laps improve prediction beyond current-state features?
- PyTorch MLP: (N,D) -> (N,) logits. GRU or LSTM: (N,K,D) -> (N,) logits; ensure each window ends at prediction time, never crosses driver/race boundaries. Keep model sizes and data splits documented; compare on the same eligible observations.
- First verify synthetic forward pass, gradient shapes and overfit a tiny batch. Then train on real data using E2 splits. Optional TCN; bidirectional model is retrospective only, not a deployable comparator.
- Output: training/validation loss, held-out model table over 3–5 seeds, PR curve, failure cases and ablation on history length.
- Acceptance: same feature set, evaluation rows, splits and threshold-selection policy; note differences in parameter counts and any model failures.

## E4 — Connect a trained model to the simulator

**Question:** Does predictive performance transfer to simulated race performance?
- Reuse the E0 decision hook and E1 encoder. Wrap a trained model in eval mode with no gradients and validation-selected threshold. Use valid compound selection and enforce dry-compound rules. Begin with a fixed compound-choice rule shared across all learned policies; only add learned compound prediction if enough time.
- Compare historical replay, fixed strategy, MLP and one causal sequence policy in identical race conditions. Maintain a separate history per driver, reset it each race, use only simulated states during inference; do not feed actual future positions after an intervention.
- First one driver, one track, normal race. Then extend starting positions P1/P10/P18 and SC/VSC if feasible. Use paired seeds for stochastic comparisons; report uncertainty across runs.
- Output: action traces, race time/position/points, completed laps, differences from historical replay, representative strategy plots and failure analysis.
- Acceptance: model decisions demonstrably applied, all competing policies evaluated on the same conditions, report incomplete races separately. No claim of real-world optimality. Regret against enumerated feasible 1–2 stop strategies is optional and must be called simulator-relative.

## E5 — Research report (continuous, not a final-week task)

For each PR/experiment: question, source, inputs/outputs, exact command and environment, observations, limitation, next decision. Contributions: introduction/related work; data and labels; model architecture and offline evaluation; simulation and policy evaluation; limitations and ethics. Everyone should understand the whole pipeline. Course guideline specifies an 8-page main report excluding citations, Limitations/Ethics and Contributions; verify any newer course announcement before choosing the final page budget.

## Implementation conventions

Use NumPy for feature transformations, mathematical checks and array operations; pandas for table/read/write convenience; PyTorch for neural models and optimization. Keep upstream TUM TensorFlow dependency for simulator compatibility, not as our training framework. The guided notebook follows coursework-style setup, purpose, implementation, tests and inline questions; reusable logic stays in Python modules. Do not check in private credentials, large FastF1 caches, or generated outputs without considering data license/privacy. Commit compact derived evidence only when appropriate. AI assistance does not replace test evidence or each contributor's ability to explain the method.

## Current verified evidence (Nov 20)

- Local Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5: the exact committed `f1_ai/pit_pipeline.py` (blob `0f191c9bcae059c4fd641aac717e823c5464095a`) passed the synthetic feature/pit-timing checks; toy BCE 0.714567 -> 0.000583, logits shape (3,). Not generalization evidence.
- Notebook authored but not executed end-to-end. Live FastF1 extraction and TUM intervention remain unverified because FastF1/TensorFlow are absent and network access is unavailable in the authoring environment.
- The saved prior replay is real repository evidence, not a fresh simulation. Real-data access, mapping, intervention, model training and test results must be established on the team's machine.
