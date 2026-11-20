# Team execution guide — Silverstone proof of concept

**CS 682: Modern Introduction to Neural Networks | November 20, 2026**  
Proposed for team review. No teammate assignments or new experimental results are implied by this document.

## 1. Research question and common reading

Can a temporal neural network predict historical F1 pit decisions better than a current-state baseline, and do those predictions translate to improved simulated race outcomes? Predicting a historically observed action does not establish its optimality. Compare prediction and decision quality separately. Our contribution is an empirical comparison and integration, not a claim of a new simulator or novel architecture.

All four members should read the methods, experiments and limitations of:
- Heilmeier et al., [Virtual Strategy Engineer (2020)](https://doi.org/10.3390/app10217805): published pit timing and compound-selection networks embedded in the TUM simulator. [Original code](https://github.com/TUMFTM/race-simulation).
- Sasikumar et al., [pit-stop prediction study (2025)](https://doi.org/10.3389/frai.2025.1673148): sequence-model motivation; check exactly what was evaluated before claiming a gap.
- Course Assignment 1 (two-layer classifier), lecture 11 (RNN/GRU/LSTM). MLP means multilayer perceptron, not a separate training paradigm.

Every member should be able to explain the diagram below, the source of the labels, the model inputs, the policy action, and one limitation of simulated evaluation.

## 2. Two stages, one input contract

```text
TRAIN (offline)
FastF1 laps -> label + shared features -> whole-race split -> train MLP/GRU
                                               |-> held-out prediction metrics
                                                       |
                                             saved weights + threshold
                                                       |
EVALUATE (closed-loop)                                  v
TUM simulated state -> same encoder -> model -> policy action -> TUM race update
         ^                                                        |
         +---------------------- next lap ------------------------+
                           -> trace + time + position
```

Historical label: pit in during lap t (PitInTime observed). It is an outcome, never a feature. The model predicts on information available at the start of lap t. After an intervention, use the new simulator states rather than historical observations. Never pass true future laps or outcomes into inference.

Existing minimal encoder in `f1_ai/pit_pipeline.py`: ordered float32 vector [(t-1)/planned_laps, previous_position/field_size, previous_pits]. Lap 1 omitted in historical extraction; explicitly fix or skip inference for lap 1 consistently. Audit difference between declared decision time and actual VSE callback timing. Current code does not deploy a trained model. The alternative is a fixed lap-24 stop, not an NN prediction.

Proposed interfaces (agree together; not implemented by this document):
- Data: table keyed by race_id, driver, lap, with the three features above and binary pit label; split manifest by whole race. Further features require source/timing/units/missingness audit by integration owner.
- Model: PyTorch input (N,D) for MLP, (N,K,D) for sequence; output (N,) logits; saved checkpoint with model config, feature order, scaler (if used), history length, threshold, split ID, seed. Use sigmoid for probability; threshold selected on validation.
- Simulator: policy(state, history) -> None (stay out) or allowed TUM compound code (pit). Start with historical/explicit fixed policies. Learned timing uses a common legal compound rule, not an unsupported learned compound prediction. Log requested and executed actions; reset each race, prevent invalid/repeated stops and handle dry-compound rules. Keep upstream source intact.
- Experiment: consume checkpoint and configuration, call simulator, store metadata, action trace and outcome. Pair seeds and conditions; keep other driver policies fixed, not their resulting positions.

## 3. Exact work packages and handoffs

Owner assignments and checkpoint dates are confirmed at the meeting, not inferred here. Existing issues remain the task tracker.

| Issue / owner | First executable objective | Deliverable / receiver | Acceptance test |
|---|---|---|---|
| #2 Simulator | Run 2019 Silverstone historical replay and fixed lap-24 A2 policy for VER with other drivers using historical strategies. | Reproduction command; trace CSVs; comparison CSV; minimal policy interface for #1. | Actual executed stops match requested schedule, VER completes 52 laps, compare to saved replay, verify other drivers' policies unchanged. Report any failure. |
| #4 Data | Extract Silverstone VER observation/label rows and document exact timing; expand to several accessible dry races/all drivers only after extraction works. | historical_states.csv; data dictionary; reusable extraction/split function; manifest and train/val/test arrays for #3. | Print pit labels, class counts and missingness; no current-lap features; disjoint race IDs; no window crosses driver/race; hold out Silverstone for fitted-model evaluation if feasible. |
| #3 Model | Train existing small MLP on actual training races; implement logistic baseline and tiny-batch test. | Reusable training/inference code, weights + config, validation threshold, metrics table and loss plot to #1. | Tiny batch overfits, real held-out AP/precision/recall/F1 reported on same eligible rows, no threshold tuning on test. Only then implement one causal GRU and flattened-window MLP on same history. LSTM optional if time. |
| #1 Integration / research lead | Compare historical and simulated feature rows at laps around observed pits; define shared schema, then run all policies with identical conditions. | feature_comparison.csv; timing/unit audit; one integrated experiment command; combined results and report interpretation for team. | Matched feature meanings and decision timing; model checkpoint from #3 actually drives callback; outputs trace requested/executed actions and paired race time/position, with limitations. |

Dependency order: #2 and #4 parallel; #1 audits the interface from both; #3 can test synthetic data immediately but trains only after #4 delivers splits; #1 closes the loop after #2 and #3 deliver. Do not block on GRU development to validate the MLP. Simulator owner need not know PyTorch internals; model owner need not rewrite TUM.

Common initial checkpoint: Sunday November 22, 3 PM is proposed in existing issues; confirm at meeting. On blocker: push branch, exact command/error, what is done, next reproducible step, and tag dependency owner. Another teammate can pick up by checking out the branch. Do not overwrite concurrent work.

## 4. Repository organization and reuse

Current code remains where it is while the active local simulation work completes. This is the proposed layout, not a claim that files have been moved or interfaces implemented.

```text
README.md / AGENTS.md              # setup and working conventions
race-simulation/                    # original published TUM, compatibility changes documented
f1_ai/pit_pipeline.py               # existing proof of concept, retain until imports/tests migrated
f1_ai/data.py                       # proposed: reusable historical extraction, splits
f1_ai/models.py                     # proposed: architectures, checkpoint loading
f1_ai/simulator.py                  # proposed: policy adapter, not another simulator
f1_ai/evaluation.py                 # proposed: shared metrics
notebooks/                          # proposed: runnable demonstrations, imports above
experiments/                        # proposed: scripts/configs for reproducibility
results/                            # compact licensed derived tables, figures, metadata
tests/                              # regression tests, synthetic checks, integration checks
docs/experiment_plan.md             # detailed scientific protocol (E0–E5)
docs/team_execution.md              # this guide
docs/result_template.md             # standard run record
```

Do not duplicate core model/data/adapter code between notebooks. Each owner provides one importable API, one demonstration notebook with visible outputs, one test and one exact command. Move notebook/module paths only after active local edits are reviewed and all imports/tests pass. Do not move upstream files.

Work on issue-specific branches; fetch and merge/rebase latest main as appropriate, open PR with exact executed command and exit status, review by a different person, merge after tests and review. Keep raw caches, credentials, large weights and environments outside Git. Document where permitted checkpoints can be obtained and their hash, rather than committing large blobs. Derived outputs only where sharing is permitted; otherwise provide regeneration command. Do not claim results before running.

## 5. Outputs and experiments

Use a distinct run directory per experiment and seed. Store command, commit hash, environment, config, stdout/stderr, exit status, input provenance, metrics, traces and plots. Keep failures visible. Use `docs/result_template.md`. Notebooks should be executed from a fresh kernel and saved with permitted outputs. Record actual sample size, not just percentages.

E0 validates the environment; E1 validates shared input; E2 validates prediction; E3 compares architectures; E4 compares downstream outcomes. No claim that a fixed policy or synthetic-trained model improves real racing. The saved historical replay is prior evidence (VER P2 simulated vs P5 actual), not our new run. See the existing [experiment plan](experiment_plan.md).

## 6. Meeting and paper

At the meeting each person states which issue they will own, first artifact, dependency, exact acceptance test, and checkpoint. Everyone reads common papers and reviews one other workstream. Every run should contribute a methods paragraph and an evidence-backed result or limitation to the report. Integration lead owns coherent research narrative, not every implementation. Course guideline in repository planning notes says 8 main pages excluding references, limitations/ethics and contributions; confirm any newer instructor instructions before fixing a 15-page total.
