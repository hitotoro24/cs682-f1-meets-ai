# Experiment run record (copy into the run folder)

- Experiment ID (E0–E4), run ID, owner, date:
- Research question / hypothesis:
- Git commit and local changes:
- Python version, dependency versions, OS, device:
- Data source, race/driver IDs, split manifest, feature order and timing:
- Model architecture, checkpoint hash, threshold, seed, config:
- Simulator conditions, driver policy, initial state, seed:
- Exact execution command:
- Exit status and stdout/stderr log path:
- Input/output artifacts and regeneration instructions:
- Observation: actual numeric results, sample size, plots:
- Acceptance tests and status (pass/fail/not run):
- Interpretation, limitations, possible confounders:
- Decision / next minimal test:
- Paper contribution / reviewer:

Save readable CSV tables and figures plus permitted executed notebook outputs. For long runs, log both streams using `python -u ... 2>&1 | tee run.log`; in zsh enable `set -o pipefail` so failures are not hidden by tee. Record the exit status. Keep raw caches, secrets and unrestricted large output out of Git. A successful synthetic test is not real-race validation. Never invent missing numbers.
