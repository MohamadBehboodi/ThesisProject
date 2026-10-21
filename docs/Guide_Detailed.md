# Windows execution guide - dev v3

## Existing environment

Use the Windows Python environment where CUDA-enabled PyTorch already works.
Check the interpreter and device before execution:

```powershell
python -c "import sys,torch; print(sys.executable); print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

No new external runtime package was added. For a fresh environment, install a suitable
CUDA-enabled PyTorch following https://pytorch.org/get-started/locally/ before the
remaining requirements. Native Windows locking uses msvcrt; WSL is not required.

## First steps on dev

```powershell
git fetch origin
git switch dev
git pull --ff-only origin dev
python -m unittest discover -s tests -v
python run.py preflight --config configs/default.json
python -u run.py benchmark --config configs/benchmark.json 2>&1 | Tee-Object -FilePath benchmark.log
```

Preflight checks device availability, the static Adam memory lower bound, and
estimated output-volume capacity. It prints parameters, FEs, population buffers and
retained-weight estimates. It cannot certify activation/workspace memory or runtime;
the benchmark supplies those measurements. Benchmark can be run even if the full
experiment's disk-capacity check fails, provided its own bounded workspace fits.

Benchmark settings are in configs/benchmark.json:
one warmup step, three timed steps, two full training fitness passes per case,
two sequence modes and first/final training cohorts. Six host optimizers use four
agents for one generation while preserving every weight dimension.
Adjusting these diagnostic repeat counts changes benchmark work, not the main experiment.

The report directory contains Benchmark_Report.md and benchmark.json. Temporary model,
checkpoint and population buffers are removed after the benchmark. A failed/partial
report is retained; there is no silent CPU or smaller-model fallback. If a training
case fails, its error is recorded. Do not run a benchmark alongside another GPU job.

Estimates use measured seconds per real time step, conservative longest-batch Adam
timings and small-population host overhead. Scenario bounds are sensitivity calculations,
not confidence intervals. A population exceeding RAM/page cache may be slower than
the small-population extrapolation. Checkpoint load measurements are warm-cache values.

## Configure one output volume

Every notebook uses configs/default.json. Set data_path/output_root there or pass
--output-root to the CLI. A CLI output override is local to that command; use the same
root for every phase of that experiment. Relative config paths are project-relative.

```powershell
python run.py preflight --config configs/default.json --output-root "E:\ThesisRuns\dev-v3"
```

Use the drive letter actually available on the server. All code/data/scientific settings
are fingerprinted. New output roots protect old runs from incompatible changes.
A running experiment must not pull new code or modify its config.

## Notebook order

| Notebook | Work |
|---|---|
| 00 | Environment/data; optional server benchmark cell |
| 01 | Baseline validation for both sequence modes; development-only choice |
| 02 | DLSTM, logistic regression and training-majority validation |
| 03 | CHOA validation |
| 04 | WCHOA validation |
| 05 | Reconstructed GSOBL-CHOA validation |
| 06 | TAR-CHOA validation |
| 07 | GA and WOA validation; required main comparators |
| 08 | Freeze OOF thresholds/protocol, then final training/evaluation |
| 09 | Ensemble/seed/year metrics, class-specific PR, paired cluster statistics, costs |
| 10 | Recover essential artifacts from saved notebook outputs |

Stored outputs were cleared in the dev templates; historical outputs remain in main's
commit history and the researcher's backups. Recovery functions remain independent.

## Explicit CLI phases after benchmark review

```powershell
python run.py prepare --config configs/default.json
python -u run.py sequence --config configs/default.json
python -u run.py validate --config configs/default.json
python run.py freeze --config configs/default.json
python -u run.py final --config configs/default.json
python run.py analyze --config configs/default.json
```

Individual models can be validated with --models CHOA, --models GA WOA, etc.
Use the same config and root throughout. The all action performs these phases in order
and first enforces preflight's static checks. Running python run.py alone performs
only preflight. The final run remains a later decision following server benchmarking
and the dev-to-main merge.

## Resume and backup

Re-run the same action/config/root. Completed runs are skipped. Adam restores model,
optimizer and RNG state at epoch boundaries. Population search commits alternating
disk slots at generation boundaries. A partial generation may be replayed; actual calls
are logged in physical_evaluations.jsonl and may exceed the committed FE budget.
Old v1/v2 checkpoints are intentionally incompatible with the changed objectives/operators.

timing.json and training_timing.json accumulate active work. A hard kill can lose time
after the last progress checkpoint; the next invocation flags an uncertain previous
session and never counts offline downtime as training.

Completed searches remove population buffers with retain_populations=false. Interruptions
retain them for resume. Windows maps are explicitly closed before cleanup. Back up
model.pt, training_state.pt, predictions, configs and protocol files on durable storage.
Notebook capsules preserve small JSON/CSV artifacts but omit model/population weights.

## Outputs

Each run saves predictions.csv, metrics.json, config.json, scaler.json, sample_ids.json,
history.json and complete.json. Neural runs save model.pt; LR saves model.joblib;
MajorityClass saves majority_model.json. Hybrid diagnostics.json includes bounds,
warm/best fitness, changed parameter count, distance, opposition counters and TAR phases.

Analysis includes model_comparison.csv, seed_metrics.csv, per-year metrics,
compute_costs.csv, shared_warm_training_costs.json, experiment_budget.json,
paired_comparisons.csv and diagnostic plots. Shared warm-training costs are reported
once separately from method-specific search time. Seeds are computational repetitions,
not additional independent company-years.
