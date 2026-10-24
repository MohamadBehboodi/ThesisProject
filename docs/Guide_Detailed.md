# Windows execution guide - final v4

## Environment and launch

Use the existing environment in which both CUDA benchmarks succeeded. This release
adds no third-party runtime dependency. Check the selected interpreter/device:

```powershell
python -c "import sys,torch; print(sys.executable); print(torch.__version__,torch.version.cuda,torch.cuda.is_available())"
```

Update the project before starting any final work:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m unittest discover -s tests -v
python run.py preflight --config configs/final.json
```

After tests and preflight pass, launch the complete experiment:

```powershell
python -u run.py all --config configs/final.json 2>&1 | Tee-Object -FilePath final_v4.log -Append
```

Use your existing persistent terminal session or keep the terminal open. Do not run
several copies of all on the same experiment. Locks prevent conflicting writes but
are not a parallel job scheduler. The command does not provide a background service.

The current profile is [256,128,64,30], batch 32, 30 Adam epochs, population 30 and
5,000 FEs for each of the six optimizers per split/seed. There are three seeds,
six validation folds and nine comparison methods. See Final_Run.md for the measured
cost basis and the adaptation relative to the base paper.

## Timing, space and output volume

The two uploaded server benchmarks completed. At final settings, measured-rate
scenarios are 58.11-78.36 hours for full_year and 32.60-48.68 hours for last100. Sequence
choice still depends on development validation. Plan for roughly 3-5 days allowing
variation and additional work; no fixed finish time is guaranteed. Pauses are excluded
from these active-time estimates.

The static population buffers total about 121.75 MiB, and projected retained neural
artifacts total 572.22 MiB. Prepared data, reports, logs and filesystem overhead add
space beyond the approximately 0.69 GiB static peak. Keeping at least several GiB free
on the destination volume is practical. Preflight checks the device, a static memory
lower bound and estimated disk capacity; it does not resize the experiment.

The canonical root is outputs/final_v4/main. All four main config aliases contain
identical settings and this same root. Old outputs/dev_v3 paths are not reused.
For another SSD, use the same override in every command for this experiment:

```powershell
python run.py preflight --config configs/final.json --output-root "E:\ThesisRuns\final-v4"
python -u run.py all --config configs/final.json --output-root "E:\ThesisRuns\final-v4" 2>&1 | Tee-Object -FilePath final_v4.log -Append
```

Choose a drive letter that exists. Notebook users should instead set the desired
output_root in the shared config before starting, and keep it fixed. Switching only
one phase to a different root creates a different experiment directory.

## Pipeline and notebooks

The all action runs prepare, sequence, validate, freeze, final and analyze in order.
It first enforces preflight. It fits scalers and objectives on training samples,
selects sequence mode using DLSTM development folds, selects exact OOF thresholds,
then evaluates the frozen protocol. Final test outcomes do not choose architecture,
thresholds or budgets. No early stopping or automatic budget reduction occurs.

All notebooks read configs/default.json, which is identical to final.json.

| Notebook | Work |
|---|---|
| 00 | Environment/data; optional repeat benchmark of the current profile |
| 01 | DLSTM validation of both sequence modes and development choice |
| 02 | DLSTM, logistic regression and majority validation |
| 03 | CHOA validation |
| 04 | WCHOA validation |
| 05 | Reconstructed GSOBL-CHOA validation |
| 06 | TAR-CHOA validation |
| 07 | GA and WOA validation |
| 08 | Freeze thresholds/protocol, then final training/evaluation |
| 09 | Ensemble, seed/year metrics, paired cluster statistics and costs |
| 10 | Recover essential artifacts from saved notebook outputs |

The CLI all command replaces manually executing notebooks 00-09. If running individual
phases is preferred, use the same config/root for each:

```powershell
python run.py prepare --config configs/final.json
python -u run.py sequence --config configs/final.json
python -u run.py validate --config configs/final.json
python run.py freeze --config configs/final.json
python -u run.py final --config configs/final.json
python run.py analyze --config configs/final.json
```

Do not execute notebook runs in parallel with the CLI experiment. Historical cost
profiles benchmark.json, benchmark_compact.json and paper_budget.json are not final
config aliases. Optional benchmarks do not produce final predictions or choose a
protocol. With no arguments, python run.py performs preflight only.

## Resume, monitoring and backups

If WinError 5 occurs during atomic file replacement, see Windows_IO_Hotfix.md for
the reviewed runtime-only update and read-only check_resume.py command. It preserves
the published final-v4 experiment identity, including Windows CRLF source checkouts.
This exception applies only to the exact reviewed I/O revisions; other source or
scientific setting changes continue to require a fresh experiment.

Re-run the same all command to resume. Completed runs are skipped. Adam restores model,
optimizer and RNG state at epoch boundaries. Population search commits alternating
slots at generation boundaries. A partial generation can be replayed, so physical
FEs may exceed the nominal committed budget after interruption. Do not delete
population buffers for an interrupted search.

Console output reports Adam progress, committed FEs and best fitness. Each run saves
config.json, metrics.json, predictions.csv, history.json, timing.json and complete.json.
Hybrid diagnostics record warm/best fitness, changed parameter count, effective bounds,
operator counts and TAR phase calls. The physical_evaluations.jsonl ledger records
actual work, including replay. Timers accumulate active work and flag uncertain
hard-kill intervals. Shared warm training is accounted for separately.

Completed searches close and remove population files with retain_populations=false.
Back up model.pt, training_state.pt, predictions, protocol files and the selected config
on durable storage. Capsules in logs/notebook outputs preserve small JSON/CSV evidence,
but omit neural weights and population buffers. Source, scientific config and dataset
changes require a fresh output root; do not update code/config during the final run.

## Final deliverables

The analysis folder contains model_comparison.csv, seed_metrics.csv,
paired_comparisons.csv, compute_costs.csv, experiment_budget.json,
shared_warm_training_costs.json, per-year metrics and diagnostic plots. Report both
profit-positive and loss-positive metrics. Compare hybrids with the shared DLSTM,
logistic and majority baselines. Record architecture/budget deviations from the base
article. Identical warm/best weights or absent improvement are valid findings to
report, not reasons to change the held-out protocol.
