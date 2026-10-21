# DLSTM stock-profit prediction - dev v3

Research project by Mohamad Hasan Behboodi, Allameh Tabataba'i University.

This branch implements the agreed independent reconstruction of DLSTM with
CHOA, WCHOA, reconstructed GSOBL-CHOA, TAR-CHOA, real-coded GA and standard WOA.
The aim is scientific fidelity with documented implementation choices.
Financial-data timing and the annual/daily merge were verified by the researcher;
the annual ratios are intentionally constant within each company-year.

## Update the Windows server

Run these commands inside your existing project folder and activated Python environment:

```powershell
git fetch origin
git switch dev
git pull --ff-only origin dev
python -m unittest discover -s tests -v
python run.py preflight --config configs/default.json
```

The current CUDA-enabled PyTorch installation can be retained. This revision adds
no third-party runtime dependency. See requirements.txt for the supported versions.

## Run the limited server benchmark first

```powershell
python -u run.py benchmark --config configs/benchmark.json 2>&1 | Tee-Object -FilePath benchmark.log
```

The command prints the location of **Benchmark_Report.md** and **benchmark.json**.
They are saved in a timestamped folder under
outputs/dev_v3/benchmark/benchmarks. Send both reports for review before choosing
the final architecture and budget.

The benchmark keeps widths **[4000, 2000, 2000, 30]**. It measures a few Adam steps,
full fitness passes on training cohorts, GPU peaks, checkpoint I/O, and the six
optimizers' host overhead using a small population with the full vector dimension.
It produces no validation/test predictions and does not freeze a final protocol.
CPU smoke timing is not a substitute for this CUDA measurement.

To use another SSD, choose an existing drive letter and override only the output path:

```powershell
python -u run.py benchmark --config configs/benchmark.json --output-root "E:\ThesisRuns\dev-benchmark"
```

The full population has two buffers totaling about **215.19 GiB**. The preflight
also includes retained checkpoints/model weights in its disk estimate. Free capacity
must be on the output volume. It does not pool free space across SSDs.

## Profiles and execution

| Config | Purpose |
|---|---|
| configs/default.json | Canonical dev profile; all comparators; equal 50,000 FE budgets |
| configs/benchmark.json | Same large architecture/settings, bounded diagnostic work |
| configs/smoke.json | CPU execution check only; tiny network; not thesis results |
| configs/final.json | Separate final-candidate output root; same matched scientific settings |
| configs/matched_budget.json | Explicit matched-budget profile |
| configs/paper_budget.json | Optional sensitivity profile: TAR 300,000 FE, others 50,000 |
| configs/with_classical_comparators.json | Compatibility profile; GA/WOA are now included in every main profile |

Every notebook reads **configs/default.json**. Notebook 07 is now required for GA/WOA.
Notebook 00 also has an opt-in benchmark cell. Notebook 02 includes the training-majority
baseline. See docs/Guide_Detailed.md for the phase order.

With no arguments, **python run.py performs preflight only**. After benchmark review
and the agreed dev-to-main merge, the explicit complete-run command is:

```powershell
python -u run.py all --config configs/default.json 2>&1 | Tee-Object -FilePath final_run.log
```

There is no automatic architecture, budget or CPU fallback. dev is not merged into
main by this update. Update code/configuration only between experiments; use a fresh
output_root after a scientific change. Existing outputs/main_windows and outputs/final_v1
are not reused.

## Main changes

- Shared Half-RMSE plus first-layer L1 for Adam and all metaheuristic fitness calls.
- CHOA group schedules and a coordinatewise Gauss-mapped chaotic-position branch.
- TAR evaluates only its active lambda phase, determined from the current FE count.
- Reconstructed GSOBL: standard opposition on the weakest half, greedy acceptance.
- Exact OOF threshold search, tied probabilities handled as a group.
- Common deterministic bounds expanded to contain the unchanged warm start, or an explicit error policy.
- GA and WOA are main comparators under their real names; ARGA/ARWOA are not claimed.
- Class-specific loss metrics and a training-majority baseline.
- Active-time accumulation, physical FE ledger, objective components, vector-change
  diagnostics and explicit Windows memory-map cleanup.

Detailed formulas, limitations and traceability are in docs/Method_Decisions.md.
Model weights and resumable checkpoints need separate backups; notebook capsules
recover essential tabular/JSON artifacts, not neural weights.
