# DLSTM stock-profit prediction - final v4

Research project by Mohamad Hasan Behboodi, Allameh Tabataba'i University.

The final profile was chosen from two training-only benchmarks on the Windows RTX 3080
server and the researcher's few-day compute allowance. It uses three LSTM layers and
compares DLSTM, logistic regression, training-majority, CHOA, WCHOA, reconstructed
GSOBL-CHOA, TAR-CHOA, real-coded GA and WOA. Data timing and the annual/daily merge were
verified by the researcher.

## Adopted final profile

| Setting | Value |
|---|---|
| LSTM widths / dense width | 256, 128, 64 / 30 |
| Trainable parameters | 531,932 |
| Batch size / Adam epochs | 32 / 30 |
| Population / FEs per optimizer, split and seed | 30 / 5,000 |
| Seeds / validation folds | 3 / 6 |
| Total hybrid FEs | 630,000, including initial populations |
| Sequence choice | full_year versus last100, selected by development validation |
| Canonical output root | outputs/final_v4/main |

The network retains depth with moderate widths. Architecture and search budgets differ
from the base article; these are documented resource adaptations. Timing does not
establish predictive accuracy, convergence or superiority over the baselines.

The 1,000-FE calibration report estimated 6.65-15.79 hours for the complete schedule.
Scaling its measured rates to the final 5,000-FE budget gives:

| Selected sequence | Estimated complete schedule |
|---|---|
| full_year | 58.11-78.36 hours, about 2.4-3.3 days |
| last100 | 32.60-48.68 hours, about 1.4-2.0 days |

Allow roughly 3-5 days for uninterrupted work and operational variation. These are
scenarios rather than completion guarantees. See [Final_Run.md](docs/Final_Run.md) and
[the machine-readable plan](docs/Final_Run_Plan.json).

## Launch on the Windows server

Inside the project folder and the existing working CUDA-enabled Python environment:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m unittest discover -s tests -v
python run.py preflight --config configs/final.json
python -u run.py all --config configs/final.json 2>&1 | Tee-Object -FilePath final_v4.log -Append
```

Start the last command after tests and preflight pass. Use your existing persistent
terminal session or keep the terminal open throughout the run. The all action prepares
sequences, validates both sequence modes, selects a mode, validates all methods, freezes
OOF thresholds, trains/evaluates the final models and creates the statistical analysis.
No additional benchmark is required for these measured widths and batch size.

To resume, repeat the same all command with the same config and output root. Completed
runs are skipped; interrupted work resumes from committed checkpoints. Do not pull
code or edit scientific settings during a run. Old dev benchmark and smoke roots are
separate from the new final root. With no arguments, python run.py performs preflight only.

## Configuration and artifacts

configs/default.json, final.json, matched_budget.json and with_classical_comparators.json
are identical aliases of this one final profile and output root. All notebooks read
default.json, so notebook and CLI execution share the same experiment.

configs/benchmark.json and benchmark_compact.json preserve the two historical cost
measurements. configs/paper_budget.json preserves the original expensive architecture
and asymmetric paper budget as an optional reference; it is not part of the final run.
configs/smoke.json remains a tiny CPU execution check, not a source of thesis results.

Results appear in outputs/final_v4/main/analysis. Main tables are model_comparison.csv,
seed_metrics.csv, paired_comparisons.csv and compute_costs.csv. The root also preserves
sequence_choice.json and protocol_locked.json. Keep model.pt, training checkpoints,
predictions and protocol/config files in backups; notebook capsules omit neural weights
and population buffers.

See [the Windows guide](docs/Guide_Detailed.md), [method decisions](docs/Method_Decisions.md),
[validation record](docs/Dev_Validation.md) and [first benchmark review](docs/Server_Benchmark_Review.md).
