# Final v4 selection and launch

On 2026-10-21, the researcher authorized a moderate network and a complete schedule
lasting a few days. The choice below uses training-only timing and memory reports,
without inspecting new validation or final-test outcomes.

## Evidence and decision

Both server benchmarks completed on native Windows/CUDA. The original 144,412,092-
parameter interpretation produced multi-year full-plan scenarios. The second benchmark,
on commit df918fa, measured [256,128,64,30] with batch 32 and population-30/1,000-FE
schedule scenarios. It estimated 11.76-15.79 hours for full_year and 6.65-9.84 hours
for last100. Training peaks were at most 0.194 GiB allocated on the approximately
10-GiB dedicated-memory RTX 3080.

The final model keeps these measured widths and batch size. Three recurrent layers
and 531,932 trainable parameters provide a moderate capacity candidate; sufficiency
and predictive quality remain empirical questions for development validation and
the frozen evaluation protocol. Low GPU-memory consumption alone is not a reason to
increase width. There is no claim that this capacity is optimal or equivalent to the
base article's large architecture.

Search budget increases to 5,000 FEs for each method, split and seed, with population
30. All six optimizers have identical FE allowances, including 30 initialization
evaluations and every additional proposal. This creates 630,000 hybrid FEs across
six validation folds plus final training and three seeds. It gives more search
opportunity than the calibration scenario, but does not establish convergence.
Training stays at 30 epochs. All nine comparators and both development sequence
choices remain in the planned study.

Recomputing the estimator with these settings gives:

| Selected sequence | Active-time scenario, hours | Continuous days |
|---|---:|---:|
| full_year | 58.11-78.36 | 2.42-3.26 |
| last100 | 32.60-48.68 | 1.36-2.03 |

These are measured-rate extrapolations to five times the FEs, with warm training,
host operators and checkpoint I/O included as in the estimator. They are not a timed
execution of the full plan or a statistical confidence interval. Allow roughly 3-5
days for operational variation and additional work. The estimator excludes LR,
majority and statistical-analysis time; interruption replay and pauses can add time.
The full plan is not yet executed on the server.

The machine-readable evidence, benchmark checksum, adopted settings, counts and
estimates are in Final_Run_Plan.json. The base-paper architecture, population and
budget deviations must be stated in the thesis. Convergence, minority-class metrics
and any absent improvement over the warm baseline must be reported transparently.

## One final configuration

configs/final.json, default.json, matched_budget.json and with_classical_comparators.json
are byte-identical aliases of FINAL_V4_MAIN, using outputs/final_v4/main. The benchmark
configs preserve previous measurements. The original paper_budget profile remains
an expensive optional reference and is not selected by all. No old benchmark or
smoke model/checkpoint is reused for final training.

## Start on Windows

In the project folder and the existing working Python environment:

```powershell
git fetch origin
git switch main
git pull --ff-only origin main
python -m unittest discover -s tests -v
python run.py preflight --config configs/final.json
python -u run.py all --config configs/final.json 2>&1 | Tee-Object -FilePath final_v4.log -Append
```

Start all after tests/preflight pass. Use a persistent terminal session or keep the
terminal open. The all action performs the full development/final pipeline in order;
a further benchmark is unnecessary for these measured dimensions and batch size.
To resume, repeat the last command without changing the config, code or output root.
See Guide_Detailed.md for output-volume overrides, individual phases and backups.

After completion, review outputs/final_v4/main/analysis/model_comparison.csv,
seed_metrics.csv, paired_comparisons.csv, compute_costs.csv and all hybrid diagnostics.
The frozen protocol and saved predictions provide the basis for Chapter 4.
