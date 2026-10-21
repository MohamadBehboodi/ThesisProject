# dev v3 validation record

Validated against the development changes based on main commit
fc9a7b9b4545c2fa5158dc56f682965fd217c880. These checks establish implementation
behavior and pipeline execution; they are not final thesis results.

## Automated checks

```powershell
python -m unittest discover -s tests -v
```

All 30 tests pass using Python 3.12.14 and PyTorch 2.6.0+cpu on Linux. The suite covers:

- Exact threshold partitions, tied probabilities, endpoint probabilities and class metrics.
- Hand-calculated CHOA/weighted proposals, nonlinear schedule endpoints, reproducible
  coordinatewise chaos, unchanged warm vectors and strict-bound rejection.
- Current-FE TAR phase accounting and active weakest-half opposition in one dimension.
- Interruption/resumption equivalence for all six optimizers, including mmap closure
  and file deletion after return.
- Shared Adam/full-cohort fitness objectives, exact resumed Adam weights/RNG and
  accumulated active timing.
- Cross-process lock exclusion and release on the native test platform.
- A complete synthetic-data pipeline with all nine comparators, 19 completed runs,
  analysis and deletion of completed population buffers.
- A bounded benchmark with four training-only cases, six host-operator measurements,
  separate reports and temporary-file cleanup. Failure reporting preserves a requested
  CUDA device instead of silently falling back to CPU.
- Full-architecture parameter, artifact and equal-FE schedule counts.

Python modules, JSON configurations and notebook code cells parse successfully.
The release manifest covers tracked files except itself, and unchanged data/vendor
hashes are verified. Whitespace checks pass.

## Complete supplied-data smoke run

Executed the complete CLI pipeline with configs/smoke.json and an isolated output root:

```powershell
python -u run.py all --config configs/smoke.json --output-root <fresh-smoke-folder>
```

The run uses the full supplied CSV, CPU, widths [8,6,4,3], one epoch, seed 17,
population four, eight FEs per optimizer except TAR's 20-FE smoke budget, and one
validation year. It completed with exit code zero, selected full_year, froze nine
OOF thresholds and produced 19 completed runs. The analysis contains all nine models,
loss-class metrics, cost accounting, convergence plots and evaluation plots. The
four-panel evaluation layout was inspected. These tiny settings must not be used to
claim model quality, final GPU speed or feasibility of the large experiment.

## Development-host mmap observation

The long-running command host can re-materialize a memory-mapped file after the Python
process has closed and deleted it. This was reproduced independently: create a NumPy
memmap, allow the command to yield, close/unlink it, observe absence inside the process,
then observe the file reappear after command completion. A non-yielding control does
not show this behavior. A traced supplied-data run confirmed both population files
were closed and absent immediately after deletion; fixture cleanup tests also pass.
This host-specific observation is recorded separately from application cleanup.

## Required server follow-up

Native Windows locking/deletion and CUDA execution have not been tested here.
The [server guide](Guide_Detailed.md) provides tests, preflight and the bounded benchmark.
Review Benchmark_Report.md and benchmark.json from the intended GPU and output SSD
before agreeing final widths, population or FE budgets. No final GPU experiment has
been launched, and dev has not been merged into main by this update.
