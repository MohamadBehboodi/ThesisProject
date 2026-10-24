# Implementation validation record

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

## Server evidence and final-profile update

The researcher subsequently supplied two completed native Windows/CUDA benchmark
reports. These exercise bounded Adam steps, full training-cohort fitness passes,
checkpoint I/O and all six host optimizers with temporary-directory cleanup. They
support the adopted [256,128,64,30], batch-32 final profile. Native Windows interruption
recovery and the complete final schedule are not established by these bounded reports.

The final profile increases the measured 1,000-FE schedule to 5,000 equal FEs per
method/split/seed. The resource-count test now expects 531,932 parameters and 630,000
FEs, with 39 shared warm runs and 165 saved neural models. The same 30-test suite
passed again for this release (27.921 seconds, Linux/Python 3.12.14/PyTorch 2.6.0+cpu).
The adopted model's parameter count and forward shape were also checked. Config aliases,
module/notebook syntax, manifest and unchanged
data/vendor hashes are also checked. No final GPU experiment is launched by this
development workflow. See Final_Run.md for the server launch after tests/preflight.

## Windows atomic-I/O hotfix - 2026-10-24

The reported WinError 5 during timing-file replacement led to the operational-only
patch described in Windows_IO_Hotfix.md. The complete expanded suite ran 43 tests:
42 passed and one Windows-only real-handle test was skipped on Linux (13.763 seconds).
Tests include transient/permanent denial, old-target preservation, bounded retries,
timer throttling, original exception preservation, exact final-v4 identity continuity,
LF/CRLF compatibility, read-only auditing and deterministic checkpoint replay.
The numerical model, dataset processing and final settings remain unchanged. Native
Windows lock-owner diagnosis and execution of the real-handle test remain server-side.
