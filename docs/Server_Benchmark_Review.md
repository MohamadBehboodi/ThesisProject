# Server benchmark review - 2026-10-21

Historical review of the first benchmark. The compact calibration described below
subsequently completed; the adopted final profile and launch are in Final_Run.md.

The supplied Benchmark_Report.md and benchmark.json report completed execution on
Windows with CUDA, using dev commit 6bb1d27. This confirms the measured training and
fitness cases ran; it does not certify the full experiment's runtime, free disk
capacity or model performance.

## Measured large-model result

The architecture [4000,2000,2000,30] has 144,412,092 trainable parameters. Dedicated
GPU memory is about 10 GiB. Measured peak allocated training memory was 3.022 GiB;
the largest reserved peak was 3.852 GiB.

| Sequence mode | First-fold fitness seconds | Final-training fitness seconds | Full schedule scenario, hours | Continuous years (365 days) |
|---|---:|---:|---:|---:|
| full_year | 13.6530 | 29.8434 | 56,500.67-78,348.78 | 6.45-8.94 |
| last100 | 6.4605 | 13.2369 | 36,415.92-57,220.42 | 4.16-6.53 |

Each FE evaluates the complete training cohort. The full plan contains six optimizers
times seven splits times three seeds times 50,000 FEs = 6,300,000 FEs. At the measured
29.8434 seconds, 50,000 final-cohort FEs alone take 17.27 days for one method and seed,
excluding proposal construction, storage and warm training.

Independent reconstruction of training-cohort lengths and measured seconds per step
gives about 36,934-39,229 hours of fitness work for full_year and 16,850-18,102 hours
for last100, before host overhead. Thus the multi-year conclusion is not just an
artifact of extrapolating the four-agent host benchmark.

Measured host proposal/storage overhead is also material: about 2.29-20.09 seconds
per post-initialization FE, depending on method. The 200-agent two-buffer footprint
is 215.19 GiB, larger than the server's reported 64 GB RAM. Paging/storage pressure
can make full-population operation slower than the small-population extrapolation.
The static peak disk estimate is 370.13 GiB; the uploaded report does not report
available capacity on the destination volume.

These ranges are cost scenarios, not confidence intervals. Reducing only Adam epochs
would not resolve the search cost: the report estimates roughly 12-13 hours of warm
training, against tens of thousands of search hours.

## Next diagnostic profile

configs/benchmark_compact.json is a proposed calibration profile, not an approved final
experiment or a claim of equivalence to the base paper's architecture:

| Setting | Candidate |
|---|---|
| Hidden widths | [256,128,64,30] |
| Trainable parameters | 531,932 |
| Batch size | 32 |
| Proposed population | 30 |
| Proposed FEs per method/split/seed | 1,000, including initialization |
| Proposed total hybrid FEs | 126,000 |
| Adam epochs | 30 |
| Seeds / validation folds / optimizers | 3 / 6 / 6 |

Both sequence modes are measured. The final sequence choice would still use the
development-validation protocol. All methods use the common objective and warm start.
The proposal reduces network capacity and search budget; these deviations must be
explicitly documented in the thesis if adopted. No convergence or accuracy claim is
made from this cost diagnostic. The proposed budget includes population initialization;
1,000 FEs are not 1,000 generations and are not an established convergence guarantee.

The candidate's static two-buffer footprint is 121.75 MiB; projected retained neural
artifacts are 572.22 MiB. Static peak disk is about 0.69 GiB, excluding prepared data,
reports and filesystem overhead. Runtime cannot be obtained by simply scaling the
large-model time by the parameter-count ratio; batch, launch and recurrent computation
costs must be measured on the server.

Run only this bounded diagnostic next:

```powershell
git fetch origin
git switch dev
git pull --ff-only origin dev
python -u run.py benchmark --config configs/benchmark_compact.json
```

It uses a separate outputs/dev_v3/benchmark_compact root and the same bounded counts
of training/fitness repetitions as the first benchmark. The host probe still uses
four agents. Schedule scenarios extrapolate to the proposed 30-agent/1,000-FE plan;
the benchmark itself does not run that full plan, use held-out outcomes or freeze a
final protocol. Review the newly printed Benchmark_Report.md and benchmark.json
before setting a final profile and merging dev into main. Existing default/final
profiles are unchanged by this diagnostic addition.
