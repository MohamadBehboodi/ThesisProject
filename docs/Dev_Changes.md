# dev v3 change record

The main branch is unchanged. Work proceeds on dev; a later merge and final run follow
Windows server benchmark review.

## Agreed scope implemented

- Researcher-verified temporal data/annual-ratio design accepted and preserved.
- Half-RMSE plus L1 shared by Adam and all metaheuristics.
- Explicit nonlinear CHOA groups and coordinatewise Gauss-position branching.
- Reconstructed weakest-half opposition and greedy selection for GSOBL.
- Current-FE TAR phase selection; inactive proposals cannot become the best vector.
- Exact OOF thresholds and unchanged warm start within recorded common effective bounds.
- GA, WOA and a training-majority baseline included in main configs/notebooks.
- Equal-FE primary profiles; larger TAR budget retained as a separate sensitivity profile.
- Loss-class metrics, objective components, vector-distance and operator counters.
- Accumulated active timing, resumed-work accounting, Windows-safe map cleanup.
- Bounded training-only server benchmark; static resource/disk preflight.
- One Windows guide and one default config; no-argument entry point performs preflight only.
- Fresh dev experiment roots and regenerated release manifest.

## Scientific choices

These are independent, documented reconstructions. Endpoint-normalized CHOA schedules,
chaotic-vector seeding and weakest-half opposition are not claimed to be byte-identical
author source. GA/WOA are not labeled ARGA/ARWOA. The dataset and final architecture widths
are retained. No final CUDA training or Windows GPU timing was executed in the development
container.

## Next review

1. Switch the server to dev and run the tests/preflight.
2. Run benchmark with configs/benchmark.json on the intended GPU/output SSD.
3. Review Benchmark_Report.md and benchmark.json; choose feasible common budgets/architecture.
4. Make any agreed follow-up commits on dev.
5. Merge dev into main after review, then launch the explicitly selected final experiment.

The benchmark does not choose settings from evaluation outcomes and does not auto-start
the final experiment.
