# Scientific implementation decisions - final v4 profile

## Scope and accepted data design

This is an independent research implementation. Scientific mechanisms and transparent,
testable choices take priority over copying the authors' source byte for byte.
The researcher has verified collection dates, temporal alignment and availability.
The first ten annual accounting ratios are intentionally repeated in daily market rows.
The supplied label is company-year t -> profitability t+1; it is not shifted again.

The 18 features, >=100-day eligible cohort, chronological ordering, train-only scaling,
true-length packing, full-year/last100 development selection, existing embargo/folds,
test years and seeds are retained. After the Windows benchmarks, widths are set to
[256,128,64,30], batch size 32, population 30 and a matched budget of 5,000 FEs.
This resource-based choice precedes final-model fitting and does not use held-out outcomes.
Previous discussion of missing publication
timestamps is not treated as a data-validity blocker after the researcher's confirmation.

## Model and common objective

Three native LSTM layers (256,128,64), a Dense hidden layer of 30 and two output logits.
Sigmoid/tanh remain inside LSTM gates; ReLU and dropout .1 follow recurrent outputs;
the dense hidden layer uses ReLU. This is a declared interpretation of the base paper's
layer description. The final network has 531,932 trainable parameters for 18 inputs.
The original [4000,2000,2000,30] interpretation has 144,412,092 parameters and remains
in the historical large-model benchmark profile. Its measured multi-year full-plan
cost led to the explicitly documented capacity/budget adaptation. The thesis does
not claim exact architecture or budget reproduction of the base article, nor that
the smaller model has equivalent predictive accuracy. See Final_Run.md for evidence.

Default data loss is 0.5*sqrt(mean((softmax(logits)[:,1]-y)^2)). The squared-error mean
has a 1e-12 floor only to avoid an undefined zero-root derivative. Cross entropy is an
optional common-objective profile via fitness=cross_entropy; it uses logits for numerical
stability. Adam and every metaheuristic use the same selected objective plus coefficient
1e-6 times the sum of absolute first-LSTM input and recurrent weight entries.
Adam uses mini-batch estimates; metaheuristics score the full training cohort in eval mode.
Dropout is disabled for fitness/inference. No validation labels enter fitness.

The warm start uses 30 Adam epochs, then a direct search of all trainable weights/biases
without further gradient training per candidate. This two-stage protocol is our explicit
reconstruction. Final training has a fixed epoch count and does not use test-driven
early stopping. Class losses remain unweighted in the primary reconstruction; class
weighting is a possible separately predeclared sensitivity experiment.

## Shared bounds and initialization

A common warm checkpoint is used per split/seed/sequence mode. Row zero is copied exactly.
Requested symmetric bounds start at [-1,1]. With weight_bound_policy=expand_to_initial,
the effective absolute bound is max(requested bound, nextafter(max(abs(warm)),+inf)) when
necessary. All methods derive the same effective bound from the same warm vector.
The bound and out-of-requested-range count are recorded. policy=error instead stops
before population creation. The baseline is never silently clipped.

Remaining agents use clipped warm_normal perturbations with std=.01. Uniform initialization
remains available. All searches use disk-backed float32 vectors at the full parameter dimension.

## CHOA group dynamics and chaotic-position branch

Four leader vectors are selected by stable ascending fitness, including the archived best.
The driver proposal is based on the driver. These correct obvious source-code mistakes.
Agents are assigned once to four balanced shuffled groups using their optimizer RNG;
group assignments are checkpointed. Coefficients follow four ChOA2-inspired curves.
For q=committed_FE/budget, B=budget, the endpoint-normalized schedules are:

```text
f1(q) = 2.5 * (1 - log(1 + (B-1)*q)/log(B))
f2(q) = 2.5 * (1 - q^3)
f3(q) = 2.5 * (exp(-16*q^2) - exp(-16)) / (1 - exp(-16))
f4(q) = 2.5 * (1-q)^2
```

These retain the source table's logarithmic/cubic/exponential/quadratic shapes while
removing its terminal .5 floor to obtain the base article's stated 2.5-to-zero range.
Endpoint normalization and FE-based progress are our declared choices, not exact author code.

The normal four-leader proposals use A=2*f*r1-f, C=2*r2, and
lambda*leader-A*abs(C*leader-m*x), averaged across leaders for CHOA/TAR.
The scalar distance-map m follows a carried Gauss map, with a deterministic restart
at .7 for a finite-precision near-zero state.

With probability .5 a candidate instead uses a chaotic-position branch (base Eq.7).
Each coordinate begins with an independent reproducible seed in (0,1), receives three
vectorized Gauss-map iterations, then maps to [-bound,bound].
Coordinatewise seeded iterations avoid broadcasting one scalar to millions of weights.
The map initialization/iteration count are explicit implementation choices.
The branch, groups, leader selection and RNG are identical mechanisms across CHOA variants.

## WCHOA

Normal updates use normalized magnitude weights w_k=abs(X_k)/sum_j(abs(X_j));
a zero denominator receives equal weights. The result is sum_k(w_k*X_k) without
an additional division by four. This removes duplicated abs(X1) numerators in the
supplied MATLAB code and gives a well-defined weighted combination. We document this
as the reconstructed weighted operator, not a claim of exact source identity.
The same chaotic-position branch applies.

## Reconstructed GSOBL-CHOA

The old population-normalized coordinate-distance gate was removed. It could make
opposition impossible for many combinations of population and vector dimension.
Each generation selects the weakest ceil(N/2) agents using stable fitness ranking.
For these agents, standard opposition x_opp=L+U-x (here -x for symmetric bounds)
is evaluated and accepted only if it improves that agent. A subsequent weighted chimp
proposal is also evaluated if FE budget remains and retained only if better.
Attempts, accepted oppositions and vector changes are logged.

Weakest-half selection is an agreed reconstruction choice. It is not attributed to the
original authors as an exact equation and is labeled reconstructed GSOBL-CHOA in the
research description. It preserves the chosen combination of weighting, opposition
and greedy acceptance without a dimension-dependent inactive gate.

## TAR-CHOA phases

Eqs14-15 use the literal exponent:
1-tan((r-.5)*pi*theta/max_FE). Numerical clamps prevent overflow; optional stated_ranges
is a distinct declared policy. The printed expressions disagree with the prose ranges,
so literal is the primary documented interpretation.

Per agent: evaluate the stochastic alternative and greedily retain it; evaluate the base
chimp proposal; determine the CURRENT FE fraction immediately before the weighted proposal;
evaluate only lambda1 for q<.5, otherwise lambda2. Choose the best eligible incumbent/base/
active-weighted proposal. An inactive lambda proposal is neither evaluated nor archived.
The archived best can therefore no longer bypass phase restrictions. No call exceeds the
committed FE budget. A budget ending mid-agent retains the best eligible evaluated option.

Stochastic alternative coordinate selection uses tan(pi*(r-.5)) < 1-q with uniform bounded
replacement, an explicit choice where distribution is unspecified. Theta starts at one
per agent, halves on strict improvement of that agent and increments by one otherwise.
TAR normally consumes up to three FEs per agent update, versus the previous four-proposal
implementation. Every actual call, including replay after interruption, is logged.

## GA, WOA and the majority baseline

GA is a real-coded genetic algorithm: three-way tournament parent selection, per-coordinate
arithmetic crossover, Gaussian coordinate mutation (probability .01, std .02), greedy
survivor selection and best archive. WOA uses its standard encircling/random-search/spiral
operators, A=2*a*r-a, C=2*r, a=2*(1-q), spiral b=1 and l in [-1,1].
They search the same full weight vector with common warm start, objective, bounds and FE budget.

GA and WOA are primary comparators under their actual names. GA_reference and WOA_reference
remain accepted API aliases only. We do not relabel these methods as ARGA/ARWOA, whose
definitions are not established by the base paper. The substitution is stated in the thesis.

MajorityClass fits only training-label counts; ties select class zero. It predicts that
class for every evaluation sample. On this dataset the majority is profitable.
LR retains its fixed 34 window summaries, train-only scaling and C=1.

## Fair comparison and threshold selection

Default/final/matched/with_classical_comparators are identical aliases of the final
profile, allocating 5,000 FEs to each of the six metaheuristics with population 30.
Across six validation folds plus final training and three seeds this is 630,000 FEs.
The 30 initial-agent evaluations are included in each budget. More evaluations than
the 1,000-FE calibration plan give additional search opportunity within the user's
few-day compute allowance; this is not a convergence guarantee. Convergence curves
and warm-start improvement must be reported even when no improvement is observed.
TAR's 300,000-FE paper setting is a separate historical paper_budget profile with
the original large architecture; it is not part of the final run.
FEs include population initialization and all additional opposition/TAR proposals.
Equal FEs do not imply equal wall-clock cost.

Thresholds maximize development OOF seed-ensemble Balanced Accuracy over every achievable
decision boundary. Equal probabilities move together. BA ties prefer the attainable
threshold nearest .5, then the lower threshold. The p>=threshold convention is preserved;
nextafter(1,+inf) is a finite sentinel for an all-negative decision when p=1 occurs.
No test labels set thresholds. Sequence selection remains based on mean validation
fold/seed BA at .5, with a full-year tie break.

## Reporting, persistence and limits

Metrics include class-specific loss precision/recall/F1 and loss-positive average precision
in addition to primary BA, profit-class metrics, macro F1, AUC, Brier and log loss.
Plots distinguish profit-positive and loss-positive PR curves.
Company-cluster bootstrap, paired model-swap randomization and Holm adjustment are retained.
Seeds are not independent company-year samples; previously examined test years remain
a controlled reanalysis of that period.

Epoch checkpoints contain Adam and Python/NumPy/Torch/CUDA RNG state. Alternating population
slots checkpoint groups, theta, Gauss state, RNG, counters and best vector each generation.
Memory maps are explicitly closed on success and exception, supporting Windows deletion.
Cumulative timers exclude downtime; hard kills may lose time after the last checkpoint
and are flagged on the next invocation. Physical FE logs include replayed work.
Shared warm cost is reported separately to avoid charging it once to every hybrid.

Changes in sources/data/scientific config require a new output_root. The final profile
uses outputs/final_v4/main; dev benchmark/smoke profiles use outputs/dev_v3/...
and never overwrite previous experiment roots. GPU reproducibility
across hardware/PyTorch releases is not guaranteed. Architecture and budgets are not
silently reduced. The benchmark measures training-only work and produces scenario
estimates; it cannot guarantee a full-population completion time.

## Evidence

Base article: Tang, Yang & Khishe (2023), Heliyon e19431,
https://doi.org/10.1016/j.heliyon.2023.e19431 .
Author archives in vendor are unchanged. CHOA group shapes also follow the supplied
Brief_Mathematical_Model.pdf inside ChOA-V2.zip. See Sources.md for publication/release links.
