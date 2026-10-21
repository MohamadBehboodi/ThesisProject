## Evidence hierarchy

1. Tang, Yang, Khishe (2023), Heliyon e19431, https://doi.org/10.1016/j.heliyon.2023.e19431
2. Author MATLAB releases supplied by the user: ChOA-V2.zip, WChOA.zip, GSOBL-ChOA.zip.
3. CHOA mathematical note included in ChOA-V2/Brief_Mathematical_Model.pdf.
4. Explicit implementation choices below where the above do not determine a unique method.

## Traceability table

| Item | Evidence / implementation | Status |
|---|---|---|
| Hidden widths | Section 6: 4000,2000,2000,30 | retained |
| Hidden layer types | Three recurrent LSTM layers followed by dense layer of 30 | interpretation; paper is ambiguous |
| ReLU | Applied to outputs of each LSTM and dense hidden layer | interpretation; sigmoid/tanh remain inside PyTorch LSTM gates |
| Output | Two logits; Softmax probabilities | retained; train CE consumes logits |
| Input features | 18 user features, not the paper's 15 | dataset adaptation |
| Dropout | 0.1 after each of the three recurrent layers | Section 6 |
| L1 | First recurrent layer input/recurrent weight matrices; coefficient 1e-6 | paper specifies first layer, not coefficient or reduction; chosen sum reduction |
| Gradient epochs | Adam 30 epochs, same baseline checkpoint for all hybrids within split/seed | 30 retained; connection to optimizer unspecified in paper |
| Hybrid training protocol | Adam warm start then full-vector metaheuristic search; no further gradient training per candidate | explicit two-stage reconstruction, NOT verified original protocol |
| Parameters searched | All trainable weights and biases, including recurrent matrices and both PyTorch LSTM bias vectors | direct search, NOT hyperparameter search |
| Fitness | 0.5 sqrt(mean((P(class=1)-y)^2)) + L1 | Eq19 interpreted for probabilities; equals 2-class one-hot RMSE averaged over classes |
| Baseline loss | Cross entropy + same L1 | chosen classification training objective; not identical to metaheuristic fitness |
| Search population | 200 | Table3 states this for CHOA; inherited for other variants as a comparison choice |
| CHOA/WCHOA/GSOBL budget | 50,000 objective evaluations | 200×250 for CHOA; equal FE for variants, NOT 250 generations for every variant |
| TAR budget | 300,000 FE | Section5.2; reported separately, not compute-matched to 50,000-FE methods |
| Initial population | Clipped warm vector plus Normal(0,0.01) perturbations; first candidate warm vector | unspecified by paper; bounds [-1,1] are chosen |
| Sequence | Full-year versus last100, selected using baseline temporal validation only | not specified in paper; documented data design |
| Short company-years | Exclude <100 observations in both modes | retain prior cohort definition; padding does not require this threshold |
| Preprocessing | Prior fixed log1p market transforms, signed log1p PE; train-only scaling | dataset-specific, not claimed from paper |
| Temporal split | Validation feature years1396–1401; test1402–1403; default one-year label gap | replaces random70/30; necessary conservative prediction design |
| LR | 34 fixed summaries from same observation window, train-only StandardScaler, C=1 | requested additional statistical baseline; not a time-ordered sequence model |
| Threshold | Development OOF seed-ensemble Balanced Accuracy on fixed grid0.05–0.95 | chosen, frozen before final test |
| Metrics | Balanced Accuracy primary; F1, macroF1, ROC-AUC, average precision, etc. | classification adaptation requested by user |

## Important financial timing limitation

The supplied t+1 labels are accepted without another shift. A label gap of one
feature year is used: validation1396 trains through1394; test1402–1403 trains
through1400 with labels through1401. Test1403 uses that same fixed model, not a
new model retrained on test1402. The retained cohort is unchanged by sequence
length.

## CHOA and WCHOA corrections / choices

- Leaders are selected by ascending fitness; the archived best can join them.
  This replaces the MATLAB `fitness > Driver_score` condition starting at Inf,
  which cannot update Driver for finite fitness. Four leaders are actually used.
- X4 starts from Driver, not Chaser while measuring distance to Driver.
- A=2*f*r1-f, C=2*r2 with uniform r1,r2, and f=2*(1-FE/budget).
  This follows the basic coefficient equations; it does not reproduce the
  MATLAB group-specific multipliers nor the alternative f curves in the note.
  The ambiguous group assignment is not claimed resolved.
- The Gauss map is carried between generations (one shared m per generation),
  starting at0.7; a near-zero absorbing state resets to0.7. The published
  `chaos(3,1,1)` instead returns0.7 every call. This is a documented correction/
  choice, not bitwise author-code emulation.
- WCHOA uses w_k=abs(X_k)/sum(abs(X_j)), then sum(w_k X_k), with equal weights
  at zero denominator. This corrects repeated abs(X1) numerators and the extra
  division by4 in the published code. Original weighted paper equations still
  require author confirmation; this normalized interpretation is labeled here.
- Candidates and all weights are clipped to symmetric bounds. Updating rows is
  synchronous per generation; the archived best is preserved separately.

## GSOBL-CHOA reconstruction

The source corOppose path is retained in structure: first population row as
reference, threshold f, squared coordinate-distance statistic normalized by
N(N²−1), and opposition of selected coordinates from MATLAB row4 onward.
Vector `size` comparisons are replaced by scalar counts. Symmetric bounds make
the opposite coordinate -x. We explicitly evaluate and greedily accept an
improving opposite, then evaluate the weighted chimp proposal and greedily
retain the better candidate. This acceptance/extra FE is a reconstruction of
the described greedy step, not the unchanged public code. Because the published
corOppose formula uses N, not rank correlation of fitness, we do not relabel it
as a statistically valid Spearman coefficient. The source `upper/lower` expansion
does not alter fixed bounds when proposals are clamped; fixed bounds are used.

## TAR-CHOA reconstruction and unresolved contradictions

Eqs14–15 are parsed as:

    exponent = 1 - tan((r - 1/2) * pi * theta / max_FE)
    lambda1 = (1 - FE/max_FE)^exponent
    lambda2 = (2 - 2*FE/max_FE)^exponent

The equations' values do not always match the prose ranges [0,1] and [0.5,1].
Default `tar_lambda_policy=literal` follows the printed expression, with numerical
clamps only: exponent[-20,20], log-result[-40,40], base>=1e-12 and q<1. Optional
`stated_ranges` clips to the prose ranges, and changes the experiment fingerprint.
Neither interpretation is claimed confirmed by the author.

Fig6 stochastic alternative: independently select coordinates when
tan(pi*(r-.5)) < 1-FE/max_FE, replace selected values uniformly within bounds,
evaluate and greedily retain. The replacement distribution is unspecified and
uniform is an explicit choice. We then evaluate the unweighted and both weighted
chimp proposals; the first half uses lambda1 for acceptance, the second lambda2.
Random coefficients are independently sampled for the proposals. Every actual
fitness call counts, including rejected/inactive proposals, and the best evaluated
vector is archived. No candidate is evaluated after the configured budget.
Theta starts at1 per agent, increments by1 if the accepted agent does not improve,
and halves otherwise. The units, initialization and increment are unspecified in
the paper. Symbols kappa,J,beta,zeta in Fig6 are not independently defined in the
supplied method; no invented hidden mechanism is attributed to the authors.

TAR's300k FE and other methods'50k FE cannot support claims of superiority at
equal compute. A matched50k configuration is included for a separate sensitivity
experiment; it does not replace the main300k setting.

## ARGA / ARWOA

No verified definition is available. No functions named ARGA or ARWOA are shipped.
Optional GA_reference and WOA_reference are transparent added comparators:
GA uses 3-way tournament, arithmetic crossover, independent Gaussian coordinate
mutation and greedy survivor selection. WOA uses the standard encircling/random
search/spiral update. Neither is a substitute that may be labeled ARGA/ARWOA in
the thesis. Their inclusion requires the separate config to be selected before
starting an experiment. ARGA/ARWOA rows should be marked unavailable, not fabricated.

## Computational fidelity and checkpoints

Full widths/populations/budgets remain in final.json. Disk-backed float32 vectors
and chunkwise arithmetic change storage, not dimensions. Population buffers are
transactional across generations: a crash resumes the last committed generation
and may replay the partial generation. physical_evaluations.jsonl records completed
calls including replay; committed FE drives the algorithm. A crash exactly between
returning fitness and writing the ledger can leave one completed call unlogged.
Epoch checkpoints include Adam and RNG states. CUDA reproducibility across
different hardware/PyTorch versions is not guaranteed. Do not edit configuration
or code mid-experiment: fingerprints reject mixed runs.

Reference technical documentation:
- https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
- https://www.tensorflow.org/guide/keras/understanding_masking_and_padding
