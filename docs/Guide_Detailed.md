# Start with a linux server which has GPU
# Recommended python version is 3.11 or 3.12, Other requirements has been explained at "requirements.txt"

# If you use bash, you can run following command to setup project:
  ```bash
  cd DLSTM_Thesis
  python -m venv .venv
  source .venv/bin/activate
  python -m pip install -r requirements.txt
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  jupyter lab
  ```

# If CUDA is set to `False`, install the PyTorch version compatible with the server's driver/CUDA using the official instructions at https://pytorch.org/get-started/locally/. The final configuration does not automatically switch to CPU or a smaller network. Installing the packages requires an internet connection on the server; however, training does not require internet access once installation is complete.

# Notebooks: 
  | File | Task |
  | 00 | Environment review, parameter count, and data preparation |
  | 01 | DLSTM validation for the full year and the last 100 days; input length selection |
  | 02 | Validation of DLSTM and logistic regression for the selected configuration |
  | 03 | CHOA validation |
  | 04 | WCHOA validation |
  | 05 | GSOBL-CHOA validation |
  | 06 | TAR-CHOA validation |
  | 07 | Optional GA and WOA comparisons; not equivalent to ARGA/ARWOA |
  | 08 | Threshold and protocol finalization; final training and test evaluation |
  | 09 | Results table, plots, and pairwise statistical comparisons |
  | 10 | Independent data retrieval from saved notebook outputs |

# All notebooks read from `configs/final.json`; configure the data and output paths only there. Upon re-execution, completed runs are not repeated, and incomplete runs resume from the checkpoint. The notebooks contain no hardcoded final outputs.

# Continues running with "tmux":
  Using tmux for long-running processes ensures that a browser disconnection does not stop the server session. You should also save the console output:

  ```bash
  python -m thesis.cli preflight --config configs/final.json
  python -u -m thesis.cli all --config configs/final.json > final_run.log 2>&1
  ```

  Phased execution is also possible: the prepare, sequence, validate, freeze, final, and analyze actions. For example:

  ```bash
  python -u -m thesis.cli validate --models CHOA --config configs/final.json
  ```

# Avoid running two processes simultaneously for the same run; the operating system prevents this. Do not modify code, data, or settings mid-experiment; select a new `output_root` and experiment name for each change.

# Project size:
  For `hidden_widths=[4000, 2000, 2000, 30]`, 30 epochs, and a population of 200, a budget of 50,000 FEs was allocated for CHOA/WCHOA/GSOBL, while 300,000 FEs were reserved for TAR. The architecture consists of three LSTM layers and one Dense layer; however, the paper does not specify the exact type of each layer. This interpretation is inconsistent with the parameter count table provided in the paper—a discrepancy that remains unresolved. The `preflight` command displays the precise network size and the minimum disk space required for the population.

  The on-disk population and candidates are evaluated on the GPU one by one; the disk does not replace the population or the search dimensions. Upon completion of a run with `retain_populations=false`, the population buffers are deleted, though `model.pt` and the results are preserved. Ensure that base model checkpoints and final weights are stored in persistent storage. The code will return an error if disk space is insufficient; it does not automatically adjust the budget or architecture.

  Final speed cannot be estimated from a small-scale CPU run. First, observe a preflight and a single execution on the server. This package does not guarantee that a direct search of hundreds of millions of weights will complete within the timeframe claimed in the paper.

# A small test:
  `configs/smoke.json` utilizes a small network and a very low budget; it is intended solely to verify the execution path:

  ```bash
  python -m unittest discover -s tests -v
  python -u -m thesis.cli all --config configs/smoke.json > smoke.log 2>&1
  ```

  The smoke results are not valid for the fourth season. Its output folder is separate from the final one.

# Important scientific decisions:
  - All weights and biases are optimized directly; there is no hyperparameter search.
  - The 30-epoch method described in the paper is ambiguous: in this version, the base model is first trained using Adam, and subsequently, optimizers refine the weights of that same model.
  - The input window is selected using validation and the base model; all models utilize the same window and the same share-years. The mask is based on actual length, not zero values.
  - To prevent the premature use of the following year's label, a one-year gap has been incorporated:
  training up to the year 1400, with testing on data from 1402 and 1403. The decision regarding `label_gap_years` is
  methodological.
  - ARGA and ARWOA have not yet been identified; we do not have heuristic algorithms by these names in the package. The `configs/with_classical_comparators.json` file includes only `GA_reference` and `WOA_reference` as additional comparators; select this configuration before starting the full experiment.
  - Comparing TAR with 300,000 FEs against other methods with 50,000 FEs is not a matched-budget comparison. The file `configs/matched_budget.json` is available for a separate sensitivity analysis using 50,000 FEs.
  - A complete table of ambiguities, corrections, and selected formulas is provided in `docs/Method_Decisions.md`.

# Standard output for each model running:
  The files `predictions.csv`, `metrics.json`, `config.json`, `scaler.json`, `sample_ids.json`, `history.json`, and `model.pt` (or `model.joblib` for LR) are saved under `outputs/.../runs/sequence/stage/model/seed/fold`. The final training checkpoint includes the state of the Adam optimizer and the random number generator. Checkpoints are created at the boundary of each generation; if execution is interrupted, the incomplete portion of the last generation may be recomputed, and additional actual FEs are recorded in `physical_evaluations.jsonl`.

  The final analysis comprises the aggregated prediction across seeds, seed-specific and year-specific metrics, ROC/PR curves and confusion matrices, the confidence interval for the difference in Balanced Accuracy (using cluster bootstrapping), and a paired permutation test with Holm correction. The seeds are not assumed to be statistically independent samples. Economic interpretation and consideration of the limitations regarding generalizability to other years remain necessary.

# Restore from notebook results:
  After each full execution, critical data is printed in the output as a compressed capsule accompanied by a checksum. Save the notebook containing these outputs and download a standalone copy. The final cell of each notebook, when set to `RESTORE=True`, restores these capsules into a fresh folder and functions even without the `thesis` package. Deleting or truncating the outputs breaks this restoration process. Single-sample probabilities or network weights cannot be reconstructed simply by viewing or printing `F1`; the weights are not embedded within the capsule, so you must back up the `model.pt`/`checkpoint` file separately.
