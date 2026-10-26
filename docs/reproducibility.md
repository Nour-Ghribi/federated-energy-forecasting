# Reproducibility

The pinned reference stack targets Python 3.11. Direct requirements are pinned;
the validation environment is recorded in `requirements-lock-windows.txt` after
installation. This platform-specific full snapshot is evidence, not a Linux
requirements file. Linux/Ray dependencies have their own installation extra.

Runs record package versions, Python, platform, CUDA availability, GPU identity,
git commit if available, source SHA256 and the complete resolved configuration.
The comparison manifest is saved before model fitting. Partition seed 2026 is
independent of training seeds: partition assignments stay fixed across 42/43/44.

Seeds control Python, NumPy, Torch initialization and DataLoader shuffle.
Deterministic Torch algorithms are requested, CuDNN autotuning is disabled, and
CPU thread count is two. Set CUBLAS_WORKSPACE_CONFIG before CUDA initialization.
Bitwise agreement across different hardware, drivers, BLAS implementations,
Torch versions or XGBoost CPU/GPU kernels is not promised. CPU equivalence tests
run on the same machine and verify identical FedAvg/FedProx-zero parameters.

For CUDA, choose an official PyTorch 2.8.0 wheel compatible with the installed
driver from https://pytorch.org/get-started/previous-versions/ and record its full
version. The GPU profile never installs a driver or changes compute settings.

Commands default to smoke, whose results must not support scientific conclusions.
The full campaign is intentionally opt-in and can take hours. No sample limit,
round reduction or seed omission is applied implicitly. `--seed` is an explicit
restriction and appears in resolved configuration. Inference time and training
time are separate; tree-search training includes all candidates. SHAP is an
additional analysis cost and should not be confused with model inference.

Reruns create new directories. Summary tables deduplicate repeated runs of an
identical protocol and seed rather than counting them as independent replicates.
Standard deviation is sample std (ddof=1); a single seed has undefined std.
Hourly errors are serially correlated: three seeds describe optimization
variability, not confidence over independent draws of the electricity system.

Raw CSVs remain untracked. The source's clock convention cannot be established
from naive timestamps alone. The baseline uses civil-hour intervals and labels
that limitation. To claim exact elapsed-hour forecasting across DST, add a
provenance-backed timezone/hour-ending adapter and a separate protocol.

The Adam state persists across centralized epochs, and across local epochs of
one federated round; federated clients reset it when a new global model is
received. The global reference for FedProx remains fixed across those local
epochs. Centralized and federated best checkpoints are selected using the same
held-out validation metric in MW (RMSE for MSE training, MAE for MAE training).
Even with local regional scales, model selection is performed in physical units.
