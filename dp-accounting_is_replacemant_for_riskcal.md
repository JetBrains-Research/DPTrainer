### Overview

This guide explains how to migrate Python projects from Microsoft’s `riskcal` to `dp-accounting` (this repo’s `jbr.fed.accounting` package). It covers installation, API mapping, calibration equivalents, and validation tips so you can replace `riskcal` with minimal friction and identical or tighter guarantees.

Key takeaways:
- Use `PLDAccountant` as a drop‑in step-based accountant (like `riskcal.CTDAccountant`).
- Replace error‑rate calibration utilities (beta/advantage) with the provided calibration functions.
- Keep your DP-SGD loop structure; only the imports, constructor, and function names change.

### Installation and Compatibility

- Uninstall or remove direct usage of `riskcal` where possible.
- Install the dp-accounting Python package for this project (exposed as `jbr.fed.accounting`).
  - If you vend this repo: ensure `packages/dp-accounting` is installed in your environment.
- Supported metrics in `dp-accounting` Python API:
  - Step-based composition: `PLDAccountant.step(noise_multiplier=..., sample_rate=...)`
  - Queries: `get_epsilon(delta=)`, `get_delta(epsilon=)`, `get_advantage()`, `get_beta(alpha=)`
  - Calibration: `find_noise_for_epsilon(...)`, `find_noise_for_beta(...)`

### Concept and API Mapping

- riskcal’s `CTDAccountant` → dp-accounting’s `PLDAccountant`
- Step update: `accountant.step(noise_multiplier, sample_rate)` → same signature in `dp-accounting`
- Epsilon query: `get_epsilon(delta=...)` → same in `dp-accounting`
- Delta query: `get_delta(epsilon=...)` → same in `dp-accounting`
- Advantage: `get_advantage()` → same in `dp-accounting`
- Beta (FNR at fixed FPR α): `get_beta(alpha=...)` → same in `dp-accounting`
- Calibration for error rates:
  - riskcal: `find_noise_for_err(...)` / `find_noise_multiplier_for_err_rates(...)`
  - dp-accounting: `find_noise_for_beta(beta=..., alpha=..., sample_rate=..., num_steps=...)`
- Calibration for (ε, δ):
  - riskcal: typically custom or not primary API
  - dp-accounting: `find_noise_for_epsilon(epsilon=..., delta=..., sample_rate=..., num_steps=...)`

### Minimal Migration: Step-Based DP-SGD Accounting

Before (riskcal):
```python
from riskcal.accountants import CTDAccountant

accountant = CTDAccountant()
for _ in range(1000):
    accountant.step(noise_multiplier=1.1, sample_rate=0.01)

epsilon = accountant.get_epsilon(delta=1e-5)
advantage = accountant.get_advantage()
beta = accountant.get_beta(alpha=0.01)
```

After (dp-accounting):
```python
from jbr.fed.accounting import PLDAccountant

accountant = PLDAccountant()
for _ in range(1000):
    accountant.step(noise_multiplier=1.1, sample_rate=0.01)

epsilon = accountant.get_epsilon(delta=1e-5)
advantage = accountant.get_advantage()
beta = accountant.get_beta(alpha=0.01)
```

Notes:
- The control flow is identical. Swap the import and class name.
- Default discretization and internals may differ, but the interface is the same.

### Migrating Calibration Utilities

1) Error‑rate target (f-DP) — hold FPR `alpha`, target FNR `beta`:

Before (riskcal):
```python
from riskcal.calibration import find_noise_for_err

noise = find_noise_for_err(sample_rate=0.01, num_steps=1000, alpha=0.01, beta=0.9)
```

After (dp-accounting):
```python
from jbr.fed.accounting import find_noise_for_beta

noise = find_noise_for_beta(beta=0.9, alpha=0.01, sample_rate=0.01, num_steps=1000)
```

2) (ε, δ) target — find minimal noise multiplier for a given `epsilon` and `delta`:

Before (typical alternatives in riskcal or custom code):
```python
# Often done via custom search or different tools
```

After (dp-accounting):
```python
from jbr.fed.accounting import find_noise_for_epsilon

noise = find_noise_for_epsilon(
    epsilon=1.0,
    delta=1e-5,
    sample_rate=0.01,
    num_steps=1000,
)
```

Tips:
- Keep `sample_rate = batch_size / dataset_size`.
- Keep `num_steps = total optimizer steps`.
- Use the same `alpha` and `beta` targets you used with `riskcal` to reproduce behaviors.

### Heterogeneous Training Schedules

If you previously varied noise by phase (e.g., warmup vs main) by running separate loops and manually composing in `riskcal`, keep the same pattern with `PLDAccountant` by running separate accountants or by continuing the same accountant with different `noise_multiplier` per step.

Example with a single accountant and varying noise:
```python
from jbr.fed.accounting import PLDAccountant

acc = PLDAccountant()
# Warmup
for _ in range(100):
    acc.step(noise_multiplier=1.2, sample_rate=0.01)
# Main
for _ in range(900):
    acc.step(noise_multiplier=1.0, sample_rate=0.01)

epsilon = acc.get_epsilon(delta=1e-5)
```

### Adjacency and Sensitivity Considerations

- Default analyses in step-based DP-SGD use add/remove adjacency, which matches standard practice in `riskcal` step APIs.
- If you relied on replace-adjacency results in `riskcal`, be mindful that replace-adjacency effectively doubles L2 sensitivity for Gaussian mechanisms. In dp-accounting’s Python step API, stick to the default unless you have a specific reason and explicit support for replace-adjacency in your workflow.

### Practical Validation Checklist

Use this to confirm your migration preserves (or tightens) guarantees.

- Fixed configuration parity:
  - Same `sample_rate`, `num_steps`, and, if applicable, the same truncation/batching policy.
  - Same `alpha` and `beta` when comparing error‑rate metrics.
  - Same `delta` when comparing `epsilon` queries.
- Expected invariants:
  - `epsilon_dp_accounting` should be within a small tolerance of `epsilon_riskcal` (often tighter in dp-accounting due to improved discretization/FFT composition).
  - `advantage` and `beta(alpha)` should closely match or be slightly smaller (tighter) with dp-accounting.
- Programmatic A/B:
```python
def run(accountant_cls):
    acc = accountant_cls()
    for _ in range(1000):
        acc.step(noise_multiplier=1.1, sample_rate=0.01)
    return {
        "eps@1e-5": acc.get_epsilon(delta=1e-5),
        "beta@alpha=0.01": acc.get_beta(alpha=0.01),
        "adv": acc.get_advantage(),
    }

from riskcal.accountants import CTDAccountant
from jbr.fed.accounting import PLDAccountant

print("riskcal:", run(CTDAccountant))
print("dp-accounting:", run(PLDAccountant))
```
- If results differ materially, verify: `sample_rate`, `num_steps`, clipping behavior, and any dataset-size‑dependent `delta` conventions.

### Common Pitfalls and How to Avoid Them

- Mixing delta conventions: Ensure you pass the same `delta` to `get_epsilon(delta=...)` after migration.
- Sample rate vs batch size: Always use `sample_rate = batch_size / dataset_size` (float).
- Calibration domains: If your previous search relied on specific noise ranges or grid steps, dp-accounting’s calibration uses robust defaults. If needed, expose and tune search config in your application for parity.
- Truncation/batch capping: If your production system uses fixed batch sizes (truncated Poisson) and you validated riskcal against that regime, keep the same batching policy in your training loop during comparison.

### End-to-End Migration Example

Use-case: Maintain the same privacy metrics and then re‑calibrate for a new (ε, δ) target.

Before (riskcal):
```python
from riskcal.accountants import CTDAccountant
from riskcal.calibration import find_noise_for_err

# Existing training accounting
acc = CTDAccountant()
for _ in range(3750):
    acc.step(noise_multiplier=0.8, sample_rate=16 / 20000)
print("eps@1e-5:", acc.get_epsilon(delta=1e-5))
print("beta@alpha=0.01:", acc.get_beta(alpha=0.01))

# Recalibrate for target beta
nm = find_noise_for_err(sample_rate=16/20000, num_steps=3750, alpha=0.01, beta=0.9)
```

After (dp-accounting):
```python
from jbr.fed.accounting import PLDAccountant, find_noise_for_beta, find_noise_for_epsilon

# Existing training accounting
acc = PLDAccountant()
for _ in range(3750):
    acc.step(noise_multiplier=0.8, sample_rate=16 / 20000)
print("eps@1e-5:", acc.get_epsilon(delta=1e-5))
print("beta@alpha=0.01:", acc.get_beta(alpha=0.01))

# Recalibrate for target beta
nm = find_noise_for_beta(beta=0.9, alpha=0.01, sample_rate=16/20000, num_steps=3750)

# Or, recalibrate for (ε, δ)
nm_ed = find_noise_for_epsilon(epsilon=4.0, delta=5e-5, sample_rate=16/20000, num_steps=3750)
```

### Performance Notes

dp-accounting has been benchmarked to be notably faster than both Google’s `dp_accounting` and `riskcal` for common DP-SGD workloads, especially for repeated queries of `epsilon`/`delta` and calibration loops. Expect like‑for‑like or tighter results with lower compute time.

### What to Ask Your Team to Run

Because this environment cannot execute your full `scripts/` workloads, please run your existing training script twice (once with `riskcal`, once with `dp-accounting`) and share:
- Final `epsilon@delta`, `advantage`, and `beta(alpha)`
- Any calibration outputs for `find_noise_for_beta` or `find_noise_for_epsilon`

We’ll help verify parity and advise on any observed differences.

### FAQ

- Do I need to change my training loop? No. Keep the loop; only the accountant import and calibration helpers change.
- Are f‑DP metrics supported? Yes: `get_advantage()` and `get_beta(alpha=...)`, plus `find_noise_for_beta(...)`.
- Can I mix metrics from different attacks? No. Keep metrics internally consistent to the DPO/DP-SGD analysis you’re using.
- What if I used event-level APIs? dp-accounting primarily exposes step-based Python APIs for DP-SGD; if you used more granular event constructs in riskcal, replicate them via step calls or reach for the Rust/functional API in advanced use cases.

### Summary Migration Checklist

- Replace `from riskcal.accountants import CTDAccountant` with `from jbr.fed.accounting import PLDAccountant`.
- Replace `from riskcal.calibration import find_noise_for_err` with `from jbr.fed.accounting import find_noise_for_beta`.
- If you need (ε, δ) calibration, add `from jbr.fed.accounting import find_noise_for_epsilon`.
- Keep `step(noise_multiplier, sample_rate)` loops and `get_epsilon`/`get_delta`/`get_advantage`/`get_beta` queries unchanged otherwise.
- Validate parity on your workloads; adjust only if you intentionally want tighter bounds or different targets.
