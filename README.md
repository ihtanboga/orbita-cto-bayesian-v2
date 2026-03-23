# ORBITA-CTO Bayesian Simulation v2: Longitudinal Proportional Odds

A modular Bayesian meta-analysis that predicts what ORBITA-CTO will show — on the **exact scale** (daily transition OR) the trial will report.

## What changed from v1?

v1 predicted a frequentist summary OR (1.33, CrI 0.45–3.41). Expert review identified a fundamental mismatch: ORBITA-CTO will report a **Bayesian longitudinal proportional-odds daily transition OR**, not a cross-sectional summary OR.

v2 fixes this by:

| Fix | What | Why |
|-----|------|-----|
| **Correct estimand** | Day-180 daily transition OR | Matches ORBITA-CTO's actual analysis |
| **μ anchor** | ORBITA-2 Bayesian PO β=0.1991 (SE=0.0288) | Same scale, 25K patient-days of data |
| **Cutting feedback** | μ locked to ORBITA-2, CTO data only updates δ | Prevents noisy data from corrupting the anchor |
| **Reparametrization** | δ_open = δ_CTO + b_bias | Honest: these are inseparable with current data |
| **Uncertainty-aware crosswalk** | SAQ→daily PO with residual variance | Replaces deterministic Chinn conversion |
| **Bias scenarios** | bias=0.00 to 0.10 | Shows range of true sham-controlled effect |

## Results (v2.2 Final)

| Assumed bias | Mean OR | 95% CrI | Pr(benefit) |
|---|---|---|---|
| 0.00 (optimistic) | **1.10** | 0.80 – 1.46 | 73% |
| 0.05 (moderate) | **1.05** | 0.76 – 1.39 | 60% |
| 0.10 (pessimistic) | **1.00** | 0.73 – 1.33 | 47% |

**Interpretation**: The true sham-controlled daily transition OR likely falls between 1.00 and 1.10. Small benefit, wide uncertainty.

### What the model learns from data vs prior

| Parameter | Shrinkage | Data-informed? |
|---|---|---|
| μ (ORBITA-2 anchor) | 0% | Prior-driven (by design: cutting feedback) |
| **δ_open (CTO+bias)** | **48%** | **Yes** |
| **τ (heterogeneity)** | **31%** | **Yes** |
| **σ_cross (crosswalk)** | **44%** | **Yes** |
| κ, w_map, γ, α₀ | <8% | Prior-driven |

### Convergence

- 4 chains × 150K iterations (30K burn-in, thin=2 → 240K samples)
- All R-hat < 1.01
- All ESS > 237K

## Model equation

```
log_OR_i = μ + δ_open × (1 − κ × I_contam) + ε_i
ε_i ~ Normal(0, τ²)
```

Where δ_open = δ_CTO + b_bias (inseparable). For ORBITA-CTO (sham=1), predicted effect = μ + δ_open − b_bias. Since b_bias is unknown, we show scenarios.

## Files

| File | Description |
|------|-------------|
| `model_final.py` | Complete model: MCMC, multi-chain, Shapley, diagnostics |
| `method.md` | Full methodology with expert feedback incorporated |
| `results_final.json` | All results, sensitivity, diagnostics |

## Key references

- ORBITA-2 (NEJM 2023): β=0.1991, SE=0.0288 (Supp Table S8)
- EuroCTO (EHJ 2018): SAQ AF diff +5.23
- COMET-CTO (Int Heart J 2021): SAQ AF diff +13.0
- DECISION-CTO (Circ 2019): SAQ AF diff +0.83

## Honest framing

> This model is largely prior-driven. μ comes entirely from ORBITA-2 (by design). δ_open is partially learned from 3 CTO trials. δ_CTO and b_bias cannot be separated. The MCMC propagates uncertainty but does not discover new information beyond what the priors and 3 trial-level summaries provide. This is stated, not hidden.
