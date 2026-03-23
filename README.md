# What If We Could Preview ORBITA-CTO?
### A Bayesian Simulation on the Longitudinal Proportional-Odds Scale

**[Live Demo](https://ihtanboga.github.io/orbita-cto-bayesian-v2/)**

---

## Background

ORBITA-CTO is an upcoming 50-patient, double-blind, sham-controlled pilot trial testing PCI for chronic total coronary occlusions (CTO). Its primary endpoint is a daily ordinal angina symptom score (0–79) recorded via smartphone over 6 months — the same endpoint ORBITA-2 used. The trial will analyze this with a Bayesian longitudinal proportional-odds (PO) model.

We built a Bayesian simulation to predict the trial's primary result before publication, using only published evidence.

---

## The Philosophy

The observed effect of any PCI trial is not one thing — it is four layers stacked on top of each other:

1. **Real treatment effect (μ)**: Does opening the artery actually help? Only measurable in a sham-controlled trial where the patient doesn't know whether they received PCI or placebo.

2. **CTO modifier (δ_open)**: CTO patients have mature collateral networks already supplying the blocked territory. They are partially compensated — the incremental symptom benefit shrinks.

3. **Placebo inflation (bias)**: In open-label trials, patients know they received PCI and report feeling better regardless. In sham-controlled trials, this term is zero.

4. **Noise (ε)**: Every trial differs in patient mix, follow-up duration, endpoints.

Two binary switches control the equation: the CTO switch activates collateral attenuation, the sham switch silences placebo inflation.

---

## The Model

### Equation

```
Effect_i = μ + δ_open × CTO_i + bias × (1 − sham_i) + ε_i
ε_i ~ Normal(0, τ²)
```

### Parameters

| Parameter | Prior | Meaning |
|---|---|---|
| **μ** | Robust MAP from ORBITA-2: w×N(0.199, 0.058) + (1−w)×N(0, 0.5) | True PCI benefit over sham (daily PO scale). Locked via cutting feedback — CTO data cannot modify it. |
| **δ_open** | Normal(−0.10, 0.20) | CTO collateral attenuation + open-label placebo inflation. These are inseparable with current data (all CTO trials are open-label), so we report their sum. |
| **bias** | Inseparable from δ_open | Cannot be estimated independently. Shown as scenarios (0.00 to 0.10). |
| **κ** | Beta(2, 2) | Contamination dampening for DECISION-CTO (both arms received non-CTO PCI). |
| **τ** | HalfNormal(0.10) | Between-trial heterogeneity. |
| **γ** | Normal(0.014, 0.007) | Crosswalk slope: SAQ angina frequency points → daily PO log-OR. |
| **α₀** | Normal(0, 0.10) | Crosswalk intercept (construct drift). |
| **σ_cross** | HalfNormal(0.15) | Crosswalk residual uncertainty. |

---

## Step-by-Step: How We Got the Prediction

### Step 1 — Anchor μ from ORBITA-2

ORBITA-2 tested PCI vs sham in non-CTO patients using the same endpoint ORBITA-CTO will use (daily ordinal angina diary). Their Bayesian PO model found:

- Daily transition OR = 1.22 (β = 0.199, SE = 0.029)
- This is reported as the posterior mean in Supplementary Table S8

This is μ — the true PCI effect over sham. We use it as our anchor with a robust MAP prior (mixture of informative + vague) and inflate the SE by 2× for transport uncertainty.

**Cutting feedback**: μ lives entirely in the prior. ORBITA-2 does not appear in the likelihood. CTO trial data cannot update μ — it can only learn δ_open and τ.

### Step 2 — Translate CTO Trial Data (Crosswalk)

Three open-label CTO trials exist. None reported daily transition ORs. They reported SAQ angina frequency mean differences:

| Trial | SAQ AF Difference | SE |
|---|---|---|
| EuroCTO | +5.23 points | 1.78 |
| COMET-CTO | +13.0 points | 4.56 |
| DECISION-CTO | +0.83 points | 0.76 |

These SAQ scores live on a different scale than the daily PO log-OR. We need a bridge.

ORBITA-2 measured **both** scales:
- SAQ AF difference = 14.4 points
- Daily PO β = 0.199

This gives the crosswalk conversion rate:

```
γ = 0.199 / 14.4 = 0.01383 daily-PO-log-OR per SAQ-AF-point
```

Applying the crosswalk:

```
EuroCTO:      5.23 × 0.014 = 0.072 (mapped daily PO log-OR)
COMET-CTO:   13.0  × 0.014 = 0.180
DECISION-CTO: 0.83 × 0.014 = 0.011
```

This conversion is imperfect — single-point calibration from one trial. The uncertainty is propagated through the model via γ (slope uncertainty) and σ_cross (residual crosswalk noise).

### Step 3 — Fit the Equation via MCMC

We run Metropolis-Hastings MCMC (150,000 iterations, 4 chains) to find the values of δ_open, τ, and crosswalk parameters that best explain all three CTO trials simultaneously, given the locked μ from ORBITA-2.

A single δ_open must explain all three trials at once:

```
EuroCTO expected:    μ + δ_open + bias = 0.199 + δ_open + bias  vs observed 0.072
COMET-CTO expected:  μ + δ_open + bias = 0.199 + δ_open + bias  vs observed 0.180
DECISION expected:   μ + δ_open×(1−κ) + bias                    vs observed 0.011
```

Result: **δ_open ≈ −0.047 ± 0.145** (posterior mean ± SD)

This is not a single number — it is a distribution. The 95% credible interval spans −0.31 to +0.25. The data partially inform δ_open (48% shrinkage from prior) but substantial uncertainty remains.

### Step 4 — Predict ORBITA-CTO

ORBITA-CTO is sham-controlled (bias off) and CTO (δ_open on):

```
Effect = μ + δ_open + ε
       = 0.199 + (−0.047) + noise
       = 0.152 + noise
```

But δ_open = δ_CTO + bias (inseparable). For a sham-controlled trial, we need δ_CTO alone:

```
δ_CTO = δ_open − bias
```

Since bias is unknown, we show scenarios:

```
bias = 0.00 → δ_CTO = −0.047 → Effect = 0.152 → OR 1.10
bias = 0.05 → δ_CTO = −0.097 → Effect = 0.102 → OR 1.05
bias = 0.10 → δ_CTO = −0.147 → Effect = 0.052 → OR 1.00
```

Each prediction uses the full posterior (60,000 draws), not point estimates. The reported OR and CrI reflect the entire distribution.

### Step 5 — Simulate the Actual Trial

We generated 500 virtual ORBITA-CTO trials:
- 25 PCI + 25 sham patients
- 180 days of ordinal angina diary data (5 simplified categories)
- Cutpoints calibrated from ORBITA-2's published baseline distribution
- Subject random effects (σ = 1.0)

Each simulated trial was re-analyzed with a proportional odds model to recover the treatment OR and test whether the 95% CI excludes 1.0.

---

## Results

### Primary Prediction (Day-180 daily transition OR)

| Open-label bias assumed | Mean OR | 95% CrI | Pr(benefit) |
|---|---|---|---|
| 0.00 (optimistic) | **1.10** | 0.80 – 1.47 | 73% |
| 0.02 | 1.08 | 0.79 – 1.44 | 69% |
| **0.05 (most plausible)** | **1.05** | **0.76 – 1.39** | **60%** |
| 0.08 | 1.02 | 0.74 – 1.36 | 53% |
| 0.10 (pessimistic) | 1.00 | 0.73 – 1.33 | 47% |

Note: "bias" here refers to the assumed open-label placebo inflation **in the source CTO trials** — not in ORBITA-CTO itself (which is sham-controlled and has no bias). Higher assumed source bias → more of δ_open was placebo → true CTO attenuation is larger → prediction is worse.

### Trial Simulation (N=50, 180 days)

| Bias | Recovered OR | Power | Angina-Free Days Diff | 95% Range |
|---|---|---|---|---|
| 0.00 | 1.13 | 54% | +2.9 days / 6 months | −22 to +28 |
| **0.05** | **1.08** | **44%** | **+1.2 days / 6 months** | **−25 to +26** |
| 0.10 | 1.05 | 45% | +0.1 days / 6 months | −24 to +24 |

At N=50, the trial has approximately coin-flip odds (~44%) of achieving statistical significance.

Clinical translation: about 1 extra angina-free day over 6 months — difficult for patients to perceive.

### What the Model Learns from Data

| Parameter | Posterior Mean ± SD | Shrinkage | Data-informed? |
|---|---|---|---|
| μ | 0.133 ± 0.129 | 0% | No (by design — cutting feedback) |
| **δ_open** | **−0.047 ± 0.145** | **48%** | **Yes** |
| κ | 0.463 ± 0.223 | 0% | No (prior-driven) |
| **τ** | **0.065 ± 0.050** | **31%** | **Yes** |
| γ | 0.010 ± 0.007 | 5% | Marginal |
| α₀ | 0.025 ± 0.096 | 8% | Marginal |
| **σ_cross** | **0.083 ± 0.067** | **44%** | **Yes** |
| w_map | 0.535 ± 0.222 | 2% | No (prior-driven) |

Three parameters are genuinely data-informed (δ_open, τ, σ_cross). The rest are prior-driven — expected and honest given 3 data points.

### Shapley Variance Decomposition

| Source | Contribution to prediction uncertainty |
|---|---|
| μ (ORBITA-2 anchor) | 33% |
| δ_open (CTO + bias combined) | <1% |
| τ (between-trial heterogeneity) | 33% |
| Crosswalk (γ, α₀, σ_cross) | 33% |

Uncertainty is evenly split. δ_open contributes almost nothing because its posterior is wide and centered near zero.

### Convergence

- 4 chains × 150,000 iterations (30K burn-in, thin=2 → 240K total samples)
- All R-hat < 1.01
- All ESS > 237,000

### Sensitivity (10 scenarios, bias = 0.05)

| Scenario | Mean OR | 95% CrI | Pr(benefit) |
|---|---|---|---|
| **Primary (base case)** | **1.05** | **0.76 – 1.39** | **60%** |
| Vague δ_open prior | 1.07 | 0.76 – 1.47 | 64% |
| No CTO attenuation | 1.08 | 0.78 – 1.42 | 68% |
| Strong CTO attenuation | 0.99 | 0.74 – 1.26 | 44% |
| Wider crosswalk | 1.04 | 0.75 – 1.41 | 59% |
| Wider τ | 1.05 | 0.72 – 1.46 | 61% |
| Without DECISION-CTO | 1.06 | 0.77 – 1.44 | 63% |
| ORBITA-2 only (no CTO data) | 0.98 | 0.31 – 1.94 | 44% |

Robust: mean OR ranges 0.98–1.08 across all scenarios.

---

## Evidence Base

| Trial | Design | Sham | What they measured | What we used | How we got it |
|---|---|---|---|---|---|
| **ORBITA-2** | Non-CTO | Yes | Daily ordinal angina diary | β = 0.199, SE = 0.029 | **Direct** — same scale as target |
| **EuroCTO** | CTO | No | SAQ AF mean difference: +5.23 pts | 0.072 daily PO log-OR | Crosswalk: 5.23 × γ |
| **COMET-CTO** | CTO | No | SAQ AF mean difference: +13.0 pts | 0.180 daily PO log-OR | Crosswalk: 13.0 × γ |
| **DECISION-CTO** | CTO* | No | SAQ AF mean difference: +0.83 pts | 0.011 daily PO log-OR | Crosswalk: 0.83 × γ |
| **ORBITA-CTO** | **CTO** | **Yes** | **Daily ordinal angina diary** | **??? → OR 1.05–1.10** | **This model** |

*DECISION-CTO: both arms received non-CTO PCI (dampened by κ).

### The Equation Applied to Each Trial

```
ORBITA-2 (sham=1, CTO=0):
  Effect = μ + δ_open×0 + bias×(1−1) + ε = μ + ε = 0.199 + ε → OR ≈ 1.22
  Only μ is active. Cleanest measurement.

EuroCTO (sham=0, CTO=1):
  Effect = μ + δ_open + bias + ε = 0.199 + (−0.047) + bias + ε
  Observed (crosswalk): 0.072. Both switches on.

COMET-CTO (sham=0, CTO=1):
  Effect = μ + δ_open + bias + ε = 0.199 + (−0.047) + bias + ε
  Observed (crosswalk): 0.180. Same structure, larger noise (N=100, single center).

DECISION-CTO (sham=0, CTO=1, contaminated):
  Effect = μ + δ_open×(1−κ) + bias + ε = 0.199 + (−0.047)×0.54 + bias + ε
  Observed (crosswalk): 0.011. δ_open dampened because both arms got PCI.

ORBITA-CTO (sham=1, CTO=1):
  Effect = μ + δ_open + ε = 0.199 + (−0.047) + ε = 0.152 + ε → OR 1.05–1.10
  Sham kills bias. CTO activates δ_open. Prediction target.
```

Same μ, same δ_open everywhere. What changes: which switches are on.

---

## Key Design Decisions

### Why δ_open instead of separate δ_CTO + bias?

All CTO trials in our dataset are open-label. δ_CTO (collateral attenuation) and bias (placebo inflation) are always active together. The model can learn their sum but cannot separate them. Reporting them separately would be misleading — the individual values would be prior-driven artifacts.

We reparametrize: δ_open = δ_CTO + bias, and show bias scenarios for the sham-controlled prediction.

### Why cutting feedback?

ORBITA-2 provides μ with high precision (SE = 0.029). CTO trials are noisy (open-label, small, different endpoints). Without cutting feedback, noisy CTO data would pull μ off the ORBITA-2 anchor — corrupting the most reliable piece of information.

With cutting feedback: μ is locked. CTO data can only inform δ_open and τ. Information flows one way.

### Why crosswalk instead of direct comparison?

CTO trials measured SAQ angina frequency (0–100, single timepoint). ORBITA-CTO measures daily ordinal angina diary (0–79, daily for 6 months). Different instruments, different scales.

ORBITA-2 measured both → conversion rate: 1 SAQ point = 0.014 daily PO log-OR. This crosswalk is single-point calibrated, imperfect. The conversion uncertainty (γ ± SD, σ_cross) is propagated through the model.

### Why robust MAP for μ?

Normal prior would mean blind trust in ORBITA-2. Robust MAP is a mixture:
- ~70%: μ ≈ ORBITA-2's value (0.199 ± 0.058)
- ~30%: μ could be anything (vague fallback)

If CTO data conflicts with ORBITA-2, the model shifts toward the vague component — a safety mechanism.

---

## Limitations

1. **δ_CTO and bias are inseparable** — we predict μ + δ_open but the true sham-controlled CTO effect requires knowing bias separately
2. **Crosswalk is single-point calibrated** — one trial (ORBITA-2) bridges SAQ and daily PO scales
3. **No time extrapolation** — ORBITA-2 followed 84 days, ORBITA-CTO follows 180 days; we assume μ is constant (μ_Δ = 0)
4. **Cutpoints are approximate** — derived from published summaries, not individual patient data
5. **Model is largely prior-driven** — μ entirely from prior (by design), δ_open partially learned (48% shrinkage), other parameters mostly prior-driven. The MCMC propagates uncertainty but does not discover new information beyond what the priors and 3 trial-level summaries provide.

---

## Files

| File | Description |
|---|---|
| `model_final.py` | Complete model: MCMC, multi-chain convergence (R-hat, ESS), Shapley variance decomposition, prior-posterior diagnostics, 10 sensitivity scenarios |
| `simulate_trial.py` | Posterior predictive trial simulation: N=50 × 180-day ordinal angina diary, power analysis, angina-free days clinical translation |
| `index.html` | Interactive discussion page (deployed to GitHub Pages) |
| `results_final.json` | Full model output: posteriors, predictions, diagnostics |
| `simulation_results.json` | Trial simulation results |

---

## References

- ORBITA-2 (NEJM 2023): β = 0.1991, SE = 0.0288 (Supplementary Table S8)
- ORBITA-CTO design paper (Front Cardiovasc Med 2023)
- EuroCTO (EHJ 2018, EuroIntervention 2023): SAQ AF diff +5.23
- COMET-CTO (Int Heart J 2021, Front CV Med 2023): SAQ AF diff +13.0
- DECISION-CTO (Circulation 2019): SAQ AF diff +0.83

---

## Bottom Line

When sham control removes the open-label component, the predicted CTO PCI effect on symptoms is modest — daily transition OR 1.05–1.10. At N=50, there are approximately coin-flip odds (~44%) of statistical significance, translating to roughly 1 extra angina-free day over 6 months. The physiology will improve; whether patients perceive it remains the open question ORBITA-CTO was designed to answer.

> Simulation, not prediction. Assumption-driven, not identified. Published evidence only — no insider data.
