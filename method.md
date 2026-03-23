# ORBITA-CTO Predictive Model v2.2 (Final)
## Modular Bayesian Simulation on the Longitudinal Proportional-Odds Scale

---

## 0. Executive Summary

We built a Bayesian simulation to predict what ORBITA-CTO — the first sham-controlled CTO PCI trial — will show. The project evolved through 4 iterations (v1 → v2.0 → v2.1 → v2.2), each guided by expert methodological review.

**Final answer (v2.2):** The predicted Day-180 daily transition OR is **1.05–1.10** depending on the assumed open-label bias. A 50-patient trial has **44–54% power** to achieve statistical significance. The clinical benefit translates to approximately **0–3 extra angina-free days** over 6 months — difficult for patients to perceive.

| Model | Estimand | Mean OR | 95% CrI | Width |
|---|---|---|---|---|
| v1 | Frequentist summary OR | 1.33 | 0.45–3.41 | 2.96 |
| v2.0 | Daily transition OR | 1.05 | 0.86–1.29 | 0.43 |
| v2.1 | + Expert round-1 fixes | 1.10 | 0.75–1.49 | 0.74 |
| **v2.2 (bias=0)** | **+ Reparametrized** | **1.10** | **0.80–1.46** | **0.66** |
| **v2.2 (bias=0.05)** | **+ Assumed moderate bias** | **1.05** | **0.76–1.39** | **0.63** |

---

## 1. The Problem We Solved

### 1.1 What is ORBITA-CTO?

ORBITA-CTO is a 50-patient, double-blind, sham-controlled pilot trial testing PCI for chronic total coronary occlusions. Its primary endpoint is a **daily ordinal angina symptom score** (0–79, recorded via smartphone, 6 months). The trial will analyze this with a **Bayesian longitudinal proportional-odds model** — the same method ORBITA-2 used.

### 1.2 What we wanted to do

Predict the trial's primary result before publication, using evidence from 5 published trials:
- **Sham-controlled, non-CTO**: ORBITA, ORBITA-2
- **Open-label, CTO**: EuroCTO, COMET-CTO, DECISION-CTO

### 1.3 What went wrong with v1

v1 used a hierarchical Bayesian meta-analysis on **frequentist cross-sectional summary OR** values. This had three fundamental problems:

1. **Wrong estimand**: ORBITA-CTO will report a daily transition OR, not a summary OR. ORBITA-2 reported both: summary OR 2.21 and daily transition OR 1.22. These are different numbers from the same data.

2. **Metric chaos**: The 5 input trials reported different things — exercise time (seconds), SAQ mean differences, CCS class proportions, near-null differences. We forced everything onto a common "log-OR" scale via ad-hoc conversions (Chinn formula, proportion→OR).

3. **Massive information loss**: ORBITA-2's own Bayesian model used ~25,000 patient-days of data and achieved CrI width 0.14. Our model used 7 summary statistics and got CrI width 2.96 — 20× wider.

---

## 2. What We Did

### 2.1 Expert Review Process

We presented the problem to a statistical expert who provided two rounds of detailed feedback:

**Round 1 (9 recommendations):**
- Define the estimand as the Day-180 daily transition OR
- Anchor μ to ORBITA-2's Bayesian PO coefficient (β=0.1991, SE=0.0288)
- Implement cutting feedback: μ locked to ORBITA-2, CTO data only updates δ_CTO
- Build uncertainty-aware crosswalk from SAQ to daily PO scale
- Use proper meta-analytic likelihood with within-trial SEs + between-trial τ²
- Posterior predictive simulation of N=50 trial

**Round 2 (5 recommendations):**
- Reparametrize: δ_open = δ_CTO + b_bias (inseparable with current data)
- Simplify crosswalk: remove θ_base, θ_time, θ_meds (no data to update them = "uncertainty theater")
- Fix μ_Δ=0 in base case (no data for Day 84→180 extrapolation)
- Use Shapley effects for variance decomposition (not marginal sums)
- Report prior→posterior information gain (shrinkage, KL)

### 2.2 Key Methodological Decisions

**Decision 1: Estimand = Day-180 daily transition OR**

ORBITA-CTO's primary analysis will produce a daily transition OR from a Bayesian longitudinal PO model. We predict this exact number.

**Decision 2: μ from ORBITA-2's own Bayesian posterior (cutting feedback)**

ORBITA-2 is the only trial with the same endpoint (daily ordinal angina diary) analyzed with the same method (Bayesian PO). Its model coefficient β=0.1991 (SE=0.0288) is our anchor.

We use a **robust MAP prior**: a mixture of the ORBITA-2 posterior (weighted) and a vague prior (fallback), with weight w ~ Beta(2,2). This protects against transport failure.

The SE is inflated 2× for transport uncertainty (different population, different follow-up duration).

**Cutting feedback** means ORBITA-2 lives entirely in the prior for μ — it does not appear in the likelihood. CTO trial data cannot move μ.

**Decision 3: δ_open = δ_CTO + b_bias (reparametrization)**

All CTO trials in our dataset are both open-label AND CTO. The bias term and CTO attenuation are always active together. The model can learn their sum (δ_open) but cannot separate them.

Instead of pretending they're separable, we reparametrize:
- δ_open = "total modification when going from sham non-CTO to open-label CTO"
- For ORBITA-CTO (sham CTO), we need μ + δ_CTO = μ + δ_open − b_bias
- Since b_bias is unknown, we show **bias scenarios** (0.00, 0.02, 0.05, 0.08, 0.10)

This is the most honest approach given the data.

**Decision 4: Crosswalk = α_0 + γ only**

CTO trials report SAQ angina frequency differences. We need to convert these to the daily PO scale. ORBITA-2 measured both:
- SAQ AF diff = 14.4 points
- Daily PO β = 0.1991
- Crosswalk slope: γ = 0.1991 / 14.4 = 0.01383

Expert round-2 advised removing the enriched crosswalk modifiers (θ_base, θ_time, θ_meds) because with a single calibration point, these are pure prior with no data to update them.

Final crosswalk: `y_mapped = α_0 + γ × ΔSAQ`

**Decision 5: DECISION-CTO included with contamination parameter**

DECISION-CTO is problematic: both arms received non-CTO PCI, SAQ scores near ceiling (>94). Rather than exclude it, we include it with a contamination dampening parameter κ ~ Beta(2,2):

`δ_effective = δ_open × (1 − κ × I_contam)`

When κ→1, DECISION-CTO's δ is completely dampened. The data inform κ.

**Decision 6: μ_Δ = 0 (no time extrapolation in base case)**

ORBITA-2 followed for 84 days; ORBITA-CTO follows for 180 days. There is zero data on how the treatment effect changes after Day 84. Rather than add a prior-only parameter, we fix μ_Δ=0 (assume no change) and note this as a limitation.

---

## 3. The Model (Final, v2.2)

### 3.1 Equation

```
log_OR_i = μ + δ_open × (1 − κ × I_contam_i) + ε_i
ε_i ~ Normal(0, τ²)
```

For ORBITA-CTO prediction (sham=1, CTO=1):
```
pred = μ + δ_open − b_bias + noise(τ)
```
where b_bias is unknown → shown as scenarios.

### 3.2 Parameters (8 total)

| Parameter | Prior | Role |
|---|---|---|
| μ | Robust MAP: w×N(0.1991, 0.058) + (1−w)×N(0, 0.5) | ORBITA-2 base effect |
| w | Beta(2, 2) | MAP weight |
| δ_open | Normal(−0.10, 0.20) | CTO attenuation + open-label bias combined |
| κ | Beta(2, 2) | DECISION-CTO contamination dampening |
| τ | HalfNormal(0.10) | Between-trial heterogeneity |
| γ | Normal(0.01383, 0.007) | Crosswalk slope |
| α_0 | Normal(0, 0.10) | Crosswalk intercept |
| σ_cross | HalfNormal(0.15) | Crosswalk residual uncertainty |

### 3.3 Data Inputs

| Trial | SAQ AF diff | SE | Contam | Crosswalk → daily PO |
|---|---|---|---|---|
| EuroCTO | 5.23 | 1.78 | 0 | α_0 + γ×5.23 |
| COMET-CTO | 13.0 | 4.56 | 0 | α_0 + γ×13.0 |
| DECISION-CTO | 0.83 | 0.76 | 1 | α_0 + γ×0.83 (dampened by κ) |

μ comes entirely from ORBITA-2 prior (cutting feedback).

### 3.4 Likelihood

For each CTO trial i:
```
y_i_mapped ~ Normal(θ_i, sqrt(v_i))
θ_i = μ + δ_open × (1 − κ × contam_i)
v_i = γ² × SE(SAQ)² + σ_cross² + τ²
```

---

## 4. What We Found

### 4.1 Primary Results

| Bias scenario | Mean OR | Median OR | 95% CrI | Pr(benefit) |
|---|---|---|---|---|
| bias=0.00 (optimistic) | 1.10 | 1.10 | 0.80–1.46 | 73% |
| bias=0.02 | 1.08 | 1.08 | 0.79–1.44 | 69% |
| **bias=0.05 (moderate)** | **1.05** | **1.05** | **0.76–1.39** | **60%** |
| bias=0.08 | 1.02 | 1.02 | 0.74–1.36 | 53% |
| bias=0.10 (pessimistic) | 1.00 | 1.00 | 0.73–1.33 | 47% |

### 4.2 Posterior Parameters

| Parameter | Mean ± SD | 95% CrI | Learned from data? |
|---|---|---|---|
| μ | 0.133 ± 0.129 | — | **No** (prior-driven, by design) |
| **δ_open** | **−0.047 ± 0.145** | — | **Yes** (48% shrinkage) |
| κ | 0.463 ± 0.223 | — | No (prior-driven) |
| **τ** | **0.065 ± 0.050** | — | **Yes** (31% shrinkage) |
| γ | 0.010 ± 0.007 | — | Marginal (5%) |
| α_0 | 0.025 ± 0.096 | — | Marginal (8%) |
| **σ_cross** | **0.083 ± 0.067** | — | **Yes** (44% shrinkage) |
| w | 0.535 ± 0.222 | — | No (prior-driven) |

Three parameters are genuinely data-informed: δ_open, τ, σ_cross. The rest are prior-driven — which is expected and honest given 3 data points.

### 4.3 Shapley Variance Decomposition

| Source | Contribution |
|---|---|
| μ (ORBITA-2 anchor) | 33% |
| δ_open (CTO+bias) | <1% |
| τ (heterogeneity) | 33% |
| Crosswalk (γ, α_0, σ) | 33% |
| **Sum check** | **100%** |

Uncertainty is evenly split between the ORBITA-2 anchor, between-trial heterogeneity, and crosswalk imprecision. δ_open contributes almost nothing because its posterior is wide (prior-dominated) and centered near 0.

### 4.4 Convergence

- 4 chains × 150,000 iterations (30K burn-in, thin=2 → 240K total samples)
- All R-hat < 1.01 (converged)
- All ESS > 237,000

### 4.5 Sensitivity (10 scenarios, bias=0.05)

| Scenario | Mean OR | 95% CrI | Pr(>1) |
|---|---|---|---|
| Primary | 1.05 | 0.76–1.39 | 60% |
| Vague δ_open | 1.07 | 0.76–1.47 | 64% |
| No attenuation | 1.08 | 0.78–1.42 | 68% |
| Strong attenuation | 0.99 | 0.74–1.26 | 44% |
| Narrow μ (1.5×) | 1.05 | 0.76–1.39 | 61% |
| Wide μ (3×) | 1.05 | 0.75–1.40 | 61% |
| Wide crosswalk | 1.04 | 0.75–1.41 | 59% |
| Wide τ | 1.05 | 0.72–1.46 | 61% |
| No DECISION-CTO | 1.06 | 0.77–1.44 | 63% |
| ORBITA-2 only | 0.98 | 0.31–1.94 | 44% |

Robust across most scenarios. The outlier is "ORBITA-2 only" — without CTO data, δ_open is pure prior and uncertainty explodes.

---

## 5. Posterior Predictive Trial Simulation

### 5.1 Setup

We simulated ORBITA-CTO as it will actually be conducted:
- N=50 patients (25 PCI + 25 sham)
- 180 days of ordinal angina diary data (5 simplified categories)
- Cutpoints calibrated from ORBITA-2's published baseline distribution
- Subject random effects (σ=1.0, approximate from diary ICC)
- 500 simulated trials per bias scenario

### 5.2 Results

| Bias | Recovered OR | Power (95% CI excl 1) | Angina-free days diff |
|---|---|---|---|
| 0.00 (optimistic) | 1.13 | **54%** | +2.9 days [−21.8, +27.6] |
| 0.05 (moderate) | 1.08 | **44%** | +1.2 days [−24.5, +26.4] |
| 0.10 (pessimistic) | 1.05 | **45%** | +0.1 days [−24.3, +23.5] |

### 5.3 Clinical Translation

In the most optimistic scenario (bias=0), PCI gives approximately **3 extra angina-free days over 6 months**. In the moderate scenario (bias=0.05), it's **1 extra day**. In both cases, the 95% interval includes substantial negative values — the trial could easily show no benefit or even slight harm.

At N=50, the trial has roughly **coin-flip odds** (44–54%) of achieving statistical significance in the benefit direction.

---

## 6. Honest Assessment

### 6.1 What the model learns from data

| Component | Source | Data-informed? |
|---|---|---|
| μ (base PCI effect) | ORBITA-2 Bayesian PO posterior | No — locked via cutting feedback (by design) |
| δ_open (CTO modifier) | 3 CTO trials via crosswalk | Partially (48% shrinkage) |
| b_bias (separable) | Nothing | No — inseparable from δ_CTO |
| Crosswalk (γ, α_0) | ORBITA-2 single calibration | Marginally |
| τ (heterogeneity) | 3 CTO trials | Partially (31%) |

### 6.2 What this means

The model is **largely prior-driven**. This is not a failure — it's the honest consequence of predicting an unprecedented trial from sparse evidence. The MCMC propagates uncertainty through all uncertain links but does not discover new information beyond what the priors and 3 trial-level summaries provide.

The value of the model is:
1. **Correct scale**: predicts the metric ORBITA-CTO will actually report
2. **Structural clarity**: makes explicit the role of placebo bias and collateral attenuation
3. **Calibrated uncertainty**: CrI width 0.63–0.66, neither artificially tight nor uselessly wide
4. **Scenario transparency**: bias scenarios show the range of plausible outcomes
5. **Clinical translation**: angina-free days make the effect size tangible

### 6.3 Limitations

1. **δ_CTO and b_bias are inseparable** — we can predict μ+δ_open but not the true sham-controlled CTO effect without assuming a bias value
2. **Crosswalk is single-point calibrated** — one trial (ORBITA-2) provides the only bridge between SAQ and daily PO scales
3. **No time extrapolation** — Day 84 to Day 180 is assumed constant (μ_Δ=0)
4. **Cutpoints are approximate** — derived from published summaries, not IPD
5. **ORBITA used a different PEP** (exercise time) — included only via crosswalk with large uncertainty

---

## 7. Files

| File | Description |
|---|---|
| `model_final.py` | v2.2 model: MCMC, multi-chain, Shapley, diagnostics, sensitivity |
| `simulate_trial.py` | N=50×180 posterior predictive trial simulation |
| `method.md` | This file |
| `results_final.json` | Model output (all posteriors, predictions, diagnostics) |
| `simulation_results.json` | Trial simulation output |

---

## 8. Relationship to v1

v1 lives at https://github.com/ihtanboga/cto_discussion with an interactive discussion page at https://ihtanboga.github.io/cto_discussion/. It presents the conceptual framework (the equation, cascade dilution, scenarios) clearly and remains useful for clinical communication.

v2 is the methodologically rigorous version that predicts the correct estimand on the correct scale. The two models predict **different metrics** and their OR values cannot be directly compared:

| | v1 | v2.2 |
|---|---|---|
| Predicts | 12-week kümülatif summary OR | Day-180 daily transition OR |
| Scale | Cross-sectional PO | Longitudinal PO |
| Typical value | ~1.3 | ~1.05 |
| CrI width | ~3.0 | ~0.65 |
| Use case | Clinical communication | Methodological prediction |
