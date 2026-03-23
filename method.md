# ORBITA-CTO Predictive Model v2.1: Longitudinal Proportional Odds
## Expert-Reviewed, Modular Bayesian Synthesis

---

## 1. Estimand

**Target**: Day-180 daily transition OR from a Bayesian longitudinal proportional-odds model of the ordinal angina diary (same model family ORBITA-CTO will use).

All parameters must live on the **cumulative logit / daily transition log-OR scale**.

---

## 2. Model Architecture

### 2.1 Core Equation

```
log_OR_i = μ(t_i) + δ_CTO × CTO_i × (1 − κ × I_contam_i) + b_bias × (1 − sham_i) + ε_i
ε_i ~ Normal(0, τ²)
```

### 2.2 New Components (Expert Additions)

**μ(t) — Time-varying base effect:**
```
μ(t) = μ_84 + μ_Δ × (t − 84) / 84
μ_Δ ~ Normal(0, 0.07)  # weak prior for 84→180 day extrapolation
```
- μ_84 = ORBITA-2 beta (anchored at Day 84)
- μ_Δ allows effect to grow/shrink beyond Day 84 with uncertainty

**κ — Contamination parameter (for DECISION-CTO):**
```
κ ~ Beta(2, 2)  # 0–1, sönümler δ_CTO when both arms got non-CTO PCI
```

### 2.3 Modular Structure (Cutting Feedback)

**Module 1: μ anchor — Robust MAP from ORBITA-2**

Instead of fixed prior, use robust MAP (mixture):
```
μ_84 ~ w × Normal(0.1991, 0.0288 × inflate) + (1−w) × Normal(0, 0.5)
w ~ Beta(2, 2)  # data-conflict weight
inflate = 2.0   # transport uncertainty (expert: 2–3×)
```
This protects against transport failure: if CTO data strongly conflict with ORBITA-2, w drops and μ moves toward vague prior.

**Module 2: CTO trials → δ_CTO, b_bias, κ**

Only CTO trial data updates these. μ locked via cutting feedback.

**Module 3: Prediction + Simulation**

Posterior predictive simulation of N=50 × 180-day trial.

---

## 3. Enriched Crosswalk Model (Expert Section 3, 11, 12, 13)

### 3.1 Full Crosswalk Equation

```
logOR_PO_i = α_0 + γ × ΔSAQ_i + θ_base × f(BaselineSAQ_i) + θ_time × g(time_i) + θ_meds × I(meds_off_i) + u_i
u_i ~ Normal(0, σ²_cross)
```

**Parameters:**
- α_0 ~ Normal(0, 0.10) — intercept (construct drift between SAQ AF and daily diary)
- γ ~ Normal(0.01383, 0.007) — slope (wider than v2.0's 0.005)
- θ_base ~ Normal(0, 0.005) — baseline SAQ modifier (tavan etkisi)
  - f(BaselineSAQ) = (BaselineSAQ − 60) / 40, centered at ORBITA-2 median
- θ_time ~ Normal(0, 0.03) — temporal mismatch adjustment
  - g(time) = (measurement_weeks − 12) / 12, centered at ORBITA-2 follow-up
- θ_meds ~ Normal(0, 0.05) — meds-off vs meds-on effect on crosswalk
  - ORBITA-2, ORBITA-CTO: meds_off=1; CTO open-label trials: meds_off=0
- σ_cross ~ HalfNormal(0.15) — residual crosswalk uncertainty (expert: 0.12–0.18)

### 3.2 Crosswalk Calibration Data

From ORBITA-2 (the single calibration point):
- ΔSAQ_AF = 14.4 points → daily PO β = 0.1991
- BaselineSAQ_AF median ≈ 60
- Measurement time = 12 weeks
- Meds_off = 1

### 3.3 Trial-Specific Crosswalk Inputs

| Trial | ΔSAQ_AF | SE | BaselineSAQ | Time(weeks) | Meds_off | CTO | Sham | Contam |
|---|---|---|---|---|---|---|---|---|
| EuroCTO | 5.23 | 1.78 | ~80 | 52 | 0 | 1 | 0 | 0 |
| COMET-CTO | 13.0 | 4.56 | ~73 | 39 | 0 | 1 | 0 | 0 |
| DECISION-CTO | 0.83 | 0.76 | ~94 | 156 | 0 | 1 | 0 | 1 |

---

## 4. Prior Specifications (Expert-Revised)

| Parameter | Prior | Rationale |
|---|---|---|
| μ_84 | Robust MAP: w×N(0.1991, 0.058) + (1−w)×N(0, 0.5) | Expert: inflate SE 2–3×; robust MAP for transport |
| w (MAP weight) | Beta(2, 2) | Allows data-conflict detection |
| μ_Δ | Normal(0, 0.07) | Day 84→180 extrapolation uncertainty |
| δ_CTO | Normal(0, 0.15) | Expert: center at 0 (not -0.05), wider |
| b_bias | HalfStudentT(ν=3, σ=0.12) | Expert: 0.12–0.15 scale |
| κ (contamination) | Beta(2, 2) | DECISION-CTO sönüm |
| τ | HalfNormal(0.10) | Expert: relax from 0.05 |
| α_0 (crosswalk intercept) | Normal(0, 0.10) | Construct drift |
| γ (crosswalk slope) | Normal(0.01383, 0.007) | Expert: wider |
| θ_base | Normal(0, 0.005) | Baseline SAQ modifier |
| θ_time | Normal(0, 0.03) | Temporal mismatch |
| θ_meds | Normal(0, 0.05) | Medication protocol |
| σ_cross | HalfNormal(0.15) | Expert: 0.12–0.18 |

---

## 5. Clinical Translation (Expert Section 14)

Daily transition OR 1.05 means very little to a clinician. We need to translate to:

1. **Angina-free days difference** over 180 days
2. **Probability of being in better state** at Day 180
3. **SAP-compatible primary OR** at Day 180

### Simulation Procedure:
```
For each posterior draw (μ, δ_CTO, b_bias, τ, crosswalk params):
    For each replicate (200-500):
        Simulate 50 patients × 180 days ordinal diary data
        Under PCI: η = α_k − [μ(180) + δ_CTO + b_i]
        Under sham: η = α_k − [b_i]
        Count angina-free days (category 0) per patient
        Fit Bayesian PO model → extract Day-180 OR
    Record:
        - Mean angina-free days PCI vs sham
        - Day-180 transition OR
        - Whether 95% CrI excludes 1
```

### Cutpoint Estimation (without IPD):
- Use ORBITA-2 published baseline category distribution
- Approximate α_k from published angina-free day proportions
- σ_subject from ICC or published variance decomposition

---

## 6. Sensitivity Analyses (Expert-Guided, 10 Scenarios)

1. **Primary (base case)** — all expert fixes
2. **ORBITA-2 only** — vague δ_CTO, no CTO data
3. **Include DECISION-CTO** — with κ contamination
4. **Exclude DECISION-CTO** — primary without it
5. **Wider crosswalk** — σ_cross = 0.25, γ SD = 0.01
6. **Narrower μ transport** — SE inflate = 1.5 (less transport uncertainty)
7. **Wider μ transport** — SE inflate = 3.0
8. **Strong CTO attenuation** — δ_CTO ~ Normal(-0.10, 0.08)
9. **No CTO attenuation** — δ_CTO ~ Normal(0, 0.10)
10. **Flat priors (likelihood-only)** — vague everything, show identifiability

### Required Diagnostics:
- **Contribution-to-variance decomposition**: what fraction of posterior variance comes from μ vs crosswalk vs δ_CTO/bias vs τ?
- **Prior-posterior overlap**: how much does data update each parameter?
- **Crosswalk calibration curve** with uncertainty bands
- **Leave-one-CTO-trial-out** sensitivity

---

## 7. Reporting (Expert Section 15, 17)

### Primary Report:
- v2.1 (Day-180 daily transition OR) as **primary**
- v1 (12-week summary OR) as **alternative/sensitivity**
- Side by side, never mixed

### Honest Framing (Expert Section 17):
> "Our results are modular Bayesian inferences anchored to a strong ORBITA-2 MAP prior, modulated for the CTO context with limited RCT data, and bridged via an uncertainty-aware SAQ-to-diary crosswalk. Parameter separability (δ_CTO vs bias) is limited; we rely on sensitivity analyses to bound epistemic uncertainty."

### Clinical Translation:
- Report angina-free days difference (from simulation)
- Report Day-180 OR with CrI
- Report probability of CrI excluding 1 under SAP analysis (power)

---

## 8. Implementation

- **Engine**: NumPy/SciPy MCMC (MH) — PyMC if available
- **Iterations**: 150,000 (30,000 burn-in, thin=2 → 60,000 samples)
- **Convergence**: Multiple chains, R-hat, ESS
- **Output**: JSON results, HTML report, comparison table v1 vs v2.1
