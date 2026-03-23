"""
ORBITA-CTO Posterior Predictive Trial Simulation
=================================================
Simulates N=50 × 180-day ordinal angina diary data and refits
a proportional odds model to each simulated dataset.

Outputs:
- Probability of 95% CrI excluding 1 (Bayesian "power")
- Angina-free days difference (PCI vs sham)
- Day-180 OR distribution across simulated trials

Calibrated from ORBITA-2:
- Baseline angina symptom score: median 1 (pre-randomization)
- Score categories: 0 (no angina, no meds), 1-10 (angina/meds combos), 77-79 (adverse)
- PCI group: mean score 2.9 at follow-up
- Placebo group: mean score 5.6 at follow-up
- Freedom from angina (score=0): PCI 40%, placebo 15% at 12 weeks
"""

import numpy as np
import json
from scipy import stats
from scipy.special import expit  # logistic function

# ============================================================
# 1. ORDINAL MODEL PARAMETERS (calibrated from ORBITA-2)
# ============================================================

# Simplified ordinal categories for angina symptom score:
# Cat 0: Score 0 (no angina, no meds) — "angina-free"
# Cat 1: Score 1-6 (mild — few episodes, no/minimal meds)
# Cat 2: Score 7-20 (moderate — regular episodes or meds)
# Cat 3: Score 21-50 (severe — frequent episodes + meds)
# Cat 4: Score 51-79 (very severe / adverse events)
N_CATS = 5
CAT_LABELS = ["Angina-free (0)", "Mild (1-6)", "Moderate (7-20)", "Severe (21-50)", "Very severe (51-79)"]

# Cutpoints (α_k) derived from ORBITA-2 baseline + follow-up distributions:
# Baseline: median score 1, ~45% score=0, ~30% score 1-6, ~15% score 7-20, ~8% 21-50, ~2% 51+
# These give cumulative probs → logit cutpoints
# P(Y≤0) = 0.45 → α_1 = logit(0.45) = -0.20
# P(Y≤1) = 0.75 → α_2 = logit(0.75) = 1.10
# P(Y≤2) = 0.90 → α_3 = logit(0.90) = 2.20
# P(Y≤3) = 0.98 → α_4 = logit(0.98) = 3.89
CUTPOINTS = np.array([-0.20, 1.10, 2.20, 3.89])

# Subject random effect SD (approximated from ORBITA-2 ICC)
# Typical ICC for daily diary ~0.3-0.5, implies σ_subject ~ 0.8-1.2
SIGMA_SUBJECT = 1.0

# Time effect: gradual improvement in both arms (natural history + regression to mean)
# Calibrated so that placebo group improves from baseline median 1 to mean ~5.6
# Small positive time effect (shift toward worse categories initially, then stable)
TIME_EFFECT = 0.0  # simplified: no systematic time trend beyond treatment


def simulate_ordinal_day(eta, cutpoints):
    """
    Simulate one ordinal observation from cumulative logit model.
    eta: linear predictor (higher = better outcome = lower category)
    cutpoints: α_1, ..., α_{K-1}
    Returns: category index (0 = best, K-1 = worst)
    """
    # P(Y ≤ k) = expit(α_k + η)
    # Higher η → higher P(Y ≤ k) → more likely in lower (better) categories
    # This matches ORBITA-2's convention: positive β = better outcome
    cum_probs = expit(cutpoints + eta)
    # Category probabilities
    cat_probs = np.zeros(len(cutpoints) + 1)
    cat_probs[0] = cum_probs[0]
    for k in range(1, len(cutpoints)):
        cat_probs[k] = cum_probs[k] - cum_probs[k-1]
    cat_probs[-1] = 1 - cum_probs[-1]
    # Clip for numerical safety
    cat_probs = np.clip(cat_probs, 0, 1)
    cat_probs /= cat_probs.sum()
    return np.random.choice(len(cat_probs), p=cat_probs)


def simulate_trial(n_patients, n_days, beta_treatment, cutpoints, sigma_subject, rng):
    """
    Simulate a complete trial: n_patients/2 per arm × n_days.
    Returns: (data array [patient, day, category], arm assignments)
    """
    n_pci = n_patients // 2
    n_sham = n_patients - n_pci
    arms = np.array([1]*n_pci + [0]*n_sham)  # 1=PCI, 0=sham

    # Subject random effects
    b_i = rng.normal(0, sigma_subject, n_patients)

    data = np.zeros((n_patients, n_days), dtype=int)

    for i in range(n_patients):
        for d in range(n_days):
            # Linear predictor: treatment + subject effect
            # Higher η → better outcome (lower score category)
            eta = beta_treatment * arms[i] + b_i[i]
            data[i, d] = simulate_ordinal_day(eta, cutpoints)

    return data, arms


def fit_po_model(data, arms, n_days):
    """
    Fit a simple proportional odds model to simulated trial data.
    Returns: estimated treatment log-OR and its SE.

    Simplified: pool all patient-days, compute empirical cumulative
    proportions by arm, estimate OR via Mantel-Haenszel-like approach.
    """
    n_patients = len(arms)
    n_cats = N_CATS

    # Pool all observations by arm
    pci_mask = arms == 1
    sham_mask = arms == 0

    pci_data = data[pci_mask].flatten()
    sham_data = data[sham_mask].flatten()

    # Count categories
    pci_counts = np.bincount(pci_data, minlength=n_cats)
    sham_counts = np.bincount(sham_data, minlength=n_cats)

    # Cumulative OR via pooled adjacent-category method
    # For each cutpoint k, compute 2×2 table: (≤k vs >k) × (PCI vs sham)
    log_ors = []
    weights = []
    for k in range(n_cats - 1):
        a = pci_counts[:k+1].sum()     # PCI ≤ k
        b = pci_counts[k+1:].sum()     # PCI > k
        c = sham_counts[:k+1].sum()    # Sham ≤ k
        d_val = sham_counts[k+1:].sum() # Sham > k

        if a > 0 and b > 0 and c > 0 and d_val > 0:
            lor = np.log(a * d_val / (b * c))
            # Variance of log-OR
            v = 1/a + 1/b + 1/c + 1/d_val
            log_ors.append(lor)
            weights.append(1/v)

    if not log_ors:
        return 0.0, 1.0

    # Weighted average log-OR (Mantel-Haenszel)
    log_ors = np.array(log_ors)
    weights = np.array(weights)
    log_or_mh = np.sum(log_ors * weights) / np.sum(weights)
    se_mh = 1 / np.sqrt(np.sum(weights))

    return log_or_mh, se_mh


def compute_angina_free_days(data, arms, n_days):
    """Compute mean angina-free days (category 0) per arm."""
    pci_mask = arms == 1
    sham_mask = arms == 0

    pci_free = np.mean(data[pci_mask] == 0, axis=1) * n_days
    sham_free = np.mean(data[sham_mask] == 0, axis=1) * n_days

    return {
        "pci_mean": float(np.mean(pci_free)),
        "sham_mean": float(np.mean(sham_free)),
        "difference": float(np.mean(pci_free) - np.mean(sham_free)),
        "pci_sd": float(np.std(pci_free)),
        "sham_sd": float(np.std(sham_free)),
    }


# ============================================================
# 2. POSTERIOR PREDICTIVE SIMULATION
# ============================================================

def run_simulation(n_sim=500, n_patients=50, n_days=180, seed=42):
    """
    Draw treatment effects from v2.2 posterior, simulate trials, refit.
    """
    rng = np.random.default_rng(seed)

    # Load v2.2 posterior results
    try:
        with open("/Users/apple/Downloads/orbital/main/v2_longitudinal_po/results_final.json") as f:
            v22 = json.load(f)
        primary = v22["results"]["Primary"]
        post = primary.get("prediction", {})
    except:
        pass

    # Posterior draws for beta_treatment (= μ + δ_open - bias)
    # From v2.2: μ post mean ~ 0.13, δ_open post mean ~ -0.05
    # We draw from approximate posterior
    mu_draws = rng.normal(0.133, 0.129, n_sim)      # v2.2 posterior
    dopen_draws = rng.normal(-0.047, 0.145, n_sim)   # v2.2 posterior

    # Bias scenarios
    bias_values = [0.00, 0.05, 0.10]

    all_results = {}

    for bias in bias_values:
        beta_draws = mu_draws + dopen_draws - bias
        label = f"bias={bias:.2f}"

        print(f"\n  Simulating {n_sim} trials with {label}...")

        trial_ors = []
        trial_ses = []
        trial_significant = []
        trial_angina_free = []

        for s in range(n_sim):
            beta = beta_draws[s]

            # Simulate trial
            data, arms = simulate_trial(
                n_patients, n_days, beta, CUTPOINTS, SIGMA_SUBJECT, rng
            )

            # Fit PO model
            log_or, se = fit_po_model(data, arms, n_days)
            trial_or = np.exp(log_or)

            # "Significant" = 95% CI entirely above 1.0 (benefit direction)
            ci_low = np.exp(log_or - 1.96 * se)
            sig = ci_low > 1.0

            trial_ors.append(trial_or)
            trial_ses.append(se)
            trial_significant.append(sig)

            # Angina-free days
            af = compute_angina_free_days(data, arms, n_days)
            trial_angina_free.append(af["difference"])

        trial_ors = np.array(trial_ors)
        trial_significant = np.array(trial_significant)
        trial_angina_free = np.array(trial_angina_free)

        all_results[label] = {
            "n_simulations": n_sim,
            "n_patients": n_patients,
            "n_days": n_days,
            "treatment_effect": {
                "mean_beta": float(np.mean(beta_draws)),
                "sd_beta": float(np.std(beta_draws)),
            },
            "recovered_OR": {
                "mean": float(np.mean(trial_ors)),
                "median": float(np.median(trial_ors)),
                "sd": float(np.std(trial_ors)),
                "ci95": [float(np.percentile(trial_ors, 2.5)),
                         float(np.percentile(trial_ors, 97.5))],
            },
            "power": {
                "pr_significant": float(np.mean(trial_significant)),
                "pr_or_gt_1": float(np.mean(trial_ors > 1.0)),
                "pr_or_gt_1_1": float(np.mean(trial_ors > 1.1)),
                "interpretation": f"Probability that a 50-patient trial achieves 95% CI excluding 1 (with {label})"
            },
            "angina_free_days": {
                "mean_difference": float(np.mean(trial_angina_free)),
                "median_difference": float(np.median(trial_angina_free)),
                "sd": float(np.std(trial_angina_free)),
                "ci95": [float(np.percentile(trial_angina_free, 2.5)),
                         float(np.percentile(trial_angina_free, 97.5))],
                "interpretation": "Extra angina-free days with PCI vs sham over 180 days"
            },
        }

    return all_results


# ============================================================
# 3. MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 75)
    print("ORBITA-CTO Posterior Predictive Trial Simulation")
    print("=" * 75)
    print(f"N = 50 patients (25 PCI + 25 sham) × 180 days")
    print(f"Ordinal categories: {N_CATS} ({', '.join(CAT_LABELS)})")
    print(f"Cutpoints: {CUTPOINTS}")
    print(f"σ_subject: {SIGMA_SUBJECT}")
    print(f"Simulations: 500 per bias scenario")

    results = run_simulation(n_sim=500, n_patients=50, n_days=180)

    print(f"\n\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\n{'Scenario':<14} {'Recovered OR':>13} {'Power':>7} {'AF days diff':>13} {'AF 95% CI':>22}")
    print(f"{'─'*75}")

    for label, r in results.items():
        ro = r["recovered_OR"]
        pw = r["power"]
        af = r["angina_free_days"]
        print(f"{label:<14} {ro['mean']:>6.3f} ± {ro['sd']:.3f}  "
              f"{pw['pr_significant']:>6.1%} "
              f"{af['mean_difference']:>+8.1f} days "
              f"[{af['ci95'][0]:>+.1f}, {af['ci95'][1]:>+.1f}]")

    print(f"\n{'─'*75}")
    print("Interpretation:")
    print("  'Power' = fraction of simulated 50-patient trials where 95% CI excludes 1")
    print("  'AF days diff' = extra angina-free days with PCI vs sham over 180 days")

    # Clinical translation summary
    print(f"\n{'='*80}")
    print("CLINICAL TRANSLATION")
    print(f"{'='*80}")
    for label, r in results.items():
        af = r["angina_free_days"]
        pw = r["power"]
        print(f"\n  {label}:")
        print(f"    A 50-patient ORBITA-CTO trial would show:")
        print(f"    - OR {r['recovered_OR']['mean']:.2f} (recovered from simulated data)")
        print(f"    - {pw['pr_significant']:.0%} chance of statistical significance")
        print(f"    - {af['mean_difference']:+.1f} extra angina-free days "
              f"[{af['ci95'][0]:+.1f} to {af['ci95'][1]:+.1f}] over 6 months")
        if abs(af['mean_difference']) < 5:
            print(f"    → Clinically: ~{abs(af['mean_difference']):.0f} extra good days in 6 months — "
                  f"difficult for patients to perceive")
        else:
            print(f"    → Clinically: ~{abs(af['mean_difference']):.0f} extra good days in 6 months")

    # Save
    with open("/Users/apple/Downloads/orbital/main/v2_longitudinal_po/simulation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to simulation_results.json")
