"""
ORBITA-CTO Predictive Model v2.2 (Final)
==========================================
- Reparametrized: δ_open = δ_CTO + b_bias (inseparable)
- Crosswalk: α_0 + γ only
- μ_Δ = 0 fixed, robust MAP for μ
- Exact Shapley variance decomposition (4 groups, 24 permutations)
- Multi-chain convergence: 4 chains, R-hat, ESS
- Prior→posterior diagnostics: shrinkage, KL

Target estimand: Day-180 daily transition OR (Bayesian longitudinal PO scale)
"""

import numpy as np
import json
from scipy import stats
from itertools import permutations

# ============================================================
# DATA
# ============================================================
ORBITA2_BETA = 0.1991
ORBITA2_SE = 0.0288
GAMMA_POINT = ORBITA2_BETA / 14.4  # 0.01383

TRIALS = [
    {"name": "EuroCTO",      "saq_diff": 5.23, "saq_se": 1.78, "contam": 0},
    {"name": "COMET-CTO",    "saq_diff": 13.0, "saq_se": 4.56, "contam": 0},
    {"name": "DECISION-CTO", "saq_diff": 0.83, "saq_se": 0.76, "contam": 1},
]

PNAMES = ["mu", "delta_open", "kappa", "tau", "gamma", "alpha0", "sigma_cross", "w_map"]
NP = len(PNAMES)

# ============================================================
# LOG-POSTERIOR
# ============================================================
def logpost(p, trials, c):
    mu, dopen, kappa, tau, gamma, a0, scross, w = p
    if tau < 0 or scross < 0 or not (0 <= kappa <= 1) or not (0 <= w <= 1):
        return -np.inf

    lp = 0.0
    # μ: robust MAP
    se_infl = ORBITA2_SE * c["mu_inflate"]
    lp += np.logaddexp(
        np.log(max(w, 1e-30)) + stats.norm.logpdf(mu, ORBITA2_BETA, se_infl),
        np.log(max(1-w, 1e-30)) + stats.norm.logpdf(mu, 0, 0.5))
    lp += stats.beta.logpdf(w, 2, 2)
    # δ_open
    lp += stats.norm.logpdf(dopen, c["do_mean"], c["do_sd"])
    # κ
    lp += stats.beta.logpdf(kappa, 2, 2)
    # τ
    lp += stats.halfnorm.logpdf(tau, scale=c["tau_sc"])
    # γ
    lp += stats.norm.logpdf(gamma, GAMMA_POINT, c["g_sd"])
    # α_0
    lp += stats.norm.logpdf(a0, 0, c["a0_sd"])
    # σ_cross
    lp += stats.halfnorm.logpdf(scross, scale=c["sc_sc"])

    # Likelihood
    for t in trials:
        y = a0 + gamma * t["saq_diff"]
        v = gamma**2 * t["saq_se"]**2 + scross**2 + tau**2
        d_eff = dopen * (1 - kappa * t["contam"])
        theta = mu + d_eff
        lp += stats.norm.logpdf(y, theta, np.sqrt(max(v, 1e-10)))
    return lp

# ============================================================
# MCMC (single chain)
# ============================================================
def mcmc_chain(c, trials, n_iter=150000, burn=30000, thin=2, seed=42):
    rng = np.random.default_rng(seed)
    p = np.array([ORBITA2_BETA, c["do_mean"], 0.5, 0.05, GAMMA_POINT, 0.0, 0.10, 0.7])
    prop = np.array([0.012, 0.030, 0.08, 0.015, 0.002, 0.018, 0.030, 0.10])
    samps = np.zeros((n_iter, NP))
    lp = logpost(p, trials, c)
    nacc = 0
    for i in range(n_iter):
        q = p + rng.normal(0, prop)
        lq = logpost(q, trials, c)
        if np.log(rng.random()) < lq - lp:
            p, lp = q, lq
            nacc += 1
        samps[i] = p
    return samps[burn::thin], nacc / n_iter

# ============================================================
# MULTI-CHAIN CONVERGENCE
# ============================================================
def run_multi_chain(c, trials, n_chains=4, **kwargs):
    chains = []
    acc_rates = []
    for ch in range(n_chains):
        s, ar = mcmc_chain(c, trials, seed=42 + ch * 17, **kwargs)
        chains.append(s)
        acc_rates.append(ar)

    # R-hat (Gelman-Rubin)
    rhat = {}
    ess = {}
    for j, pn in enumerate(PNAMES):
        chain_means = [np.mean(ch[:, j]) for ch in chains]
        chain_vars = [np.var(ch[:, j], ddof=1) for ch in chains]
        n = len(chains[0])
        m = n_chains
        B = n * np.var(chain_means, ddof=1)
        W = np.mean(chain_vars)
        var_hat = (1 - 1/n) * W + B / n
        rhat[pn] = float(np.sqrt(var_hat / max(W, 1e-10)))
        # Bulk ESS approximation
        ess[pn] = int(m * n * min(1, W / max(var_hat, 1e-10)))

    combined = np.concatenate(chains)
    return combined, {"rhat": rhat, "ess": ess, "acc_rates": [float(a) for a in acc_rates]}

# ============================================================
# PREDICTION
# ============================================================
def predict(samples, rng=None):
    if rng is None:
        rng = np.random.default_rng(123)
    idx = {n: i for i, n in enumerate(PNAMES)}
    mu = samples[:, idx["mu"]]
    dopen = samples[:, idx["delta_open"]]
    tau = samples[:, idx["tau"]]
    noise = rng.normal(0, np.abs(tau))

    results = {}
    for b in [0.0, 0.02, 0.05, 0.08, 0.10]:
        pred = np.exp(mu + dopen - b + noise)
        lab = f"bias={b:.2f}"
        results[lab] = {
            "mean_OR": float(np.mean(pred)),
            "median_OR": float(np.median(pred)),
            "ci95": [float(np.percentile(pred, 2.5)), float(np.percentile(pred, 97.5))],
            "ci80": [float(np.percentile(pred, 10)), float(np.percentile(pred, 90))],
            "pr_benefit": float(np.mean(pred > 1.0)),
            "pr_gt_1_10": float(np.mean(pred > 1.10)),
        }
    return results

# ============================================================
# SHAPLEY VARIANCE (exact, 4! = 24 permutations)
# ============================================================
def shapley(samples, n_inner=300):
    idx = {n: i for i, n in enumerate(PNAMES)}
    N = len(samples)
    noise_rng = np.random.default_rng(999)

    groups = {
        "mu": ["mu", "w_map"],
        "delta_open": ["delta_open", "kappa"],
        "tau": ["tau"],
        "crosswalk": ["gamma", "alpha0", "sigma_cross"],
    }
    gnames = list(groups.keys())
    ng = len(gnames)

    def f(s):
        return s[:, idx["mu"]] + s[:, idx["delta_open"]] + noise_rng.normal(0, np.abs(s[:, idx["tau"]]))

    noise_rng = np.random.default_rng(999)
    total_var = np.var(f(samples))

    def cond_var(coalition):
        if len(coalition) == 0:
            return 0.0
        if len(coalition) == ng:
            return total_var
        fixed = [g for g in gnames if g not in coalition]
        fixed_idx = []
        for g in fixed:
            fixed_idx.extend([idx[p] for p in groups[g]])
        means = np.zeros(min(n_inner, N))
        for j in range(len(means)):
            sc = samples.copy()
            for pi in fixed_idx:
                sc[:, pi] = samples[j % N, pi]
            noise_rng_local = np.random.default_rng(999)
            means[j] = np.mean(sc[:, idx["mu"]] + sc[:, idx["delta_open"]] + noise_rng_local.normal(0, np.abs(sc[:, idx["tau"]])))
        return np.var(means)

    sv = {g: 0.0 for g in gnames}
    for perm in permutations(range(ng)):
        for pos in range(ng):
            g = gnames[perm[pos]]
            before = [gnames[perm[k]] for k in range(pos)]
            after = before + [g]
            sv[g] += (cond_var(after) - cond_var(before)) / 24  # 4! = 24

    total_sv = sum(sv.values())
    pct = {k: float(v / max(total_sv, 1e-10) * 100) for k, v in sv.items()}
    return {"percentages": pct, "sum_check": float(total_sv / max(total_var, 1e-10) * 100)}

# ============================================================
# PRIOR-POSTERIOR DIAGNOSTICS
# ============================================================
def diagnostics(samples, c):
    idx = {n: i for i, n in enumerate(PNAMES)}
    priors = {
        "mu":         ("normal", ORBITA2_BETA, ORBITA2_SE * c["mu_inflate"]),
        "delta_open": ("normal", c["do_mean"], c["do_sd"]),
        "kappa":      ("beta22", 0.5, 0.2236),
        "tau":        ("halfnorm", 0, c["tau_sc"] * np.sqrt(1 - 2/np.pi)),
        "gamma":      ("normal", GAMMA_POINT, c["g_sd"]),
        "alpha0":     ("normal", 0, c["a0_sd"]),
        "sigma_cross":("halfnorm", 0, c["sc_sc"] * np.sqrt(1 - 2/np.pi)),
        "w_map":      ("beta22", 0.5, 0.2236),
    }
    result = {}
    for pn in PNAMES:
        col = samples[:, idx[pn]]
        post_m, post_s = float(np.mean(col)), float(np.std(col))
        _, _, prior_s = priors[pn]
        shrink = max(0, 1 - (post_s / max(prior_s, 1e-10))**2)
        learned = "YES" if shrink > 0.20 else ("marginal" if shrink > 0.05 else "prior-driven")
        result[pn] = {"post_mean": post_m, "post_sd": post_s, "shrinkage": float(shrink), "learned": learned}
    return result

# ============================================================
# DEFAULT CONFIG
# ============================================================
DEF = {"mu_inflate": 2.0, "do_mean": -0.10, "do_sd": 0.20, "tau_sc": 0.10,
       "g_sd": 0.007, "a0_sd": 0.10, "sc_sc": 0.15}

SENS = {
    "Primary": {},
    "Vague δ_open": {"do_sd": 0.40},
    "No attenuation": {"do_mean": 0.0, "do_sd": 0.15},
    "Strong attenuation": {"do_mean": -0.20, "do_sd": 0.10},
    "Narrow μ (1.5×)": {"mu_inflate": 1.5},
    "Wide μ (3×)": {"mu_inflate": 3.0},
    "Wide crosswalk": {"sc_sc": 0.25, "g_sd": 0.010},
    "Wide τ": {"tau_sc": 0.15},
    "No DECISION": {"_excl": True},
    "ORBITA-2 only": {"_nodata": True, "do_sd": 0.30},
}

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 80)
    print("ORBITA-CTO v2.2 FINAL — Multi-chain, Shapley, Diagnostics")
    print("=" * 80)

    all_res = {}
    for name, ov in SENS.items():
        c = {**DEF, **{k: v for k, v in ov.items() if not k.startswith("_")}}
        if ov.get("_excl"):
            tr = [t for t in TRIALS if t["name"] != "DECISION-CTO"]
        elif ov.get("_nodata"):
            tr = []
        else:
            tr = TRIALS

        print(f"\n{'─'*70}")
        print(f"  {name} | Trials: {', '.join(t['name'] for t in tr) or 'none'}")
        print(f"{'─'*70}")

        if name == "Primary":
            # Multi-chain for primary
            samples, conv = run_multi_chain(c, tr, n_chains=4)
            print(f"  4 chains | Accept: {conv['acc_rates']}")
            rh = conv["rhat"]
            es = conv["ess"]
            print(f"  R-hat: {' | '.join(f'{k}={v:.3f}' for k,v in rh.items())}")
            converged = all(v < 1.1 for v in rh.values())
            print(f"  Converged (all R-hat<1.1): {'YES' if converged else 'NO'}")
            print(f"  ESS:  {' | '.join(f'{k}={v}' for k,v in es.items())}")
        else:
            samples, acc = mcmc_chain(c, tr)
            print(f"  Accept: {acc:.3f} | N={len(samples)}")

        pred = predict(samples)
        p0 = pred["bias=0.00"]
        p5 = pred["bias=0.05"]
        print(f"  bias=0.00: OR={p0['mean_OR']:.4f} [{p0['ci95'][0]:.4f}, {p0['ci95'][1]:.4f}] Pr(>1)={p0['pr_benefit']:.3f}")
        print(f"  bias=0.05: OR={p5['mean_OR']:.4f} [{p5['ci95'][0]:.4f}, {p5['ci95'][1]:.4f}] Pr(>1)={p5['pr_benefit']:.3f}")

        entry = {"prediction": pred}

        if name == "Primary":
            print(f"\n  Shapley variance decomposition (exact, 24 perms):")
            sh = shapley(samples)
            for g, pct in sh["percentages"].items():
                print(f"    {g}: {pct:.1f}%")
            print(f"    Sum check: {sh['sum_check']:.1f}%")
            entry["shapley"] = sh

            print(f"\n  Prior→Posterior:")
            dg = diagnostics(samples, c)
            print(f"    {'Param':<14} {'Mean':>8} {'SD':>8} {'Shrink':>8} {'Learned'}")
            for pn, d in dg.items():
                print(f"    {pn:<14} {d['post_mean']:>8.4f} {d['post_sd']:>8.4f} {d['shrinkage']:>7.1%} {d['learned']}")
            entry["diagnostics"] = dg
            entry["convergence"] = conv

        all_res[name] = entry

    # Summary
    print(f"\n\n{'='*95}")
    print(f"{'Config':<22} {'b=0 OR':>7} {'b=0 CrI':>18} {'b=.05 OR':>8} {'b=.05 CrI':>18} {'Pr(>1)':>7}")
    print(f"{'─'*95}")
    for n, r in all_res.items():
        p0 = r["prediction"]["bias=0.00"]
        p5 = r["prediction"]["bias=0.05"]
        print(f"{n:<22} {p0['mean_OR']:>7.3f} [{p0['ci95'][0]:.3f},{p0['ci95'][1]:.3f}] "
              f"{p5['mean_OR']:>8.3f} [{p5['ci95'][0]:.3f},{p5['ci95'][1]:.3f}] {p5['pr_benefit']:>7.3f}")

    # Evolution table
    print(f"\n{'='*80}")
    print("MODEL EVOLUTION")
    print(f"{'─'*80}")
    p0 = all_res["Primary"]["prediction"]["bias=0.00"]
    p5 = all_res["Primary"]["prediction"]["bias=0.05"]
    rows = [
        ("v1",    "Summary OR",     "1.33", "[0.45, 3.41]",  "2.96"),
        ("v2.0",  "Daily trans OR", "1.05", "[0.86, 1.29]",  "0.43"),
        ("v2.1",  "Expert fixes",   "1.10", "[0.75, 1.49]",  "0.74"),
        ("v2.2b0","Reparam b=0",    f"{p0['mean_OR']:.2f}", f"[{p0['ci95'][0]:.2f}, {p0['ci95'][1]:.2f}]", f"{p0['ci95'][1]-p0['ci95'][0]:.2f}"),
        ("v2.2b5","Reparam b=.05",  f"{p5['mean_OR']:.2f}", f"[{p5['ci95'][0]:.2f}, {p5['ci95'][1]:.2f}]", f"{p5['ci95'][1]-p5['ci95'][0]:.2f}"),
    ]
    print(f"{'Ver':<8} {'Estimand':<18} {'Mean OR':>8} {'95% CrI':>18} {'Width':>6}")
    for r in rows:
        print(f"{r[0]:<8} {r[1]:<18} {r[2]:>8} {r[3]:>18} {r[4]:>6}")

    # Save
    with open("/Users/apple/Downloads/orbital/main/v2_longitudinal_po/results_final.json", "w") as f:
        json.dump({"model": "v2.2_final", "results": all_res}, f, indent=2, default=str)
    print(f"\nSaved to results_final.json")
