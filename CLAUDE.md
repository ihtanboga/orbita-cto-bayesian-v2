# ORBITA-CTO v2: Longitudinal PO Bayesian Simulation

## Estimand
Day-180 daily transition OR from Bayesian longitudinal proportional-odds model — the exact metric ORBITA-CTO will report.

## GitHub
- Repo: https://github.com/ihtanboga/orbita-cto-bayesian-v2
- v1 (discussion page): https://github.com/ihtanboga/cto_discussion

## Dosyalar

| Dosya | Aciklama |
|---|---|
| `model_final.py` | v2.2 final model — multi-chain MCMC, Shapley, diagnostics |
| `simulate_trial.py` | N=50×180 posterior predictive trial simulation |
| `method.md` | Detayli metodoloji (expert round-1 + round-2 feedback) |
| `results_final.json` | Model sonuclari |
| `simulation_results.json` | Trial simulasyon sonuclari |
| `CLAUDE.md` | Bu dosya |

## Model Yapisi
```
log_OR_i = mu + delta_open × (1 - kappa × I_contam) + epsilon_i
delta_open = delta_CTO + b_bias  (ayrilamaz)
```

- mu: ORBITA-2 Bayesian PO beta'sindan (0.1991, SE=0.0288), robust MAP, cutting feedback
- delta_open: CTO attenuation + open-label bias (combined, inseparable)
- kappa: DECISION-CTO contamination dampening
- Crosswalk: SAQ AF diff → daily PO scale (alpha_0 + gamma)

## Sonuclar (v2.2 Primary)

| Bias senaryosu | Mean OR | 95% CrI | Pr(benefit) |
|---|---|---|---|
| bias=0.00 (optimistic) | 1.10 | 0.80-1.46 | 73% |
| bias=0.05 (moderate) | 1.05 | 0.76-1.39 | 60% |
| bias=0.10 (pessimistic) | 1.00 | 0.73-1.33 | 47% |

## Trial Simulasyon (N=50, 180 gun)

| Bias | Recovered OR | Power | Angina-free days diff |
|---|---|---|---|
| 0.00 | 1.13 | 54% | +2.9 days |
| 0.05 | 1.08 | 44% | +1.2 days |
| 0.10 | 1.05 | 45% | +0.1 days |

## Convergence
- 4 chains × 150K iterations
- All R-hat < 1.01, ESS > 237K

## Prior-Posterior Diagnostics
- delta_open: 48% shrinkage (data-informed)
- tau: 31% shrinkage (data-informed)
- sigma_cross: 44% shrinkage (data-informed)
- mu, kappa, w_map: prior-driven (by design)

## Expert Geri Bildirimleri
- Round 1: Estimand duzelttik, cutting feedback, crosswalk, posterior predictive
- Round 2: Reparametrize (delta_open), crosswalk sadelestirme, Shapley, KL diagnostics

## Model Evrimi

| Versiyon | Estimand | Mean OR | CrI width |
|---|---|---|---|
| v1 | Summary OR | 1.33 | 2.96 |
| v2.0 | Daily transition OR | 1.05 | 0.43 |
| v2.1 | Expert fixes | 1.10 | 0.74 |
| v2.2 (b=0) | Reparametrized | 1.10 | 0.66 |
| v2.2 (b=.05) | + bias scenario | 1.05 | 0.63 |

## Honest Framing
Model largely prior-driven. mu tamamen ORBITA-2'den (cutting feedback). delta_open kismi data-informed (3 CTO trial). delta_CTO ve b_bias ayrilamaz. MCMC belirsizlik propagasyonu yapiyor, yeni bilgi kesfetmiyor. Bu sinir acikca belirtilmeli.
