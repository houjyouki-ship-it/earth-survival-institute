"""092 unpowered seed vault: Ellis-Roberts viability equation vs NordGen 100-year permafrost experiment.
No random numbers except the fixed-seed bootstrap. Run: python3 sim.py"""
import json, numpy as np, pandas as pd
from statistics import NormalDist
_N = NormalDist()
class norm:
    @staticmethod
    def ppf(p):
        p = np.asarray(p, float); return np.vectorize(_N.inv_cdf)(p) if p.ndim else _N.inv_cdf(float(p))
    @staticmethod
    def cdf(x):
        x = np.asarray(x, float); return np.vectorize(_N.cdf)(x) if x.ndim else _N.cdf(float(x))
rng = np.random.default_rng(92)
D = 365.25
vc = pd.read_csv('sources/viabilityconstants.csv')
# valid rows: positive Cw, Ch, Cq (exclude pathological fits e.g. sorghum Cw<0, Cq<0)
vc['valid'] = (vc.Cw > 1) & (vc.Ch > 0) & (vc.Cq > 1e-4) & (vc.Ke > 3)
vc['star'] = vc.Species.str.contains(r'\*')
vc['sp'] = vc.Species.str.replace('*', '', regex=False).str.strip()
def log_sigma(r, m, t):  # days
    return r.Ke - r.Cw*np.log10(m) - r.Ch*t - r.Cq*t*t
def sig_y(r, m, t): return 10**log_sigma(r, m, t)/D
v95, v85, v50 = norm.ppf(.95), norm.ppf(.85), norm.ppf(.5)
T = {'pw_-18': -18, 'perma_-3.5': -3.5, 'warm_-1.5': -1.5, 'warm_0': 0, 'warm_5': 5, 'amb_15': 15, 'amb_25': 25}
M = [4, 5, 6, 8]
V = vc[vc.valid].copy()
out = {'n_rows_total': int(len(vc)), 'n_valid': int(len(V)), 'n_valid_species_specific_T': int((~V.star).sum())}
tab = {}
for m in M:
    for k, t in T.items():
        s = np.array([sig_y(r, m, t) for r in V.itertuples()])
        tab[(m, k)] = s
        out[f'P50_med_m{m}_{k}'] = float(np.median(s*(v95-v50)))
        out[f'P85_med_m{m}_{k}'] = float(np.median(s*(v95-v85)))
# H104a ratio sigma(-18)/sigma(-3.5) at m=6, species-specific temperature constants only
Vs = V[~V.star]
ratio = np.array([sig_y(r, 6, -18)/sig_y(r, 6, -3.5) for r in Vs.itertuples()])
out['H104a_ratio_med'] = float(np.median(ratio)); out['H104a_ratio_iqr'] = [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))]
ratio_all = np.array([sig_y(r, 6, -18)/sig_y(r, 6, -3.5) for r in V.itertuples()])
out['ratio_med_allvalid'] = float(np.median(ratio_all))
# H104e moisture 6->4 gain vs temperature loss
mg = np.array([sig_y(r, 4, -3.5)/sig_y(r, 6, -3.5) for r in V.itertuples()])
out['H104e_moist_gain_med'] = float(np.median(mg))
# ---- NordGen comparison ----
ng = pd.read_csv('nordgen_annex1.csv')
def pick(sp):
    c = V[V.sp == sp]
    if len(c) == 0: return None
    c2 = c[~c.star]
    return (c2 if len(c2) else c).iloc[0]
def T_path(years, variant):
    if variant == 'const_-3.7': return np.full_like(years, -3.7)
    if variant == 'lin_-3.7_to_-1.5': return -3.7 + (2.2)*years/35.0   # NordGen estimate -1~-2 in 2021
    if variant == 'lin_-3.7_to_-1.0': return -3.7 + (2.7)*years/35.0
def pred_y35(r, y0, m, variant):
    yrs = np.linspace(0, 35, 351); tt = T_path(yrs, variant)
    rate = np.array([1/sig_y(r, m, t) for t in tt])   # probit units per year
    loss = (np.trapezoid if hasattr(np,'trapezoid') else np.trapz)(rate, yrs)
    return 100*norm.cdf(norm.ppf(min(y0, 99.5)/100) - loss), loss
rows = []
clip = lambda p: np.clip(p, 0.5, 99.5)/100
for x in ng.itertuples():
    r = pick(x.species)
    dp_obs = norm.ppf(clip(x.y0)) - norm.ppf(clip(x.y35))
    rec = dict(lot=x.lot, species=x.species, y0=x.y0, y35=x.y35, obs_dprobit=float(dp_obs),
               sigma_obs_y=float(35/dp_obs) if dp_obs > 0 else None,
               P50_obs_y=float(35/dp_obs*norm.ppf(clip(x.y0))) if dp_obs > 0 else None)
    if r is not None:
        rec['vc_row'] = r.Species + ' | ' + r.Reference
        for m in [3, 4, 5]:
            for var in ['const_-3.7', 'lin_-3.7_to_-1.5', 'lin_-3.7_to_-1.0']:
                y, loss = pred_y35(r, x.y0, m, var)
                rec[f'pred_y35_m{m}_{var}'] = float(y); rec[f'pred_dprobit_m{m}_{var}'] = float(loss)
    rows.append(rec)
R = pd.DataFrame(rows)
R.to_csv('nordgen_vs_equation.csv', index=False)
out['ng_n_lots'] = int(len(R)); out['ng_n_matched'] = int(R.vc_row.notna().sum())
out['ng_mean_y0'] = float(R.y0.mean()); out['ng_mean_y35'] = float(R.y35.mean())
out['ng_n_decline_gt5'] = int(((R.y0-R.y35) > 5).sum())
# H104c: observed loss rate vs predicted (ratio of probit loss; >=3 means equation overestimates longevity >=3x)
for m in [3, 4, 5]:
    for var in ['const_-3.7', 'lin_-3.7_to_-1.5', 'lin_-3.7_to_-1.0']:
        mm = R[R.vc_row.notna()]
        ratio = mm.obs_dprobit / mm[f'pred_dprobit_m{m}_{var}']   # sigma_pred/sigma_obs
        out[f'H104c_ratio_med_m{m}_{var}'] = float(np.median(ratio))
        out[f'H104c_frac_ratio_ge3_m{m}_{var}'] = float(np.mean(ratio >= 3))
mm = R[R.vc_row.notna()]
main_ratio = (mm.obs_dprobit / mm['pred_dprobit_m4_lin_-3.7_to_-1.5']).values
bs = [np.median(rng.choice(main_ratio, len(main_ratio))) for _ in range(5000)]
out['H104c_main_ratio_bootstrap95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
out['H104c_main_n'] = int(len(main_ratio))
# Empirical P50 from start at Svalbard (all 41 lots; non-declining lots censored = infinite)
p50 = R.P50_obs_y.fillna(np.inf).values
out['emp_P50_med_all41'] = float(np.median(p50))
for yrs in [50, 100, 200, 300, 500]:
    out[f'emp_frac_lots_P50_gt_{yrs}y'] = float(np.mean(p50 > yrs))
# time for 50% of lots to fall below 50% germination; per-lot germination at time t
def lot_germ(t):
    g = []
    for x in R.itertuples():
        if x.obs_dprobit <= 0: g.append(x.y0); continue
        g.append(100*norm.cdf(norm.ppf(clip(x.y0)) - t/x.sigma_obs_y))
    return np.array(g)
ts = np.arange(0, 501, 5)
frac50 = [float(np.mean(lot_germ(t) >= 50)) for t in ts]
out['emp_frac_lots_ge50pct_at'] = {int(t): f for t, f in zip(ts, frac50) if t in (35, 50, 100, 150, 200, 300, 500)}
# H104d: corrected equation P50 at -3.5 C (median over valid table rows) using main correction
corr = float(np.median(main_ratio))
out['correction_factor_main'] = corr
for m in [4, 5, 6]:
    s = tab[(m, 'perma_-3.5')] if m in M else None
    out[f'H104d_corrected_P50_med_m{m}'] = float(np.median(s*(v95-v50))/corr)
    out[f'corrected_P50_med_m{m}_amb_15'] = float(np.median(tab[(m, 'amb_15')]*(v95-v50))/corr)
    out[f'corrected_P50_med_m{m}_warm_5'] = float(np.median(tab[(m, 'warm_5')]*(v95-v50))/corr)
    out[f'corrected_P50_med_m{m}_pw_-18'] = float(np.median(tab[(m, 'pw_-18')]*(v95-v50))/corr)
# consistent correction: factor derived and applied at the same m (deviation from registration, see report)
for m in [3, 4, 5]:
    c = float(np.median(mm.obs_dprobit / mm[f'pred_dprobit_m{m}_lin_-3.7_to_-1.5']))
    for k, t in [('perma_-3.5', -3.5), ('warm_0', 0), ('warm_5', 5), ('pw_-18', -18), ('amb_15', 15)]:
        s_ = np.array([sig_y(r, m, t) for r in V.itertuples()])
        out[f'consistent_corr_P50_med_m{m}_{k}'] = float(np.median(s_*(v95-v50))/c)
out['walters2005_5C_P50_mean_reported'] = 52.0
# per crop group empirical
grp = {'cereal': ['Hordeum vulgare','Triticum aestivum','Secale cereale'], 'grass': ['Lolium perenne','Phleum pratense','Poa pratensis','Festuca pratensis'],
       'legume': ['Trifolium pratense','Pisum sativum'], 'brassica_beet': ['Brassica napus','Brassica oleracea','Beta vulgaris'],
       'vegetable_other': ['Allium cepa','Lactuca sativa','Cucumis sativus','Daucus carota']}
for g, sps in grp.items():
    sub = R[R.species.isin(sps)]
    out[f'group_{g}_n'] = int(len(sub)); out[f'group_{g}_mean_drop'] = float((sub.y0-sub.y35).mean())
    out[f'group_{g}_P50_med'] = float(np.median(sub.P50_obs_y.fillna(np.inf)))
staples = ['Hordeum vulgare','Triticum aestivum','Secale cereale','Pisum sativum']
sub = R[R.species.isin(staples)]
out['staple_lots_n'] = int(len(sub)); out['staple_P50_med'] = float(np.median(sub.P50_obs_y.fillna(np.inf)))
out['staple_frac_P50_gt_100y'] = float(np.mean(sub.P50_obs_y.fillna(np.inf) > 100))
# verdicts
out['H104a'] = 'support' if 1.8 <= out['H104a_ratio_med'] <= 3.5 else 'reject'
out['H104b'] = 'support' if out['P50_med_m6_perma_-3.5'] > 300 else 'reject'
out['H104c'] = 'support' if out['H104c_ratio_med_m4_lin_-3.7_to_-1.5'] >= 3 else 'reject'
out['H104d'] = 'support' if out['H104d_corrected_P50_med_m6'] < 150 else 'reject'
out['H104e'] = 'support' if out['H104e_moist_gain_med'] > out['ratio_med_allvalid'] else 'reject'
json.dump(out, open('results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in out.items(): print(k, v)
# figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
mmv = mm.copy()
ax[0].scatter(mmv['pred_y35_m4_lin_-3.7_to_-1.5'], mmv.y35, c='C0')
for _, x in mmv.iterrows(): ax[0].annotate(x['species'].split()[0][:6], (x['pred_y35_m4_lin_-3.7_to_-1.5'], x['y35']), fontsize=6)
ax[0].plot([0, 100], [0, 100], 'k--', lw=.8); ax[0].set_xlabel('Ellis-Roberts predicted germination at 35 y (%)\n(m=4%, -3.7 -> -1.5 C)'); ax[0].set_ylabel('NordGen observed 2021 (%)')
ax[0].set_title('Svalbard permafrost, 35 years')
ax[1].plot(ts, frac50, 'C3'); ax[1].set_xlabel('years in unpowered permafrost (extrapolated)'); ax[1].set_ylabel('share of 41 lots with germination >= 50%')
ax[1].set_ylim(0, 1.02); ax[1].axvline(35, color='gray', lw=.6); ax[1].set_title('Empirical extrapolation (probit-linear)')
plt.tight_layout(); plt.savefig('fig_unpowered_vault.png', dpi=130)
