"""093 seed need vs lifespan. Deterministic except lognormal Monte Carlo (seed 93). Run: python3 sim.py"""
import json, numpy as np, pandas as pd
from statistics import NormalDist
_N = NormalDist()
class norm:  # same shim as 092 (no scipy on the box)
    @staticmethod
    def ppf(p):
        p = np.asarray(p, float); return np.vectorize(_N.inv_cdf)(p) if p.ndim else _N.inv_cdf(float(p))
    @staticmethod
    def cdf(x):
        x = np.asarray(x, float); return np.vectorize(_N.cdf)(x) if x.ndim else _N.cdf(float(x))
rng = np.random.default_rng(93)
out = {}
# --- 1. lifespan curve from 092 NordGen lots (re-implemented, checked against 092 results) ---
L = pd.read_csv('../092-unpowered-seed-vault/nordgen_annex1.csv')
clip = lambda p: np.clip(p, 0.5, 99.5)/100
dp = norm.ppf(clip(L.y0)) - norm.ppf(clip(L.y35))
sig = np.where(dp > 0, 35/np.where(dp > 0, dp, 1), np.inf)
staple = L.species.str.contains('Hordeum|Triticum|Secale|Pisum')
def germ(t, mask=None):
    g = 100*norm.cdf(norm.ppf(clip(L.y0)) - t/sig)
    g = np.where(dp > 0, g, L.y0)
    return g if mask is None else g[mask]
def f(t, mask=None): return float(np.mean(germ(t, mask) >= 50))
r092 = json.load(open('../092-unpowered-seed-vault/results.json'))['emp_frac_lots_ge50pct_at']
for k, v in r092.items(): assert abs(f(int(k)) - v) < 1e-12, (k, f(int(k)), v)
out['check_092_curve_reproduced'] = True
out['f0_all'] = f(0); out['f0_staple'] = f(0, staple.values)
ts = np.arange(0, 1501)
fa = np.array([f(t) for t in ts]); fs = np.array([f(t, staple.values) for t in ts])
def F(T, arr=fa):  # vectorised lookup, T in years
    return arr[np.clip(np.round(T).astype(int), 0, 1500)]
# --- 2. empirical cases ---
E = pd.read_csv('episodes.csv')
d = E.dropna(subset=['first_return', 'disrupt_start']).copy()
d['lag_from_start'] = d.first_return - d.disrupt_start
d['lag_from_end'] = d.first_return - d.disrupt_end.fillna(d.disrupt_start)
out['cases_dated'] = d[['case', 'lag_from_start', 'lag_from_end']].to_dict('records')
out['n_dated'] = int(len(d)); out['lag_median_start'] = float(d.lag_from_start.median())
out['lag_max_start'] = float(d.lag_from_start.max())
out['H105a'] = 'untestable(n<5)' if len(d) < 5 else ('support' if d.lag_from_start.median() <= 10 else 'reject')
av = E[E.availability_main_problem.isin(['yes', 'no'])]
out['n_avail_cases'] = int(len(av)); out['frac_avail_main_problem'] = float((av.availability_main_problem == 'yes').mean())
out['haiti_lack_of_seed_share_of_responses'] = 0.04
out['H105b'] = 'support' if out['frac_avail_main_problem'] < 1/3 else 'reject'
out['frac_return_from_cold_intl_genebank_not_svalbard'] = float((~d.return_source.str.contains('Svalbard')).mean())
out['H105e'] = 'support(descriptive)' if out['frac_return_from_cold_intl_genebank_not_svalbard'] > 0.5 else 'reject(descriptive)'
# --- 3. need-time scenarios ---
N = 200000
lags = d.lag_from_start.values.astype(float)
light = rng.choice(lags, N)  # light collapse: states/ships remain, need = empirical lag
out['light_f_mean_all'] = float(F(light).mean()); out['light_f_mean_staple'] = float(F(light, fs).mean())
out['light_f_rel_all'] = out['light_f_mean_all']/out['f0_all']
out['H105c'] = 'support' if out['light_f_mean_all'] >= 0.7 else 'reject'
# heavy collapse: reach of an Arctic island by survivors' seafaring. lognormal, sigma 0.7 [assumption]
#  H_norse: median 100 y (iron-age seafaring regained; Norse claimed 1194 disputed) [assumption]
#  H_casc / H_bio: median = 038's heavy-collapse T2 medians 254 / 1204 y (Barents 1596 was early-modern)
heavy = {}
for name, med in [('H_norse100', 100), ('H_casc254', 254), ('H_bio1204', 1204)]:
    T = med*np.exp(0.7*rng.standard_normal(N))
    loc = np.exp(-T/64.0)  # 061: location-knowledge time constant 64 y (variant)
    heavy[name] = dict(median_T=med, f_mean_all=float(F(T).mean()), f_mean_staple=float(F(T, fs).mean()),
                       f_rel_all=float(F(T).mean()/out['f0_all']),
                       f_times_location_memory=float((F(T)*loc).mean()),
                       location_memory_only=float(loc.mean()))
out['heavy'] = heavy
out['H105d'] = 'support' if heavy['H_casc254']['f_mean_all'] <= 0.35 else 'reject'
# break-even: median reach time at which mean f = 0.5 (sigma 0.7)
z = rng.standard_normal(N)
meds = np.arange(5, 1201, 5)
fm = np.array([F(m*np.exp(0.7*z)).mean() for m in meds])
out['breakeven_median_T_for_f50_all'] = int(meds[np.argmax(fm < 0.5)]) if (fm < 0.5).any() else None
fms = np.array([F(m*np.exp(0.7*z), fs).mean() for m in meds])
out['breakeven_median_T_for_f50_staple'] = int(meds[np.argmax(fms < 0.5)]) if (fms < 0.5).any() else None
# which constraint binds in heavy collapse: lifespan or location memory (ratio of losses)
for k, h in heavy.items():
    h['loss_from_lifespan'] = 1 - h['f_rel_all']
    h['loss_from_location'] = 1 - h['location_memory_only']
json.dump(out, open('results.json', 'w'), indent=1, ensure_ascii=False, default=str)
print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(ts, fa, label='all 41 lots (092)'); ax.plot(ts, fs, label='staple 12 lots')
for x in lags: ax.axvline(x, color='g', alpha=.4)
for k, c in [('H_norse100', 'orange'), ('H_casc254', 'red'), ('H_bio1204', 'purple')]:
    ax.axvline(heavy[k]['median_T'], color=c, ls='--', label=f'{k} median reach')
ax.set_xscale('symlog', linthresh=10); ax.set_xlabel('years after collapse'); ax.set_ylabel('share of lots with germination >=50%')
ax.set_title('093: unpowered Svalbard-type seed lifespan vs time of need (green: historical repatriation lags)')
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig('fig_need_vs_lifespan.png', dpi=120)
