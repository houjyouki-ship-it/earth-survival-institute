"""094 farming hiatus and P_loss. Ellis-Roberts ambient storage lifespan of farmer-saved seed (092 constants),
soil seed-bank persistence (published), historical cases, and a toy P_loss model. Deterministic. Run: python3 sim.py"""
import json, numpy as np, pandas as pd, csv
from statistics import NormalDist
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
N = NormalDist(); D = 365.25
vc = pd.read_csv('../092-unpowered-seed-vault/sources/viabilityconstants.csv')
vc['valid'] = (vc.Cw > 1) & (vc.Ch > 0) & (vc.Cq > 1e-4) & (vc.Ke > 3)   # identical to 092
vc['sp'] = vc.Species.str.replace('*', '', regex=False).str.strip()
V = vc[vc.valid].copy()
assert len(V) == 130, len(V)   # 092 used 130 valid rows
def sig_y(r, m, t): return 10**(r.Ke - r.Cw*np.log10(m) - r.Ch*t - r.Cq*t*t)/D
v95 = N.inv_cdf(.95)
def life(r, m, t, p): return sig_y(r, m, t)*(v95 - N.inv_cdf(p))
# reproduce 092 headline: P50 median at m=6%, -3.5C
r092 = json.load(open('../092-unpowered-seed-vault/results.json'))
chk = float(np.median([life(r, 6, -3.5, .5) for r in V.itertuples()]))
assert abs(chk - r092['P50_med_m6_perma_-3.5']) < 1e-9*max(1, chk), (chk, r092['P50_med_m6_perma_-3.5'])
STAPLE = r'^(Hordeum vulgare|Triticum aestivum|Secale cereale|Pisum sativum|Oryza sativa|Zea mays|Phaseolus vulgaris|Glycine max|Sorghum bicolor|Pennisetum glaucum|Eleusine coracana|Vigna unguiculata|Cicer arietinum|Lens culinaris|Arachis hypogaea)'
Vs = V[V.sp.str.contains(STAPLE, regex=True)]
out = {'n_valid': int(len(V)), 'n_staple_rows': int(len(Vs)), 'repro_092_P50_m6_-3.5': chk}
grid = {}
for lab, VV in [('all', V), ('staple', Vs)]:
    for m in [10, 12, 13, 14, 16]:
        for t in [10, 15, 20, 25, 30]:
            for p in [.5, .1]:
                x = np.array([life(r, m, t, p) for r in VV.itertuples()])
                grid[f'{lab}_m{m}_t{t}_P{int(p*100)}'] = float(np.median(x))
out['L_store_median_years'] = grid
main = grid['all_m13_t20_P50']; out['H106c_P50_all_m13_t20'] = main
Lst = {p: np.array([life(r, 13, 20, p) for r in Vs.itertuples()]) for p in [.5, .1]}
out['staple_P50_m13_t20_quantiles'] = [float(np.percentile(Lst[.5], q)) for q in (10, 25, 50, 75, 90)]
out['staple_P10_m13_t20_quantiles'] = [float(np.percentile(Lst[.1], q)) for q in (10, 25, 50, 75, 90)]
# soil seed bank (published, sources/): wheat EX99 0-15 cm 175..1114 d; none viable after ~3 yr
out['L_soil_wheat_years'] = {'EX99_min': 175/D, 'EX99_max': 1114/D, 'none_detected_by_yr': 3}
# ---- historical cases ----
cases = list(csv.DictReader(open('cases.csv')))
A = [c for c in cases if c['category'] == 'farming_continued']
B = [c for c in cases if c['category'] == 'crop_untended']
out['H106a'] = {'n': len(A), 'lost': sum(c['staple_lost'] == '1' for c in A)}
out['H106a']['frac'] = out['H106a']['lost']/max(1, len(A))
out['H106b'] = {'n': len(B), 'persisted': sum(c['staple_lost'] == '0' for c in B)}
out['H106b']['frac'] = out['H106b']['persisted']/max(1, len(B))
for prop in ['vegetative_or_perennial', 'annual_seed']:
    sub = [c for c in B if c['propagation'] == prop]
    out[f'B_{prop}'] = {'n': len(sub), 'persisted': sum(c['staple_lost'] == '0' for c in sub)}
# ---- toy P_loss model (all parameters [guess] versions) ----
# group loses a seed staple if farming halts (q_halt) AND (hiatus > L_store, or seed eaten with prob q_eat)
def p_loss_group(q_halt, med_h, q_eat, L, s=1.0):
    ph = np.array([1 - N.cdf((np.log(l) - np.log(med_h))/s) for l in L])   # P(hiatus > L) per lot-row
    return q_halt*(q_eat + (1 - q_eat)*ph.mean())
res = {}
for qh in [0.05, 0.2, 0.5]:
    for mh in [1, 3, 10, 30]:
        for qe in [0.0, 0.3]:
            for thr in [.5, .1]:
                res[f'qh{qh}_h{mh}_eat{qe}_P{int(thr*100)}'] = float(p_loss_group(qh, mh, qe, Lst[thr]))
out['P_loss_group'] = res
out['H106d_main'] = res['qh0.2_h3_eat0.0_P50']
out['main_with_eat'] = res['qh0.2_h3_eat0.3_P50']
# region with G groups: crop lost regionally only if all groups lose it (independent = lower bound; corr=1 upper)
pg = out['main_with_eat']
out['region_loss'] = {f'G{G}_indep': pg**G for G in [1, 3, 10, 30]}
out['region_loss']['fully_correlated'] = pg
# combine with 093: bank useful only where lost; 093 heavy main p(germ>=50% & location known)=0.039 (254 y)
out['bank_need_times_093_heavy_main'] = {k: v*0.039 for k, v in out['region_loss'].items()}
json.dump(out, open('results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in out.items():
    if k not in ('L_store_median_years', 'P_loss_group'): print(k, v)
print('L_store all m13 t20 P50', main, ' staple P10', grid['staple_m13_t20_P10'])
for k in ['all_m12_t15_P50', 'all_m14_t25_P50', 'staple_m13_t20_P50', 'staple_m12_t15_P50', 'staple_m10_t15_P50']: print(k, grid[k])
for k in sorted(res): print(k, round(res[k], 4))
# figure
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for p, c in [(.5, 'C0'), (.1, 'C1')]:
    x = np.sort(Lst[p]); ax[0].step(x, np.arange(1, len(x)+1)/len(x), c, where='post', label=f'staple rows, P{int(p*100)} (13%, 20°C)')
ax[0].axvspan(175/D, 3, color='grey', alpha=.25, label='wheat soil seed bank (EX99..3 y)')
ax[0].axvline(4, color='k', ls=':', label='Cambodia halt ~4 y'); ax[0].set_xscale('log'); ax[0].set_xlabel('years'); ax[0].set_ylabel('cumulative share of rows'); ax[0].legend(fontsize=7)
ax[0].set_title('How long farmer-saved seed survives a hiatus')
hs = [1, 3, 10, 30]
for qh, c in [(0.05, 'C2'), (0.2, 'C0'), (0.5, 'C3')]:
    ax[1].plot(hs, [res[f'qh{qh}_h{h}_eat0.3_P50'] for h in hs], 'o-', c=c, label=f'q_halt={qh}, eat 0.3')
    ax[1].plot(hs, [res[f'qh{qh}_h{h}_eat0.0_P50'] for h in hs], 'o--', c=c, alpha=.5)
ax[1].set_xscale('log'); ax[1].set_xlabel('median hiatus (y)'); ax[1].set_ylabel('P_loss per group (seed staple)'); ax[1].legend(fontsize=7)
ax[1].set_title('Toy P_loss (dashed: no seed eating)')
plt.tight_layout(); plt.savefig('fig_hiatus_ploss.png', dpi=130)
