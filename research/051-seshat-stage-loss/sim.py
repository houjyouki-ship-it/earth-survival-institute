"""研究051：Seshat Equinox-2020 の世紀ごとの時系列から、崩壊の後の段の喪失を機械的に数える（open_problems 137）。
規則は PREREGISTRATION.md に固定。seed 20261051。ネット接続なし（sources/ の xlsx を読む）。"""
import json, time, os
import numpy as np, pandas as pd
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261051
rng = np.random.default_rng(SEED)
XL = os.path.join(HERE, 'sources', 'Equinox_on_GitHub_June9_2022.xlsx')
x = pd.ExcelFile(XL)
ts = x.parse('TSDat123'); imp = x.parse('ImpSCDat'); spc = x.parse('SPC_MilTech')
key = ['NGA', 'PolID', 'Time']
df = ts.merge(imp[key + ['Pop', 'Dupl']].rename(columns={'Pop': 'ImpPop', 'Dupl': 'DuplI'}), on=key, how='left')
df = df.merge(spc[key + ['SPC']], on=key, how='left')
assert len(df) == len(ts)
dupl_col = 'Dupl' if 'Dupl' in ts.columns else 'DuplI'

TECH = ['Script', 'WRecord', 'PractLit', 'FullTBur', 'IndigCoin', 'Court', 'Market', 'Road', 'Bridge', 'Canal', 'Mine',
        'Copper', 'Bronze', 'Iron', 'Steel']
S2SET = ['Script', 'PractLit', 'FullTBur', 'IndigCoin', 'Court', 'Market', 'Road', 'Bridge', 'Canal', 'Mine', 'Bronze', 'Iron']

def tri(v, strict=False):
    """1=あり、0=なし、nan=不明。strict なら 1/0 だけを使う。"""
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    if strict:
        out[v == 1.0] = 1; out[v == 0.0] = 0
    else:
        out[v >= 0.9] = 1; out[(v >= 0) & (v <= 0.1)] = 0
    return out

def build_series(d, drop_dupl=False):
    """NGA -> 世紀ごとの表（同じ世紀の複数行は既知の値の最大）。"""
    if drop_dupl:
        d = d[d[dupl_col] != 'y']
    S = {}
    for nga, g in d.groupby('NGA'):
        g = g.sort_values('Time')
        rows = []
        for t, gg in g.groupby('Time'):
            r = {'Time': t, 'PolID': gg.PolID.iloc[0], 'Pop': gg.ImpPop.max(), 'SPC': gg.SPC.max()}
            for c in TECH:
                vals = gg[c].values.astype(float)
                vals = vals[vals != -999]
                r[c] = vals.max() if len(vals) else -999.0
            rows.append(r)
        S[nga] = pd.DataFrame(rows).reset_index(drop=True)
    return S

def metal_level(row_tri):
    m = 0
    for lev, c in [(1, 'Copper'), (2, 'Bronze'), (3, 'Iron'), (4, 'Steel')]:
        if row_tri[c] == 1: m = lev
    return m

def find_episodes(s, var='Pop', D=0.301):
    t = s.Time.values; v = s[var].values.astype(float)
    eps = []
    for i in range(len(s)):
        if np.isnan(v[i]): continue
        prev = [j for j in range(len(s)) if 100 <= t[i] - t[j] <= 200 and not np.isnan(v[j])]
        if not prev: continue
        jmax = max(prev, key=lambda j: v[j])
        if v[i] <= v[jmax] - D:
            if eps and t[jmax] - eps[-1]['t_peak'] <= 300 and t[i] - eps[-1]['t_peak'] <= 300 + 200:
                continue
            eps.append({'i_peak': jmax, 't_peak': int(t[jmax]), 't_onset': int(t[i]), 'drop': float(v[jmax] - v[i])})
    return eps

def loss_flags(s, ip, window=500, consec=2, strict=False):
    """peak の行 ip からの喪失。戻り値：変数 -> 1/0/nan（nan＝peak でなし、または不明）、金属の段。"""
    t = s.Time.values
    tr = {c: tri(s[c].values, strict) for c in TECH}
    tp = t[ip]
    if t[-1] - tp < 200:
        return None
    w = [j for j in range(len(s)) if tp < t[j] <= tp + window]
    out = {}
    for c in TECH:
        if tr[c][ip] != 1: out[c] = np.nan; continue
        vals = tr[c][w]
        if np.all(np.isnan(vals)): out[c] = np.nan; continue
        run = 0; lost = 0
        for vv in vals:
            run = run + 1 if vv == 0 else 0
            if run >= consec: lost = 1; break
        out[c] = lost
    # 金属の段
    mlev = []
    for j in [ip] + w:
        rt = {c: tr[c][j] for c in ['Copper', 'Bronze', 'Iron', 'Steel']}
        known = [c for c in rt if not np.isnan(rt[c])]
        mlev.append(np.nan if not known else metal_level(rt))
    m0 = mlev[0]
    def run_below(th):
        run = 0
        for mv in mlev[1:]:
            run = run + 1 if (not np.isnan(mv) and mv <= th) else 0
            if run >= consec: return 1
        return 0
    if np.isnan(m0) or m0 < 1 or all(np.isnan(mv) for mv in mlev[1:]):
        out['S4'] = np.nan; out['S4b'] = np.nan
    else:
        out['S4'] = run_below(0); out['S4b'] = run_below(m0 - 1)
    # 読み
    sc = out['Script'] if not np.isnan(out['Script']) else out['WRecord']
    out['S1'] = sc
    fb = out['FullTBur']
    out['S1b'] = np.nan if (np.isnan(sc) and np.isnan(fb)) else float(np.nanmax([sc if not np.isnan(sc) else 0, fb if not np.isnan(fb) else 0]))
    present = [c for c in S2SET if not np.isnan(out[c])]
    out['S2'] = np.nan if len(present) < 2 else float(sum(out[c] for c in present) >= 2)
    out['n_S2_present'] = len(present)
    out['metal_peak'] = m0
    return out

READ = ['S1', 'S1b', 'S2', 'S4', 'S4b']

def run_version(S, var='Pop', D=0.301, window=500, consec=2, strict=False, dedupe_pol=False):
    eps_all = []
    for nga, s in S.items():
        for e in find_episodes(s, var, D):
            f = loss_flags(s, e['i_peak'], window, consec, strict)
            if f is None: continue
            eps_all.append(dict(NGA=nga, PolID=s.PolID.iloc[e['i_peak']], **e, **{k: f[k] for k in READ + TECH + ['metal_peak', 'n_S2_present']}))
    E = pd.DataFrame(eps_all)
    if dedupe_pol and len(E):
        E = E.drop_duplicates(subset=['PolID', 't_peak'])
    # 背景
    bg = []
    for nga, s in S.items():
        win = set()
        for e in find_episodes(s, var, D):
            for j in range(len(s)):
                if e['t_peak'] - 0 <= s.Time.values[j] <= e['t_peak'] + window: win.add(j)
        for j in range(len(s)):
            if j in win: continue
            f = loss_flags(s, j, window, consec, strict)
            if f is None: continue
            bg.append({k: f[k] for k in READ})
    B = pd.DataFrame(bg)
    res = {'n_episodes': int(len(E)), 'n_NGA_with_episode': int(E.NGA.nunique()) if len(E) else 0}
    for r in READ:
        col = E[r].dropna() if len(E) else pd.Series(dtype=float)
        k, n = int(col.sum()), int(len(col))
        bcol = B[r].dropna()
        pbg = float(bcol.mean()) if len(bcol) else np.nan
        res[r] = {'k': k, 'n': n, 'point': k / n if n else None, 'post_mean': (k + 0.5) / (n + 1),
                  'post_q05_50_95': [float(q) for q in np.quantile(rng.beta(k + 0.5, n - k + 0.5, 20000), [0.05, 0.5, 0.95])] if n else None,
                  'bg_rate': pbg, 'bg_n': int(len(bcol)),
                  'ratio_to_bg': (k / n) / pbg if (n and pbg and pbg > 0) else None,
                  'bg_corrected': max(0.0, (k / n - pbg) / (1 - pbg)) if (n and not np.isnan(pbg)) else None}
    return res, E

def cluster_boot(E, r, B=2000):
    g = E[['NGA', r]].dropna()
    ngas = g.NGA.unique()
    if len(ngas) == 0: return None
    vals = []
    for _ in range(B):
        pick = rng.choice(ngas, len(ngas), replace=True)
        sub = pd.concat([g[g.NGA == a] for a in pick])
        k, n = sub[r].sum(), len(sub)
        vals.append((k + 0.5) / (n + 1))
    return [float(q) for q in np.quantile(vals, [0.05, 0.5, 0.95])]

S = build_series(df)
S_nodupl = build_series(df, drop_dupl=True)
R = {'study': '051-seshat-stage-loss', 'seed': SEED, 'n_NGA': len(S), 'n_rows': int(len(df))}
main, Emain = run_version(S)
R['main'] = main
R['main_cluster_boot'] = {r: cluster_boot(Emain, r) for r in READ}
versions = {
    'pop_drop_1.0': dict(var='Pop', D=1.0),
    'spc_drop_1.0': dict(var='SPC', D=1.0),
    'pop_drop_0.5': dict(var='Pop', D=0.5),
    'consec1': dict(consec=1),
    'window300': dict(window=300),
    'window1000': dict(window=1000),
    'strict_1_0': dict(strict=True),
    'dedupe_polity': dict(dedupe_pol=True),
}
R['versions'] = {}
for name, kw in versions.items():
    R['versions'][name] = run_version(S, **kw)[0]
R['versions']['drop_dupl_rows'] = run_version(S_nodupl)[0]
Emain.to_csv(os.path.join(HERE, 'episodes_main.csv'), index=False)

# 048 の h の線形の写し
r48 = json.load(open(os.path.join(HERE, '..', '048-unknown-depth-h', 'results.json')))
HSTAR = r48['B_h_star']['A_Ord']
def lin(acc):
    sc = r48['B_h_scan'][acc]
    h = np.array(sc['h']); tot = np.array(sc['total']); fh = tot * np.array(sc['share_from_heavy']); hp = np.array(sc['heavy_prob'])
    # 線形の確認
    for arr in (tot, fh, hp):
        a, b = np.polyfit(h, arr, 1); assert np.max(np.abs(a * h + b - arr)) < 1e-12 * max(1, np.max(np.abs(arr)))
    return (lambda hh: np.interp(hh, h, tot)), (lambda hh: np.interp(hh, h, fh)), (lambda hh: np.interp(hh, h, hp))
MAP = {acc: lin(acc) for acc in ['A_Ord', 'A_XPT_S']}
U3 = r48['B_U3_ref']
def value_block(k, n):
    hs = rng.beta(k + 0.5, n - k + 0.5, 20000)
    out = {}
    for acc, (ft, ff, fp) in MAP.items():
        tot = ft(hs).mean(); fh = ff(hs).mean()
        out[acc] = {'h_mean': float(hs.mean()), 'total': float(tot), 'from_heavy': float(fh), 'share_from_heavy': float(fh / tot),
                    'ratio_to_U3': float(tot / U3[acc]['total']), 'heavy_prob': float(fp(hs).mean()),
                    'P_h_gt_hstar': float((hs > HSTAR).mean()) if acc == 'A_Ord' else None}
    return out
R['h_star_A_Ord'] = HSTAR
R['value_main'] = {r: value_block(main[r]['k'], main[r]['n']) for r in READ}
R['ref_048_V2_A_Ord_total'] = r48['B_by_reading']['V2']['A_Ord']['total']

# 判定
m = main
pm = lambda r: m[r]['post_mean']
s1_versions = [pm('S1'), R['versions']['pop_drop_1.0']['S1']['post_mean'], R['versions']['spc_drop_1.0']['S1']['post_mean']]
v2val = R['value_main']['S2']['A_Ord']['total']
R['hypotheses'] = {
    'H63a_S1_postmean_le_0.15': {'value': pm('S1'), 'supported': bool(pm('S1') <= 0.15)},
    'H63b_S4_postmean_lt_hstar': {'value': pm('S4'), 'supported': bool(pm('S4') < HSTAR)},
    'H63c_S1_ratio_bg_ge_2': {'value': m['S1']['ratio_to_bg'], 'supported': bool(m['S1']['ratio_to_bg'] is not None and m['S1']['ratio_to_bg'] >= 2)},
    'H63d_S2_in_048_V2_interval': {'value': pm('S2'), 'supported': bool(0.056 <= pm('S2') <= 0.377)},
    'H63e_S2_gt_hstar_and_S4_lt_hstar': {'value': [pm('S2'), pm('S4')], 'supported': bool(pm('S2') > HSTAR and pm('S4') < HSTAR)},
    'H63f_S1_range_3denoms_lt_0.10': {'value': s1_versions, 'range': float(max(s1_versions) - min(s1_versions)), 'supported': bool(max(s1_versions) - min(s1_versions) < 0.10)},
    'H63g_S2_A_Ord_within_10pct_of_048_V2': {'value': v2val, 'ratio': v2val / R['ref_048_V2_A_Ord_total'], 'supported': bool(abs(v2val / R['ref_048_V2_A_Ord_total'] - 1) <= 0.10)},
}

# 版ごとの h* との関係（感度、事前登録の規則の版）
R['versions_vs_hstar'] = {}
for name, v in [('main', main)] + list(R['versions'].items()):
    R['versions_vs_hstar'][name] = {r: {'post_mean': v[r]['post_mean'], 'k': v[r]['k'], 'n': v[r]['n'],
        'P_h_gt_hstar': float((rng.beta(v[r]['k'] + 0.5, v[r]['n'] - v[r]['k'] + 0.5, 20000) > HSTAR).mean()),
        'A_Ord_total': float(MAP['A_Ord'][0]((v[r]['k'] + 0.5) / (v[r]['n'] + 1)))} for r in ['S1', 'S1b', 'S2', 'S4', 'S4b']}

# 048 と重なる例の照合（事後の探索、事前登録の外）
OVER = [('Mycenaean Greece (LBA)', 'Crete', 'GrCrNPa'), ('Hittite Anatolia (LBA)', 'Konya Plain', 'TrHatNK'),
        ('Indus Valley (late Harappan)', 'Kachi Plain', 'PkUrbn2'), ('Western Roman Empire', 'Latium', 'ItRomPr'),
        ('Western Roman Empire', 'Paris Basin', 'TrRomDm'), ('Angkor', 'Cambodian Basin', 'KhAngkE'), ('Cahokia / Mississippian', 'Cahokia', 'USCahoE')]
c48 = {c['name']: c for c in r48['A_cases']}
R['explore_overlap_048'] = []
for nm, nga, pol in OVER:
    row = Emain[(Emain.NGA == nga) & (Emain.PolID == pol)]
    e = row.iloc[0] if len(row) else None
    R['explore_overlap_048'].append({'case_048': nm, 'NGA': nga, 'PolID': pol, 'L_top_048': c48[nm]['L_top'], 'L_multi_048': c48[nm]['L_multi'],
        'S1': None if e is None else e['S1'], 'S1b': None if e is None else e['S1b'], 'S2': None if e is None else e['S2'], 'S4b': None if e is None else e['S4b']})
R['loss_episodes_main'] = Emain[(Emain[READ].fillna(0).sum(axis=1) > 0)][['NGA', 'PolID', 't_peak', 't_onset', 'drop'] + READ].to_dict(orient='records')

# 図
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(8, 4.5))
labs, mus, lo, hi, cols = [], [], [], [], []
for r, lab in [('S1', 'S1 script'), ('S1b', 'S1b script or bureaucrats'), ('S2', 'S2 >=2 techs'), ('S4', 'S4 all metal'), ('S4b', 'S4b metal -1 stage')]:
    q = main[r]['post_q05_50_95']; labs.append('Seshat ' + lab); mus.append(main[r]['post_mean']); lo.append(q[0]); hi.append(q[2]); cols.append('tab:blue')
for v, lab in [('V1', 'V1 top tier'), ('V2', '>=2 techs'), ('V3', 'isolated >=2'), ('V4', 'metal')]:
    q = r48['A_h_estimates'][v]['post_q05_50_95']; labs.append('048 ' + lab); mus.append(r48['A_h_estimates'][v]['post_mean']); lo.append(q[0]); hi.append(q[2]); cols.append('tab:orange')
y = np.arange(len(labs))[::-1]
for yi, m_, l_, h_, c_ in zip(y, mus, lo, hi, cols):
    ax.plot([l_, h_], [yi, yi], color=c_, lw=2); ax.plot(m_, yi, 'o', color=c_)
ax.axvline(HSTAR, color='k', ls='--', lw=1); ax.text(HSTAR + 0.005, len(labs) - 1.5, 'h* = %.3f (P1)' % HSTAR, fontsize=8)
ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8); ax.set_xlabel('h (posterior mean, 90% interval)')
ax.set_title('Stage loss after collapse: Seshat (73 episodes, 24 NGAs) vs 048 (17 cases)', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_h_seshat_vs_048.png'), dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5, 4.5))
vn = list(R['versions_vs_hstar'].keys()); xs = np.arange(len(vn))
for j, (r, mk) in enumerate([('S1', 'o'), ('S1b', 's'), ('S2', '^'), ('S4', 'v')]):
    ax.plot(xs + (j - 1.5) * 0.12, [R['versions_vs_hstar'][n][r]['post_mean'] for n in vn], mk, label=r)
ax.axhline(HSTAR, color='k', ls='--', lw=1, label='h* (P1)')
ax.set_xticks(xs); ax.set_xticklabels(vn, rotation=35, ha='right', fontsize=8); ax.set_ylabel('posterior mean h'); ax.legend(fontsize=8, ncol=5)
ax.set_title('Robustness of h readings across preregistered rule versions', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_versions.png'), dpi=130); plt.close(fig)

R['runtime_s'] = time.time() - T0

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    return o
json.dump(clean(R), open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1)

print('episodes', main['n_episodes'], 'NGAs', main['n_NGA_with_episode'])
for r in READ:
    q = m[r]
    print(f"{r}: k/n={q['k']}/{q['n']} point={q['point']} post={q['post_mean']:.3f} q={np.round(q['post_q05_50_95'],3) if q['post_q05_50_95'] else None} bg={q['bg_rate']} ratio={q['ratio_to_bg']} corr={q['bg_corrected']} cboot={R['main_cluster_boot'][r]}")
for name, v in R['versions'].items():
    print(name, v['n_episodes'], {r: (v[r]['k'], v[r]['n'], round(v[r]['post_mean'], 3)) for r in READ})
for r in READ:
    print('value', r, {a: {k: round(vv, 5) if isinstance(vv, float) else vv for k, vv in b.items()} for a, b in R['value_main'][r].items()})
for k, v in R['hypotheses'].items(): print(k, v)
for n, v in R['versions_vs_hstar'].items(): print('vs_hstar', n, {r: (round(q['post_mean'],3), round(q['P_h_gt_hstar'],3), round(q['A_Ord_total'],5)) for r, q in v.items()})
for o in R['explore_overlap_048']: print('overlap', o)
print('runtime', R['runtime_s'])
