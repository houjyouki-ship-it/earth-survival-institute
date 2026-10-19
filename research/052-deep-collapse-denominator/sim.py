"""研究052：循環のない分母で深い崩壊の類例を選び直し、S1・S2 と P(h>h*) を出す（143）。
征服期の符号変化の疑いを CanonDat で確かめる（144）。seed 20261052。ネット不要。"""
import json, time, os, re
import numpy as np, pandas as pd
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261052
rng = np.random.default_rng(SEED)
XL = os.path.join(HERE, 'sources', 'Equinox_on_GitHub_June9_2022.xlsx')
x = pd.ExcelFile(XL)
ts = x.parse('TSDat123'); imp = x.parse('ImpSCDat'); spc = x.parse('SPC_MilTech')
scale_mi = x.parse('Scale_MI'); canon = x.parse('Equinox2020_CanonDat')
key = ['NGA', 'PolID', 'Time']

TECH = ['Script', 'WRecord', 'PractLit', 'FullTBur', 'IndigCoin', 'Court', 'Market', 'Road', 'Bridge', 'Canal', 'Mine',
        'Copper', 'Bronze', 'Iron', 'Steel']
S2SET = ['Script', 'PractLit', 'FullTBur', 'IndigCoin', 'Court', 'Market', 'Road', 'Bridge', 'Canal', 'Mine', 'Bronze', 'Iron']
READ = ['S1', 'S1b', 'S2', 'S4', 'S4b']
CONQUEST_RE = re.compile(
    r'conquest|conquer|conquered|arab(?:ic)?\s+conquest|islamic\s+conquest|muslim\s+conquest|'
    r'ottoman\s+conquest|mongol\s+conquest|spanish\s+conquest|norman\s+conquest|'
    r'invasi(?:on|ons)|annex(?:ed|ation)',
    re.I)

def tri(v, strict=False):
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    if strict:
        out[v == 1.0] = 1; out[v == 0.0] = 0
    else:
        out[v >= 0.9] = 1; out[(v >= 0) & (v <= 0.1)] = 0
    return out

def zmean(df, cols):
    """各列を非 NaN で z 化し、行ごとの平均（少なくとも1列ある行）。"""
    Z = []
    for c in cols:
        v = df[c].astype(float)
        m, s = np.nanmean(v), np.nanstd(v, ddof=0)
        if not np.isfinite(s) or s < 1e-12:
            Z.append(np.zeros(len(df)))
        else:
            Z.append(((v - m) / s).to_numpy())
    A = np.vstack(Z).T
    with np.errstate(all='ignore'):
        out = np.nanmean(A, axis=1)
    return out

def pc1_scores(df, cols):
    """完全ケースで PCA 第1成分を当て、全行に射影（欠測は列平均で埋める）。符号は平均負荷が正になる向き。"""
    X = df[cols].astype(float).to_numpy()
    col_mean = np.nanmean(X, axis=0)
    Xf = np.where(np.isnan(X), col_mean, X)
    ok = ~np.any(np.isnan(df[cols].astype(float).to_numpy()), axis=1)
    Xc = Xf[ok]
    mu = Xc.mean(axis=0); sd = Xc.std(axis=0, ddof=0); sd = np.where(sd < 1e-12, 1.0, sd)
    Zn = (Xc - mu) / sd
    C = Zn.T @ Zn / max(1, len(Zn) - 1)
    w, V = np.linalg.eigh(C)
    vec = V[:, -1]
    if vec.mean() < 0: vec = -vec
    Zall = (Xf - mu) / sd
    return Zall @ vec

# --- Scale_MI: per-imputation Scale_z, then median ---
scale_rows = []
for n, g in scale_mi.groupby('n'):
    g = g.copy()
    g['Scale_z'] = zmean(g, ['Pop', 'Terr', 'Cap'])
    g['Scale_pc1'] = pc1_scores(g, ['Pop', 'Terr', 'Cap'])
    scale_rows.append(g[key + ['Scale_z', 'Scale_pc1', 'n']])
scale_long = pd.concat(scale_rows, ignore_index=True)
scale_med = (scale_long.groupby(key, as_index=False)
             .agg(Scale_z=('Scale_z', 'median'), Scale_pc1=('Scale_pc1', 'median')))

# --- ImpSCDat complexity without Info ---
imp = imp.copy()
imp['NoInfo_z'] = zmean(imp, ['Pop', 'Terr', 'Cap', 'Hier', 'Gov', 'Infra', 'Money'])
imp['NoInfo_pc1'] = pc1_scores(imp, ['Pop', 'Terr', 'Cap', 'Hier', 'Gov', 'Infra', 'Money'])
imp['HierGov_z'] = zmean(imp, ['Hier', 'Gov'])

# merge onto TSDat123
df = ts.merge(imp[key + ['Pop', 'Dupl', 'NoInfo_z', 'NoInfo_pc1', 'HierGov_z']].rename(
    columns={'Pop': 'ImpPop', 'Dupl': 'DuplI'}), on=key, how='left')
df = df.merge(spc[key + ['SPC']], on=key, how='left')
df = df.merge(scale_med, on=key, how='left')
assert len(df) == len(ts)
dupl_col = 'Dupl' if 'Dupl' in ts.columns else 'DuplI'

DENOM_VARS = {
    'D_pop': 'Pop',
    'D_spc': 'SPC',
    'D_scale': 'Scale_z',
    'D_scale_pc1': 'Scale_pc1',
    'D_noinfo': 'NoInfo_z',
    'D_noinfo_pc1': 'NoInfo_pc1',
    'D_hiergov': 'HierGov_z',
}

def build_series(d, drop_dupl=False):
    if drop_dupl:
        d = d[d[dupl_col] != 'y']
    S = {}
    for nga, g in d.groupby('NGA'):
        g = g.sort_values('Time')
        rows = []
        for t, gg in g.groupby('Time'):
            r = {'Time': t, 'PolID': gg.PolID.iloc[0],
                 'Pop': gg.ImpPop.max(), 'SPC': gg.SPC.max(),
                 'Scale_z': gg.Scale_z.max(), 'Scale_pc1': gg.Scale_pc1.max(),
                 'NoInfo_z': gg.NoInfo_z.max(), 'NoInfo_pc1': gg.NoInfo_pc1.max(),
                 'HierGov_z': gg.HierGov_z.max()}
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
            eps.append({'i_peak': jmax, 't_peak': int(t[jmax]), 't_onset': int(t[i]),
                        'drop': float(v[jmax] - v[i])})
    return eps

def loss_flags(s, ip, window=500, consec=2, strict=False):
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
    sc = out['Script'] if not np.isnan(out['Script']) else out['WRecord']
    out['S1'] = sc
    fb = out['FullTBur']
    out['S1b'] = np.nan if (np.isnan(sc) and np.isnan(fb)) else float(np.nanmax([sc if not np.isnan(sc) else 0, fb if not np.isnan(fb) else 0]))
    present = [c for c in S2SET if not np.isnan(out[c])]
    out['S2'] = np.nan if len(present) < 2 else float(sum(out[c] for c in present) >= 2)
    out['n_S2_present'] = len(present)
    out['metal_peak'] = m0
    return out

def run_version(S, var='Pop', D=0.301, window=500, consec=2, strict=False, dedupe_pol=False):
    eps_all = []
    for nga, s in S.items():
        for e in find_episodes(s, var, D):
            f = loss_flags(s, e['i_peak'], window, consec, strict)
            if f is None: continue
            eps_all.append(dict(NGA=nga, PolID=s.PolID.iloc[e['i_peak']], **e,
                                **{k: f[k] for k in READ + TECH + ['metal_peak', 'n_S2_present']}))
    E = pd.DataFrame(eps_all)
    if dedupe_pol and len(E):
        E = E.drop_duplicates(subset=['PolID', 't_peak'])
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
    res = {'n_episodes': int(len(E)), 'n_NGA_with_episode': int(E.NGA.nunique()) if len(E) else 0,
           'var': var, 'D': float(D)}
    for r in READ:
        col = E[r].dropna() if len(E) else pd.Series(dtype=float)
        k, n = int(col.sum()), int(len(col))
        bcol = B[r].dropna()
        pbg = float(bcol.mean()) if len(bcol) else np.nan
        res[r] = {'k': k, 'n': n, 'point': (k / n if n else None),
                  'post_mean': (k + 0.5) / (n + 1),
                  'post_q05_50_95': ([float(q) for q in np.quantile(rng.beta(k + 0.5, n - k + 0.5, 20000), [0.05, 0.5, 0.95])] if n else None),
                  'bg_rate': pbg, 'bg_n': int(len(bcol)),
                  'ratio_to_bg': ((k / n) / pbg if (n and pbg and pbg > 0) else None),
                  'bg_corrected': (max(0.0, (k / n - pbg) / (1 - pbg)) if (n and not np.isnan(pbg)) else None)}
    return res, E

S = build_series(df)

# 048 mapping
r48 = json.load(open(os.path.join(HERE, '..', '048-unknown-depth-h', 'results.json')))
HSTAR = r48['B_h_star']['A_Ord']
def lin(acc):
    sc = r48['B_h_scan'][acc]
    h = np.array(sc['h']); tot = np.array(sc['total']); fh = tot * np.array(sc['share_from_heavy']); hp = np.array(sc['heavy_prob'])
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
        out[acc] = {'h_mean': float(hs.mean()), 'total': float(tot), 'from_heavy': float(fh),
                    'share_from_heavy': float(fh / tot), 'ratio_to_U3': float(tot / U3[acc]['total']),
                    'heavy_prob': float(fp(hs).mean()),
                    'P_h_gt_hstar': float((hs > HSTAR).mean()) if acc == 'A_Ord' else None}
    return out

def summarize_version(res, E=None):
    out = dict(res)
    out['value'] = {r: value_block(res[r]['k'], res[r]['n']) for r in READ}
    out['vs_hstar'] = {r: {
        'post_mean': res[r]['post_mean'], 'k': res[r]['k'], 'n': res[r]['n'],
        'P_h_gt_hstar': float((rng.beta(res[r]['k'] + 0.5, res[r]['n'] - res[r]['k'] + 0.5, 20000) > HSTAR).mean()),
        'A_Ord_total': float(MAP['A_Ord'][0]((res[r]['k'] + 0.5) / (res[r]['n'] + 1)))
    } for r in READ}
    return out

# --- Main denominators ---
SPECS = [
    ('D_pop', 'Pop', 0.301),
    ('D_spc', 'SPC', 1.0),
    ('D_scale', 'Scale_z', 0.301),
    ('D_scale_D0.5', 'Scale_z', 0.5),
    ('D_scale_D1.0', 'Scale_z', 1.0),
    ('D_scale_pc1', 'Scale_pc1', 0.301),
    ('D_noinfo', 'NoInfo_z', 1.0),
    ('D_noinfo_D0.5', 'NoInfo_z', 0.5),
    ('D_noinfo_pc1', 'NoInfo_pc1', 1.0),
    ('D_hiergov', 'HierGov_z', 1.0),
]

R = {'study': '052-deep-collapse-denominator', 'seed': SEED, 'n_NGA': len(S), 'n_rows': int(len(df)),
     'h_star_A_Ord': HSTAR, 'preregistration': 'PREREGISTRATION.md @ 2026-10-19T02:44:10Z',
     'ref_051_S2_A_Ord': 0.00461}
R['denominators'] = {}
episodes_store = {}
for name, var, D in SPECS:
    res, E = run_version(S, var=var, D=D)
    R['denominators'][name] = summarize_version(res, E)
    episodes_store[name] = E
    E.to_csv(os.path.join(HERE, f'episodes_{name}.csv'), index=False)

# --- 144: conquest / polity-succession review ---
# Equinox CanonDat の Value.Note は叙述でなく simple/list/range/uncertain/disputed の種別のみ。
# 征服の疑いは Polities の End と後継政体名、および TSDat の PolID 切替で記録する。
polities = x.parse('Polities')
CONQUEST_SUCCESSOR = re.compile(
    r'Umayyad|Abbasid|Caliphate|Ottoman|Mongol Empire|Il-khan|Spanish|Norman|Rashidun|'
    r'Arab|Islamic|Muslim|Conquest|colonial|Colonial', re.I)

def successor_polities(nga, end_year, window=80):
    rows = polities[polities.NGA == nga]
    out = []
    for _, r in rows.iterrows():
        if pd.isna(r['Start']): continue
        if end_year - 5 <= float(r['Start']) <= end_year + window:
            out.append({'PolID': str(r['PolID']), 'PolName': str(r['PolName']),
                        'Start': int(r['Start']), 'End': None if pd.isna(r['End']) else int(r['End'])})
    return out

def polity_row(nga, polid):
    rows = polities[(polities.NGA == nga) & (polities.PolID == polid)]
    if not len(rows):
        rows = polities[polities.PolID == polid]
    return None if not len(rows) else rows.iloc[0]

E_pop = episodes_store['D_pop']
focus = E_pop[(E_pop['S2'] == 1) | (E_pop['S1'] == 1) | (E_pop['S4'] == 1)].copy() if len(E_pop) else E_pop
conquest_review = []
S2_excl = []
for _, row in focus.iterrows():
    lost = lost_techs(row)
    pr = polity_row(row['NGA'], row['PolID'])
    end = None if pr is None or pd.isna(pr['End']) else int(pr['End'])
    name = None if pr is None else str(pr['PolName'])
    succs = successor_polities(row['NGA'], end, 80) if end is not None else []
    succ_flag = any(CONQUEST_SUCCESSOR.search(s['PolName'] or '') for s in succs)
    # 征服終了そのもの：政体名が帝国で End が征服世紀と一致し、直後がカリフ制など
    end_near_onset = (end is not None and abs(end - int(row['t_onset'])) <= 50)
    # TSDat: peak 政体の次の世紀の PolID
    snga = S[row['NGA']]
    after = snga[snga.Time > row['t_peak']].head(3)
    next_pols = list(after.PolID.astype(str)) if len(after) else []
    suspect = bool(succ_flag and end_near_onset and row.get('S2') == 1.0)
    # 明示的な2件（051報告・事前登録で名指し）も Polities で裏取りできれば除外
    named = (row['NGA'], row['PolID']) in {('Susiana', 'IrSasn2'), ('Upper Egypt', 'TrERom*')}
    if named and succ_flag:
        suspect = True
    # CanonDat の Value.Note 種別（叙述なしであることの記録）
    sub = canon[(canon.NGA == row['NGA']) & (canon.Polity.astype(str) == str(row['PolID']))]
    note_types = sub['Value.Note'].astype(str).value_counts().head(5).to_dict() if len(sub) else {}
    date_notes = [str(v) for v in sub['Date.Note'].dropna().astype(str).unique()[:5]]
    conquest_review.append({
        'NGA': row['NGA'], 'PolID': row['PolID'], 'PolName': name,
        't_peak': int(row['t_peak']), 't_onset': int(row['t_onset']),
        'polity_End': end,
        'S1': None if pd.isna(row['S1']) else float(row['S1']),
        'S2': None if pd.isna(row['S2']) else float(row['S2']),
        'S4': None if pd.isna(row['S4']) else float(row['S4']),
        'lost_S2_techs': lost,
        'successors': succs[:5], 'next_TSDat_PolIDs': next_pols,
        'successor_conquest_name': succ_flag, 'end_near_onset': end_near_onset,
        'suspect_conquest_coding': suspect,
        'Value.Note_types': note_types, 'Date.Note_samples': date_notes,
        'evidence': 'Polities Start/End + successor PolName; CanonDat Value.Note is coding-type only (not narrative)',
    })
    if suspect and row.get('S2') == 1.0:
        S2_excl.append((row['NGA'], row['PolID'], int(row['t_peak'])))

E_clean = E_pop.copy()
if len(E_clean):
    for nga, pol, tp in S2_excl:
        m = (E_clean.NGA == nga) & (E_clean.PolID == pol) & (E_clean.t_peak == tp)
        E_clean.loc[m, 'S2'] = 0.0

def stats_from_E(E, r='S2'):
    col = E[r].dropna() if len(E) else pd.Series(dtype=float)
    k, n = int(col.sum()), int(len(col))
    return {'k': k, 'n': n, 'point': k / n if n else None, 'post_mean': (k + 0.5) / (n + 1),
            'post_q05_50_95': [float(q) for q in np.quantile(rng.beta(k + 0.5, n - k + 0.5, 20000), [0.05, 0.5, 0.95])] if n else None}

s2_clean_stats = stats_from_E(E_clean, 'S2')
s2_clean_value = value_block(s2_clean_stats['k'], s2_clean_stats['n'])
R['conquest_review'] = conquest_review
R['S2_clean_excluded'] = [{'NGA': a, 'PolID': b, 't_peak': c} for a, b, c in S2_excl]
R['D_pop_S2_clean'] = {**s2_clean_stats, 'value': s2_clean_value,
    'P_h_gt_hstar': float((rng.beta(s2_clean_stats['k'] + 0.5, s2_clean_stats['n'] - s2_clean_stats['k'] + 0.5, 20000) > HSTAR).mean())}
E_clean.to_csv(os.path.join(HERE, 'episodes_D_pop_S2_clean.csv'), index=False)
os.makedirs(os.path.join(HERE, 'sources'), exist_ok=True)
with open(os.path.join(HERE, 'sources', 'canon_conquest_excerpts.json'), 'w') as f:
    json.dump(conquest_review, f, ensure_ascii=False, indent=1)
with open(os.path.join(HERE, 'sources', 'polities_succession_excerpts.md'), 'w') as f:
    f.write('# 征服期の継承（Polities + TSDat）\n\n')
    f.write('CanonDat の Value.Note は simple/list/range/uncertain/disputed の種別のみで、征服の叙述は含まれない。\n\n')
    for c in conquest_review:
        if c['S2'] == 1.0 or c['suspect_conquest_coding']:
            f.write(f"## {c['NGA']} {c['PolID']} ({c.get('PolName')}) peak={c['t_peak']} End={c.get('polity_End')}\n")
            f.write(f"- S2={c['S2']} lost={c['lost_S2_techs']} suspect={c['suspect_conquest_coding']}\n")
            f.write(f"- successors={c['successors']}\n- next_TSDat={c['next_TSDat_PolIDs']}\n\n")

# --- Hypotheses ---
d_scale = R['denominators']['D_scale']
d_noinfo = R['denominators']['D_noinfo']
d_spc = R['denominators']['D_spc']
d_pop = R['denominators']['D_pop']
pm = lambda block: block['S2']['post_mean']
Pgt = lambda block: block['vs_hstar']['S2']['P_h_gt_hstar']
Aord = lambda block: block['value']['S2']['A_Ord']['total']

H = {
    'H64a_D_scale_S2_lt_hstar': {
        'value': pm(d_scale), 'hstar': HSTAR,
        'supported': bool(pm(d_scale) < HSTAR)},
    'H64b_D_noinfo_S2_lt_hstar': {
        'value': pm(d_noinfo), 'hstar': HSTAR,
        'supported': bool(pm(d_noinfo) < HSTAR)},
    'H64c_abs_diff_scale_noinfo_lt_0.05': {
        'value': abs(pm(d_scale) - pm(d_noinfo)),
        'supported': bool(abs(pm(d_scale) - pm(d_noinfo)) < 0.05)},
    'H64d_D_spc_S2_gt_D_scale_S2': {
        'value': [pm(d_spc), pm(d_scale)],
        'supported': bool(pm(d_spc) > pm(d_scale))},
    'H64e_both_cyclefree_P_lt_0.5': {
        'value': {'D_scale': Pgt(d_scale), 'D_noinfo': Pgt(d_noinfo)},
        'supported': bool(Pgt(d_scale) < 0.5 and Pgt(d_noinfo) < 0.5)},
    'H64f_S2_clean_le_S2': {
        'value': [s2_clean_stats['post_mean'], pm(d_pop)],
        'supported': bool(s2_clean_stats['post_mean'] <= pm(d_pop))},
    'H64g_D_scale_A_Ord_within_15pct_of_051': {
        'value': Aord(d_scale), 'ref': 0.00461,
        'ratio': Aord(d_scale) / 0.00461,
        'supported': bool(abs(Aord(d_scale) / 0.00461 - 1) <= 0.15)},
}
R['hypotheses'] = H

# Policy conclusion
cyclefree_both_below = bool(pm(d_scale) < HSTAR and pm(d_noinfo) < HSTAR and H['H64e_both_cyclefree_P_lt_0.5']['supported'])
R['policy'] = {
    'heavy_collapse_priority': '根拠なし' if cyclefree_both_below else '根拠残存（要解釈）',
    'cyclefree_both_S2_below_hstar': cyclefree_both_below,
    'recommend_next': '122（避難所と金庫の板の費用）' if cyclefree_both_below else '分母・読みの追加感度または122',
}

# Compact main numbers for report
R['main_summary'] = {
    name: {
        'n_episodes': R['denominators'][name]['n_episodes'],
        'S1_post': R['denominators'][name]['S1']['post_mean'],
        'S2_k_n': [R['denominators'][name]['S2']['k'], R['denominators'][name]['S2']['n']],
        'S2_post': R['denominators'][name]['S2']['post_mean'],
        'S2_P_gt_hstar': R['denominators'][name]['vs_hstar']['S2']['P_h_gt_hstar'],
        'S2_A_Ord': R['denominators'][name]['value']['S2']['A_Ord']['total'],
        'S2_share_from_heavy': R['denominators'][name]['value']['S2']['A_Ord']['share_from_heavy'],
    } for name in ['D_pop', 'D_spc', 'D_scale', 'D_noinfo', 'D_hiergov', 'D_scale_D0.5', 'D_scale_D1.0', 'D_noinfo_D0.5']
}

# --- Figures ---
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8.5, 4.8))
names = ['D_pop', 'D_scale', 'D_noinfo', 'D_spc', 'D_hiergov']
labs = ['D_pop\n(half pop)', 'D_scale\n(cycle-free)', 'D_noinfo\n(cycle-free)', 'D_spc\n(has Info)', 'D_hiergov\n(admin)']
cols = ['tab:blue', 'tab:green', 'tab:olive', 'tab:red', 'tab:purple']
for i, (nm, lab, c) in enumerate(zip(names, labs, cols)):
    b = R['denominators'][nm]
    q = b['S2']['post_q05_50_95']; m = b['S2']['post_mean']
    ax.plot([q[0], q[2]], [i, i], color=c, lw=2.5)
    ax.plot(m, i, 'o', color=c, ms=8)
    ax.text(q[2] + 0.005, i, f"k/n={b['S2']['k']}/{b['S2']['n']}", va='center', fontsize=8)
ax.axvline(HSTAR, color='k', ls='--', lw=1)
ax.text(HSTAR + 0.002, len(names) - 0.6, f'h*={HSTAR:.3f}', fontsize=8)
ax.set_yticks(range(len(names))); ax.set_yticklabels(labs, fontsize=9)
ax.set_xlabel('S2 posterior mean h (90% Jeffreys interval)')
ax.set_title('Deep-collapse denominators: S2 vs h* (cycle-free in green)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_denominators_S2_vs_hstar.png'), dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 4.5))
# cycle vs no-cycle comparison bar
pairs = [('D_pop', 0), ('D_scale', 1), ('D_noinfo', 2), ('D_spc', 3)]
xs = np.arange(len(pairs))
posts = [R['denominators'][n]['S2']['post_mean'] for n, _ in pairs]
Ps = [R['denominators'][n]['vs_hstar']['S2']['P_h_gt_hstar'] for n, _ in pairs]
Aords = [R['denominators'][n]['value']['S2']['A_Ord']['total'] for n, _ in pairs]
colors = ['tab:blue', 'tab:green', 'tab:olive', 'tab:red']
bars = ax.bar(xs - 0.2, posts, 0.4, color=colors, label='S2 post mean')
ax2 = ax.twinx()
ax2.plot(xs + 0.2, Ps, 'D-', color='k', label='P(h>h*)')
ax.axhline(HSTAR, color='gray', ls='--', lw=1)
ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in pairs])
ax.set_ylabel('S2 posterior mean'); ax2.set_ylabel('P(h>h*|S2)')
ax.set_title('Cycle-free (D_scale, D_noinfo) vs cyclic SPC vs pop half', fontsize=10)
ax.set_ylim(0, max(0.25, max(posts) * 1.4)); ax2.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_cycle_vs_nocycle.png'), dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.5, 4.0))
# sensitivity thresholds for scale / noinfo
sens = ['D_scale', 'D_scale_D0.5', 'D_scale_D1.0', 'D_noinfo_D0.5', 'D_noinfo', 'D_hiergov']
ax.plot(range(len(sens)), [R['denominators'][n]['S2']['post_mean'] for n in sens], 'o-', color='tab:green')
ax.axhline(HSTAR, color='k', ls='--')
ax.set_xticks(range(len(sens))); ax.set_xticklabels(sens, rotation=25, ha='right', fontsize=8)
ax.set_ylabel('S2 posterior mean'); ax.set_title('Sensitivity of cycle-free denominators', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_scale_noinfo_sensitivity.png'), dpi=130); plt.close(fig)

R['runtime_s'] = time.time() - T0

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    return o

json.dump(clean(R), open(os.path.join(HERE, 'results.json'), 'w'), ensure_ascii=False, indent=1)

# stdout
print('seed', SEED, 'hstar', HSTAR)
for name in R['main_summary']:
    s = R['main_summary'][name]
    print(f"{name}: eps={s['n_episodes']} S2={s['S2_k_n']} post={s['S2_post']:.4f} P(h>h*)={s['S2_P_gt_hstar']:.3f} A_Ord={s['S2_A_Ord']:.5f} share_heavy={s['S2_share_from_heavy']:.3f} S1={s['S1_post']:.4f}")
print('S2_clean', R['D_pop_S2_clean'])
print('excluded', R['S2_clean_excluded'])
for k, v in H.items(): print(k, v)
print('policy', R['policy'])
print('runtime', R['runtime_s'])
