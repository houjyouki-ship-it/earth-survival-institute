"""研究098：低緯度で 4 作物が失われる原因を光と雨で分ける（open_problems 237＋236）
入力：097 の計算（そのまま再実行）、NWDC の OSF（https://osf.io/e28gq/）の CESM の月の PRECC・PRECL（150 Tg nw_ur_150_07、対照 nw_cntrl_03）、
同 OSF の 2_aggregated の FSDS の国の月平均（nw_ur_150_07: https://osf.io/download/nre4t/ 、nw_cntrl_03: https://osf.io/download/698e263d53f25accc4dfc4b3/ ）、
CLM5.0 の技術文書の作物の章（https://escomp.github.io/CTSM/release-clm5.0/tech_note/Crop_Irrigation/CLM50_Tech_Note_Crop_Irrigation.html ）。
注：出す値は模型の中の割合で、生存確率でも復興確率でもない。乱数は 097 と同じ（3 群の引き、seed 96、同じ順）。"""
import os, json, numpy as np, pandas as pd, netCDF4 as nc
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); D97 = os.path.join(HERE, '../097-hardy-crop-nuclear-winter')
os.chdir(D97); src = open('sim.py').read(); exec(src[:src.index('RES = {}')])   # 097 の前半（096 の差0の assert を含む）
o_main, GB, GP, BOK, POK = run({})
gain = (SW == 1) & ((BOK | POK).all(0))
wl = seed_loss(Y[:5, CROPS['wheat']][:, ii, jj], SEED['wheat'], 0.5, 0.01, 5)
cons = np.where(gain & grown['wheat'][ii, jj] & wl, 1.0, SW*(1-gain))
os.chdir(HERE)
R = {'repro097': dict(gain_area=float(np.sum(wA*gain)), n_gain=int(gain.sum()), q_main=o_main['hardy_plus4_r0']['q'], q_cons=summ(cons)['q'])}
assert abs(R['repro097']['gain_area']-0.2210) < 1e-4 and abs(R['repro097']['q_main']-0.4244) < 1e-4 and abs(R['repro097']['q_cons']-0.5545) < 1e-4, R['repro097']

# ---- 雨：PRECC+PRECL（m/s）→ 月の mm ----
ND = '/workspace/nwdata/scratch/alpine/joco6825/dwinstead_data/'
def prec(run, years):
    P = []
    for y in years:
        for m in range(1, 13):
            d = nc.Dataset(f'{ND}{run}/{run}.cam.h0.{y:04d}-{m:02d}.nc'); P.append((d['PRECC'][0]+d['PRECL'][0]).astype(np.float32))
    return np.array(P).reshape(len(years), 12, 96, 144)*86400*1000*DAYS[None, :, None, None]
P150 = prec('nw_ur_150_07', range(5, 11)); PCT = prec('nw_cntrl_03', range(1, 22))
P150y, _, _ = crop_years(P150, P150, 5); PCTy, _, _ = crop_years(PCT, PCT, 20)
SEAS = slice(3, 7)    # 北半球 4〜7 月、南半球 10〜1 月（CLM5 の春小麦の播種の窓 4/1〜6/15 から 4 か月）
def ratios(A150, ACT, seas=False):
    a = A150[:, SEAS].sum(1) if seas else A150.sum(1); c = (ACT[:, SEAS].sum(1) if seas else ACT.sum(1))
    cm = c.mean(0); return a/np.where(cm > 0, cm, np.nan), c/np.where(cm > 0, cm, np.nan)
Pr, Pr_ctl = ratios(P150y, PCTy); Pr_s, Pr_s_ctl = ratios(P150y, PCTy, True)
Pr, Pr_ctl, Pr_s, Pr_s_ctl = [cells(x) for x in (Pr, Pr_ctl, Pr_s, Pr_s_ctl)]

# ---- 光：FSDS の国の月平均 → 格子 ----
ALIAS = {'Ivory Coast': "C??te d'Ivoire", 'Republic of the Congo': 'Republic of Congo', 'eSwatini': 'Swaziland', 'Palestine': 'Palestina',
         'North Macedonia': 'Macedonia', 'Somaliland': 'Somalia', 'Northern Cyprus': 'Cyprus'}
F150 = pd.read_csv('sources/fsds150.csv').set_index('Nation'); FCT = pd.read_csv('sources/fsdsctl.csv').set_index('Nation')
def fs_arr(df, y0, ny):
    cols = [f'{y:04d} - {m:02d}' for y in range(y0, y0+ny) for m in range(1, 13)]
    return df[cols].to_numpy(float).reshape(len(df), ny, 12)
f150 = fs_arr(F150, 5, 6); fct = fs_arr(FCT, 1, 21)
names = []
for f in gj['features']:
    p = f['properties']; nm = next((n for n in [p['ADMIN'], p['NAME'], p.get('NAME_LONG'), ALIAS.get(p['ADMIN'])] if n in F150.index), None); names.append(nm)
def fs_crop(arr, ny, issh):
    out = np.empty((arr.shape[0], ny, 12))
    for y in range(ny): out[:, y] = np.concatenate([arr[:, y, 6:], arr[:, y+1, :6]], 1) if issh else arr[:, y]
    return out
sh_c = lat[ii] < 0
Lr = np.full((5, len(ii)), np.nan); Lr_s = Lr.copy(); Lr_ctl = np.full((20, len(ii)), np.nan); Lr_s_ctl = Lr_ctl.copy()
for k in range(len(ii)):
    nm = names[ccell[k]]
    if nm is None: continue
    r150 = F150.index.get_loc(nm); rct = FCT.index.get_loc(nm)
    a = fs_crop(f150[[r150]], 5, sh_c[k])[0]; c = fs_crop(fct[[rct]], 20, sh_c[k])[0]
    cm, cms = c.mean(), c[:, SEAS].mean()
    Lr[:, k] = a.mean(1)/cm; Lr_s[:, k] = a[:, SEAS].mean(1)/cms; Lr_ctl[:, k] = c.mean(1)/cm; Lr_s_ctl[:, k] = c[:, SEAS].mean(1)/cms
R['light_match_area'] = float(np.sum(wA*np.isfinite(Lr[0]))); R['light_match_gain_area'] = float(np.sum(wA*(gain & np.isfinite(Lr[0]))))/float(np.sum(wA*gain))

# ---- 得の格子の分類 ----
wg = wA*gain; wg = wg/wg.sum()
def classify(thr=0.5, seas=False):
    P = Pr_s if seas else Pr; Lx = Lr_s if seas else Lr
    rain = P < thr; light = np.nan_to_num(Lx, nan=9) < thr
    out = {}
    for y in range(5):
        out[f'year{y+1}'] = dict(rain_only=float(np.sum(wg*(rain[y] & ~light[y]))), light_only=float(np.sum(wg*(light[y] & ~rain[y]))),
                                 both=float(np.sum(wg*(rain[y] & light[y]))), neither=float(np.sum(wg*(~rain[y] & ~light[y]))))
    anyr = rain.any(0); anyl = light.any(0)
    out['any_year'] = dict(rain_only=float(np.sum(wg*(anyr & ~anyl))), light_only=float(np.sum(wg*(anyl & ~anyr))), both=float(np.sum(wg*(anyr & anyl))), neither=float(np.sum(wg*(~anyr & ~anyl))))
    return out, rain, light
CL, RAIN, LIGHT = classify()
R['class_main'] = CL
R['H110a'] = dict(frac_gain_worst_year_rain_lt05=float(np.sum(wg*(Pr.min(0) < 0.5))), median_worst_Pr=float(np.median(Pr.min(0)[gain])),
                  median_Pr_by_year=[float(np.median(Pr[y][gain])) for y in range(5)])
R['H110b'] = dict(frac_gain_year2_light_lt05=float(np.sum(wg*(np.nan_to_num(Lr[1], nan=9) < 0.5))),
                  median_Lr_by_year=[float(np.nanmedian(Lr[y][gain])) for y in range(5)], min_Lr_gain=float(np.nanmin(Lr[:, gain])))

# ---- q_phys：雨か光が閾値未満の年は耐寒の作物も実らないとする ----
def q_phys(thr=0.5, seas=False, w=wA, TT=TA, rule='or'):
    P = Pr_s if seas else Pr; Lx = Lr_s if seas else Lr
    bad = (P < thr) | (np.nan_to_num(Lx, nan=9) < thr) if rule == 'or' else (P < thr) if rule == 'rain' else (np.nan_to_num(Lx, nan=9) < thr)
    ok = (BOK | POK) & ~bad
    hl = seed_loss(np.where(ok, 10.0, 0.0), 1.0, 0.5, 0.01, 5, 0.0)
    return summ((switch_lost(Y) & hl).astype(float), w, TT)
R['qphys'] = {f'thr{t}': q_phys(t) for t in [0.3, 0.5, 0.7]}
R['qphys']['season_thr0.5'] = q_phys(0.5, True)
R['qphys']['rain_only_thr0.5'] = q_phys(0.5, rule='rain'); R['qphys']['light_only_thr0.5'] = q_phys(0.5, rule='light')
R['qphys']['popw_thr0.5'] = q_phys(0.5, w=wP, TT=TP)
R['H110c'] = dict(q_phys=R['qphys']['thr0.5']['q'], midpoint=(0.4244+0.5545)/2)
R['H110d'] = dict(neither_any_year=CL['any_year']['neither'])
# 平時の偽陽性：対照の作物年で同じ閾値に掛かる割合（得の格子、年あたり）
R['ctrl_false'] = {f'thr{t}': dict(rain=float(np.mean([np.sum(wg*(Pr_ctl[y] < t)) for y in range(20)])),
                                   light=float(np.mean([np.sum(wg*(np.nan_to_num(Lr_ctl[y], nan=9) < t)) for y in range(20)])),
                                   rain_season=float(np.mean([np.sum(wg*(Pr_s_ctl[y] < t)) for y in range(20)]))) for t in [0.3, 0.5, 0.7]}

# ---- 探索（登録外）：CLM5 の春小麦の「日数切れ」 ----
# 【事実：CLM5 文書】春小麦の GDDmat = GDD0（20 年の移動平均、≤1700）、粒の充填は GDDmat の 60% で始まり、播種から 150 日で収穫。
# 播種の要件を満たさなくても 6/15（南半球は 12/15）に GDD8>0 なら播く。粒の充填の前に日数切れなら粒は 0（相の 2 では a_repr=0）。
# 近似：播種を窓の始め（4/1、南は 10/1）とし、150 日 ≈ 4〜8 月の 5 か月（index 3..7）の GDD0 が 0.6×GDDmat（対照の 4〜9 月の GDD0、≤1700）に届くか。
g0 = lambda T: DY*np.clip(T, 0, None)
gdd_150 = cells(g0(T150)[:, 3:8].sum(1)); gddmat = np.minimum(1700, cells(g0(TCT)[:, 3:9].sum(1)).mean(0))
cut = gdd_150 < 0.6*gddmat; cut_late = cells(g0(T150)[:, 5:10].sum(1)) < 0.6*gddmat   # 6/15 播種の版
wgw = wg*grown['wheat'][ii, jj]; wgw = wgw/wgw.sum()
R['explore_clm5_cutoff'] = dict(frac_gain_wheat_cut_any_year=float(np.sum(wgw*cut.any(0))), frac_cut_by_year=[float(np.sum(wgw*c)) for c in cut],
    frac_cut_late_any=float(np.sum(wgw*cut_late.any(0))), median_gddmat=float(np.median(gddmat[gain])), median_gdd150_by_year=[float(np.median(g[gain])) for g in gdd_150],
    frac_cut_not_rain_not_light_any=float(np.sum(wgw*(cut.any(0) & ~RAIN.any(0) & ~LIGHT.any(0)))))

# ---- 探索（登録外）：閾値でなく比例で減らす版。耐寒の作物の増殖比 10 に m を掛ける ----
def q_prop(kind, w=wA, TT=TA, gmult=10.0):
    L_ = np.minimum(1, np.nan_to_num(Lr, nan=1)); P_ = np.minimum(1, Pr)
    m = L_*P_ if kind == 'product' else np.minimum(L_, P_) if kind == 'liebig' else L_ if kind == 'light' else P_
    hl = seed_loss(np.where(BOK | POK, gmult*m, 0.0), 1.0, 0.5, 0.01, 5, 0.0)
    return summ((switch_lost(Y) & hl).astype(float), w, TT)
R['explore_prop'] = {k: q_prop(k) for k in ['product', 'liebig', 'light', 'rain']}
R['explore_prop']['product_gmult20'] = q_prop('product', gmult=20.0)
R['explore_prop']['liebig_gmult5'] = q_prop('liebig', gmult=5.0)

# 記述：得の格子の 150 Tg の絶対の雨（mm、作物年と 4 か月）。次回の絶対の閾値の準備
Pabs = cells(P150y.sum(1)); Pabs_s = cells(P150y[:, SEAS].sum(1)); Pabs_ctl = cells(PCTy.sum(1)).mean(0)
R['desc_abs_precip'] = dict(median_ctrl_mm=float(np.median(Pabs_ctl[gain])), median_worst_mm=float(np.median(Pabs.min(0)[gain])),
    median_worst_season_mm=float(np.median(Pabs_s.min(0)[gain])), q25_worst_mm=float(np.percentile(Pabs.min(0)[gain], 25)),
    frac_gain_worst_lt300mm=float(np.sum(wg*(Pabs.min(0) < 300))), frac_gain_worst_season_lt150mm=float(np.sum(wg*(Pabs_s.min(0) < 150))))
# 記述：得の格子の緯度帯ごとの雨の比の中央値（最悪年）
R['desc_lat_bands'] = {f'{a}..{a+15}': dict(n=int(np.sum(gain & (lat[ii] >= a) & (lat[ii] < a+15))),
    med_worst_Pr=float(np.median(Pr.min(0)[gain & (lat[ii] >= a) & (lat[ii] < a+15)])) if np.any(gain & (lat[ii] >= a) & (lat[ii] < a+15)) else None,
    med_y2_Lr=float(np.nanmedian(Lr[1][gain & (lat[ii] >= a) & (lat[ii] < a+15)])) if np.any(gain & (lat[ii] >= a) & (lat[ii] < a+15)) else None) for a in range(-45, 45, 15)}
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in R.items(): print(k, json.dumps(v, ensure_ascii=False))

# ---- 図 ----
lo = np.where(lon > 180, lon-360, lon); o_ = np.argsort(lo)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for a_, arr, ttl, vm in [(ax[0], Pr.min(0), 'worst-year precip ratio 150Tg/ctrl (gain cells)', (0, 1.2)), (ax[1], Lr[1], 'year-2 FSDS ratio (country, gain cells)', (0, 1))]:
    g = np.full(LAT.shape, np.nan); g[ii[gain], jj[gain]] = arr[gain]
    bg = np.full(LAT.shape, np.nan); bg[ii, jj] = 1
    a_.pcolormesh(lo[o_], lat, bg[:, o_], cmap='Greys', vmin=0, vmax=4, shading='auto')
    im = a_.pcolormesh(lo[o_], lat, g[:, o_], cmap='RdYlBu', vmin=vm[0], vmax=vm[1], shading='auto'); a_.set_title(ttl, fontsize=9); plt.colorbar(im, ax=a_)
labs = ['097 main', 'thr0.7', 'thr0.5', 'thr0.3', 'season0.5', 'rain only', 'light only', '097 cons']
vals = [0.4244, R['qphys']['thr0.7']['q'], R['qphys']['thr0.5']['q'], R['qphys']['thr0.3']['q'], R['qphys']['season_thr0.5']['q'],
        R['qphys']['rain_only_thr0.5']['q'], R['qphys']['light_only_thr0.5']['q'], 0.5545]
ax[2].bar(range(len(labs)), vals, color=['gray']+['C0']*6+['gray']); ax[2].axhline(R['H110c']['midpoint'], color='k', ls='--', lw=0.8, label='midpoint 0.489')
ax[2].set_xticks(range(len(labs)), labs, rotation=45, fontsize=8); ax[2].set_ylabel('per-group staple loss q'); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_light_rain.png', dpi=110)
