"""研究097：核の冬で耐寒・早生の作物（大麦・ジャガイモ）が実るか（open_problems 235）
入力：NWDC の OSF（https://osf.io/e28gq/）の CESM の月の地表温度 TS と月の最低 TSMN（150 Tg nw_ur_150_07 と対照 nw_cntrl_03）、
Xia 2022 の OSF（https://osf.io/yrbse/）の CLM5 の格子の収量（095 の写し）、NE 110m の国境（095 の写し）。
作物の閾値【事実】：大麦の基準温度 0 °C（NDAWN、Alberta 農業気候図）、早生の大麦の成熟 1,300 GDD・早刈り 1,150・晩生 1,400〜1,500（北の周縁地域の穀物の手引き、Iceland）、
ジャガイモの基準温度 7 °C（Alberta、Manitoba）、Manitoba の P-days：出芽まで 155（Russet Burbank、植え付けから）、塊茎の肥大の始まり 236〜260、最大の肥大 420〜460、成熟 850（出芽から）。
注：出す値は模型の中の割合で、生存確率でも復興確率でもない。乱数は Monte Carlo（3 群の引き）のみ、096 と同じ seed 96 と同じ順で引く。"""
import json, numpy as np, netCDF4 as nc
from matplotlib.path import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(96)
S5 = '../095-farming-halt-correlation/sources/'
L = lambda f: np.ma.filled(nc.Dataset(S5+f)['yield'][:].astype(float), np.nan)
Y = L('nw150.nc'); C = np.concatenate([L('ctrl03.nc'), L('ctrl03m02.nc')])
lat = np.asarray(nc.Dataset(S5+'nw150.nc')['lat'][:]); lon = np.asarray(nc.Dataset(S5+'nw150.nc')['lon'][:])
CROPS = {'maize': 1, 'rice': 4, 'soy': 7, 'wheat': 10}
SEED = dict(maize=0.015, rice=0.05, soy=0.09, wheat=0.125)
gj = json.load(open(S5+'ne_110m_admin_0_countries.geojson'))
LON, LAT = np.meshgrid(np.where(lon > 180, lon-360, lon), lat)
pts = np.c_[LON.ravel(), LAT.ravel()]
cid = -np.ones(pts.shape[0], int); subreg = []; pops = []
for i, f in enumerate(gj['features']):
    p = f['properties']; subreg.append(p.get('SUBREGION')); pops.append(p.get('POP_EST'))
    g = f['geometry']; polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    for poly in polys:
        cid[Path(np.array(poly[0])).contains_points(pts) & (cid < 0)] = i
cid = cid.reshape(LAT.shape); W = np.cos(np.deg2rad(LAT))
CM = {k: np.nanmean(C[:, j], 0) for k, j in CROPS.items()}
grown = {k: np.isfinite(CM[k]) & (CM[k] > 0.1) for k in CROPS}
with np.errstate(all='ignore'):
    RAT = np.nanmean(np.stack([np.where(grown[k], np.clip(Y[:, j]/CM[k], 0, 1.5), np.nan) for k, j in CROPS.items()]), 0)
valid = np.all(np.isfinite(RAT[:5]), 0) & (cid >= 0)
ii, jj = np.where(valid); ccell = cid[ii, jj]; wA = W[ii, jj]/W[ii, jj].sum()
nc_ = {c: np.sum(ccell == c) for c in set(ccell)}
wP = np.array([(pops[c] or 0)/nc_[c] for c in ccell], float); wP /= wP.sum()
R = {'n_cells': int(valid.sum())}

def seed_loss(yld, s, f, xmin, T, reserve=0.0):
    Sx = np.ones(yld.shape[1:]); lost = np.zeros(yld.shape[1:], bool); Rv = np.full(yld.shape[1:], reserve)
    for t in range(T):
        H = f*Sx*np.nan_to_num(yld[t]/s, nan=0.0)
        surplus = np.maximum(H-1.0, 0); Rv = np.minimum(reserve, Rv+surplus)
        Sx = np.minimum(1.0, H); top = np.minimum(Rv, 1.0-Sx); Sx = Sx+top; Rv = Rv-top
        lost |= Sx < xmin
    return lost
def switch_lost(YY, reserve=0.0, T=5):
    Lk = np.stack([np.where(grown[k][ii, jj], seed_loss(YY[:T, j][:, ii, jj], SEED[k], 0.5, 0.01, T, reserve), True) for k, j in CROPS.items()])
    return Lk.all(0)
# 096 の 3 群（同じ乱数の順）
by_c = {c: np.where(ccell == c)[0] for c in set(ccell)}; by_r = {}
for c in by_c: by_r.setdefault(subreg[c], []).append(c)
def triples(w, N=60000):
    T = {m: np.empty((N, 3), int) for m in ['country', 'region', 'world']}
    for m in T:
        k = 0
        while k < N:
            a = rng.choice(len(ii), p=w); c0 = ccell[a]
            if m == 'world': T[m][k] = [a] + list(rng.choice(len(ii), 2, p=w)); k += 1
            elif m == 'country': T[m][k] = [a] + list(rng.choice(by_c[c0], 2)); k += 1
            else:
                cs = [c for c in by_r[subreg[c0]] if c != c0]
                if len(cs) < 2: continue
                c1, c2 = rng.choice(cs, 2, replace=False); T[m][k] = [a, rng.choice(by_c[c1]), rng.choice(by_c[c2])]; k += 1
    return T
TA = triples(wA); TP = triples(wP)
def summ(x, w=wA, TT=TA):
    o = dict(q=float(np.sum(w*x)))
    for m, T in TT.items(): o['PG3_'+m] = float(np.mean(np.prod(x[T], 1)))
    return o
SW = switch_lost(Y).astype(float); SW1 = switch_lost(Y, 1.0).astype(float)
R['check096'] = dict(switch=summ(SW), switch_reserve1=summ(SW1))
assert abs(R['check096']['switch']['q']-0.6454) < 1e-4 and abs(R['check096']['switch']['PG3_region']-0.3729) < 1e-4, R['check096']

# ---- 気温：作物の年ごとの GDD ----
z = np.load('sources/temps.npz')
DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], float)
def crop_years(ts, tn, ny):
    """ts: (years,12,lat,lon) °C。北半球は暦年、南半球は 7 月〜翌 6 月。返り値 (ny,12,lat,lon) と月の日数。"""
    out = np.empty((ny, 12)+ts.shape[2:], np.float32); outn = np.empty_like(out)
    sh = (lat < 0)[None, :, None]
    for y in range(ny):
        nh_t, nh_n = ts[y], tn[y]
        sh_t = np.concatenate([ts[y, 6:], ts[y+1, :6]]); sh_n = np.concatenate([tn[y, 6:], tn[y+1, :6]])
        out[y] = np.where(sh, sh_t, nh_t); outn[y] = np.where(sh, sh_n, nh_n)
    dsh = np.r_[DAYS[6:], DAYS[:6]]
    days = np.where(sh[..., None].transpose(0, 2, 1, 2)[0] if False else (lat < 0)[None, :, None, None].transpose(3, 0, 1, 2), dsh[:, None, None], DAYS[:, None, None])
    return out, outn, days
def season_gdd(T, TN, days, Tb, Tsow, tn_min=None):
    """各年：月平均 ≥ Tsow（かつ TSMN ≥ tn_min）の月の最長の連続（年の中、12 か月すべてなら全部）で Σ days×(T−Tb)+ 。"""
    ok = T >= Tsow
    if tn_min is not None: ok &= TN >= tn_min
    g = days*np.clip(T-Tb, 0, None)
    best = np.zeros(T.shape[:1]+T.shape[2:]); cur = np.zeros_like(best)
    for m in range(12):
        cur = np.where(ok[:, m], cur+g[:, m], 0); best = np.maximum(best, cur)
    return best
K = 273.15
T150, N150, DY = crop_years(z['ts150']-K, z['tn150']-K, 5)        # 作物年 5〜9（096 の Y[:5] に合わせる）
TCT, NCT, _ = crop_years(z['tsctl']-K, z['tnctl']-K, 20)
def cells(a): return a[:, ii, jj]
VER = {'main': dict(bar_gdd=1300, bar_sow=5.0, pot_gdd=400, pot_sow=7.0, tn_min=None),
       'barley1150': dict(bar_gdd=1150), 'barley1450': dict(bar_gdd=1450),
       'potato600': dict(pot_gdd=600), 'potato1000': dict(pot_gdd=1000), 'potato250': dict(pot_gdd=250),
       'sow3': dict(bar_sow=3.0), 'sow8': dict(bar_sow=8.0, pot_sow=10.0),
       'frost_m2': dict(tn_min=-2.0), 'frost_m5': dict(tn_min=-5.0),
       'cold_bias_p1': dict(dT=+1.0), 'cold_bias_m1': dict(dT=-1.0), 'cold_bias_p2': dict(dT=+2.0), 'cold_bias_m3': dict(dT=-3.0)}
def run(v, TT=T150, NN=N150, YY=Y, ctrl=False):
    p = dict(VER['main']); p.update(v); dT = p.pop('dT', 0.0)
    gb = cells(season_gdd(TT+dT, NN+dT, DY, 0.0, p['bar_sow'], p['tn_min']))
    gp = cells(season_gdd(TT+dT, NN+dT, DY, 7.0, p['pot_sow'], p['tn_min']))
    bok = gb >= p['bar_gdd']; pok = gp >= p['pot_gdd']
    out = {}
    sw = switch_lost(YY) if not ctrl else None
    for nm, okk in [('barley', bok), ('potato', pok), ('hardy', bok | pok)]:
        yy = np.where(okk, 10.0, 0.0)                       # 実れば増殖比 10【推測】、実らなければ 0
        for res in [0.0, 1.0]:
            hl = seed_loss(yy, 1.0, 0.5, 0.01, okk.shape[0], res)
            out[f'{nm}_alone_r{int(res)}'] = summ(hl.astype(float))
            if not ctrl:
                swr = switch_lost(YY, res)
                out[f'{nm}_plus4_r{int(res)}'] = summ((swr & hl).astype(float))
    out['frac_area_barley_ok_by_year'] = [float(np.sum(wA*b)) for b in bok]
    out['frac_area_potato_ok_by_year'] = [float(np.sum(wA*b)) for b in pok]
    return out, gb, gp, bok, pok
RES = {}
for tag, v in VER.items():
    o, gb, gp, bok, pok = run(v if tag != 'main' else {}); RES[tag] = o
    if tag == 'main': GB, GP, BOK, POK = gb, gp, bok, pok
R['versions'] = RES
R['main_popw'] = {nm: summ((switch_lost(Y, r) & seed_loss(np.where(BOK | POK, 10., 0.), 1., .5, .01, 5, r)).astype(float), wP, TP) for nm, r in [('hardy_plus4_r0', 0.), ('hardy_plus4_r1', 1.)]}
# 平時（対照）の偽陽性：対照の作物年 1〜5 と 6〜10 と 11〜15
R['ctrl'] = {}
for s0 in [0, 5, 10]:
    o, *_ = run({}, TCT[s0:s0+5], NCT[s0:s0+5], ctrl=True); R['ctrl'][f'yrs{s0+1}-{s0+5}'] = o
# H109a：北緯 40° 以北の有効な格子で、大麦が成熟に届く面積の割合（年ごと、最悪年、各年すべて）
n40 = lat[ii] >= 40; w40 = wA[n40]/wA[n40].sum()
fy = [float(np.sum(w40*b[n40])) for b in BOK]
R['H109a'] = dict(n_cells=int(n40.sum()), frac_by_year=fy, worst_year=float(min(fy)), worst_year_index=int(np.argmin(fy)),
                  year2=fy[1], all5=float(np.sum(w40*BOK.all(0)[n40])),
                  potato_by_year=[float(np.sum(w40*b[n40])) for b in POK])
# 較正の目安：対照で Iceland 付近（64N, 338-340E の陸）の大麦 GDD、平時の北緯 40° 以北の大麦の成熟の面積
gbc = season_gdd(TCT, NCT, _ if False else DY, 0.0, 5.0)
li = np.argmin(abs(lat-65.4))   # 63.5N の格子は海（TS が海面水温で年中 5 °C 以上）なので陸の 65.4N を使う
# 【事実】Reykjavík 1991–2020 の日平均（Wikipedia の気候表、出典 Icelandic Met Office）：5〜9 月 6.7, 9.8, 11.6, 11.0, 8.5 °C → 月 ≥5 °C の GDD0
obs_rvk = 31*6.7+30*9.8+31*11.6+31*11.0+30*8.5
R['calib'] = {'iceland_land_ctrl_gdd0_by_lon': {float(lon[k]): float(np.median(gbc[:, li, k])) for k in range(135, 138)}, 'reykjavik_obs_gdd0_season': obs_rvk,
    'ctrl_frac_north40_barley_ok': float(np.sum(w40*(np.median(gbc[:, ii, jj], 0) >= 1300)[n40])),
    'ctrl_median_gdd0_north40': float(np.median(np.median(gbc[:, ii, jj], 0)[n40])),
    'w150_median_gdd0_north40_by_year': [float(np.median(g[n40])) for g in GB]}
# 診断：096 の switch で失い、耐寒の作物で保つ格子（得の出どころ）
gain = (SW == 1) & ((BOK | POK).all(0))
wl = seed_loss(Y[:5, CROPS['wheat']][:, ii, jj], SEED['wheat'], 0.5, 0.01, 5)
R['gain_diag'] = dict(area=float(np.sum(wA*gain)), n=int(gain.sum()), lat_median=float(np.median(lat[ii][gain])),
    frac_abs_lat_lt30=float(np.mean(abs(lat[ii][gain]) < 30)), frac_wheat_grown=float(np.mean(grown['wheat'][ii, jj][gain])),
    frac_wheat_grown_and_lost=float(np.mean((grown['wheat'][ii, jj] & wl)[gain])),
    median_min_gdd0=float(np.median(GB.min(0)[gain])))
# 保守の版：CLM5 の小麦（寒さに強い側の作物）も種の収支で失う格子では、失う原因は気温以外（光・雨・CLM5 の 0）とみて耐寒の作物の得を数えない
cons = np.where(gain & grown['wheat'][ii, jj] & wl, 1.0, SW*(1-gain))
R['conservative'] = dict(main=summ(cons), popw=summ(cons, wP, TP), gain_area_kept=float(np.sum(wA*(gain & ~(grown['wheat'][ii, jj] & wl)))))
# 記述：実る格子（全 5 年）の緯度
ok5 = (BOK | POK).all(0)
R['desc_lat_hist_ok5'] = {f'{a}..{a+15}': [int(np.sum((lat[ii] >= a) & (lat[ii] < a+15))), int(np.sum(ok5 & (lat[ii] >= a) & (lat[ii] < a+15)))] for a in range(-60, 75, 15)}
R['ref096'] = dict(switch=0.6454, hardy_d03=0.1344, PG3_region_switch=0.3729, PG3_region_d03=0.0107, reserve1_switch=0.5133, reserve1_d03=0.09)
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
print('cells', R['n_cells'], 'check096', json.dumps(R['check096']))
print('H109a', json.dumps(R['H109a']))
print('calib', json.dumps(R['calib']))
for t, o in RES.items():
    print(t, json.dumps({k: (({kk: round(vv, 4) for kk, vv in d.items()}) if isinstance(d, dict) else [round(x, 3) for x in d]) for k, d in o.items()}))
print('popw', json.dumps(R['main_popw']))
for t, o in R['ctrl'].items(): print('CTRL', t, json.dumps({k: (({kk: round(vv, 4) for kk, vv in d.items()}) if isinstance(d, dict) else [round(x, 3) for x in d]) for k, d in o.items()}))
print('gain', json.dumps(R['gain_diag']))
print('conservative', json.dumps(R['conservative']))
print('lat hist [n, ok5]', R['desc_lat_hist_ok5'])
# 図
lo = np.where(lon > 180, lon-360, lon); o_ = np.argsort(lo)
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for a_, arr, ttl in [(ax[0], GB.min(0), 'min over years 1-5 of barley GDD0 (150 Tg)'), (ax[1], GP.min(0), 'min over years of potato GDD7 (150 Tg)')]:
    g = np.full(LAT.shape, np.nan); g[ii, jj] = arr
    im = a_.pcolormesh(lo[o_], lat, g[:, o_], cmap='viridis', vmin=0, vmax=3000 if a_ is ax[0] else 1500, shading='auto'); a_.set_title(ttl, fontsize=9); plt.colorbar(im, ax=a_)
tags = list(VER); xs = np.arange(len(tags))
for k, mk in [('hardy_plus4_r0', 'o'), ('hardy_plus4_r1', 's'), ('barley_plus4_r0', '^'), ('potato_plus4_r0', 'v')]:
    ax[2].plot(xs, [RES[t][k]['q'] for t in tags], mk+'-', label=k)
ax[2].axhline(0.6454, color='k', ls='--', lw=0.8, label='096 switch'); ax[2].axhline(0.1344, color='gray', ls=':', label='096 d=0.3')
ax[2].set_xticks(xs, tags, rotation=60, fontsize=7); ax[2].set_ylabel('per-group staple loss q'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('fig_hardy_crops.png', dpi=110)
