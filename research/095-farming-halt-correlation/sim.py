"""研究095：重い崩壊で農耕が同時に止まる範囲の相関（open_problems 232）
入力：Xia et al. 2022 の OSF（https://osf.io/yrbse/）の CLM5 の格子の収量（150 Tg と対照2本）、
NE 110m の国境（072 の写し）。094 の P(空白 > L_store)=0.745 を主の版で使う。
注：出す値は模型の中の割合で、生存確率でも復興確率でもない。乱数は Monte Carlo のみ（seed 95）。"""
import json, numpy as np, netCDF4 as nc
from matplotlib.path import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(95)
S = 'sources/'
L = lambda f: np.ma.filled(nc.Dataset(S+f)['yield'][:].astype(float), np.nan)
Y = L('nw150.nc'); C = np.concatenate([L('ctrl03.nc'), L('ctrl03m02.nc')])
lat = np.asarray(nc.Dataset(S+'nw150.nc')['lat'][:]); lon = np.asarray(nc.Dataset(S+'nw150.nc')['lon'][:])
NY = Y.shape[0]
R = {}
def ratio_cells(crops):
    rs = []
    for k in crops:
        cm = np.nanmean(C[:, k], 0)
        with np.errstate(all='ignore'):
            r = np.where(cm > 0.1, np.clip(Y[:, k] / cm, 0, 1.5), np.nan)
        rs.append(r)
    with np.errstate(all='ignore'):
        return np.nanmean(np.stack(rs), 0)  # (year, lat, lon)
RAIN = ratio_cells([1, 4, 7, 10]); IRR = ratio_cells([2, 5, 8, 11])
# 国の割り当て
gj = json.load(open(S+'ne_110m_admin_0_countries.geojson'))
LON, LAT = np.meshgrid(np.where(lon > 180, lon-360, lon), lat)
pts = np.c_[LON.ravel(), LAT.ravel()]
cid = -np.ones(pts.shape[0], int); names = []; subreg = []; pops = []
for i, f in enumerate(gj['features']):
    p = f['properties']; names.append(p.get('ADMIN') or p.get('NAME')); subreg.append(p.get('SUBREGION')); pops.append(p.get('POP_EST'))
    g = f['geometry']; polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
    for poly in polys:
        inside = Path(np.array(poly[0])).contains_points(pts)
        cid[inside & (cid < 0)] = i
cid = cid.reshape(LAT.shape)
W = np.cos(np.deg2rad(LAT))

def analyse(RT, tag, popw=False, sh=False):
    out = {}
    valid = np.all(np.isfinite(RT[:5]), 0) & (cid >= 0)
    D = 1 - RT  # 減り
    out['n_cells'] = int(valid.sum()); out['n_countries'] = int(len(set(cid[valid])))
    # 国の減り（面積の重みの平均、年2=index1 と最悪年）
    cty = {}
    for c in sorted(set(cid[valid])):
        m = valid & (cid == c); w = W[m]
        d_y2 = float(np.sum(D[1][m]*w)/w.sum()); d_pk = float(max(np.sum(D[t][m]*w)/w.sum() for t in range(5)))
        clat = float(np.sum(LAT[m]*w)/w.sum())
        cty[names[c]] = dict(D_year2=d_y2, D_peak=d_pk, lat=clat, subregion=subreg[c], n=int(m.sum()))
    out['countries'] = cty
    dy2 = np.array([v['D_year2'] for v in cty.values()]); dpk = np.array([v['D_peak'] for v in cty.values()]); cl = np.array([v['lat'] for v in cty.values()])
    out['H107a_frac_D_gt_0.9_year2'] = float(np.mean(dy2 > 0.9)); out['H107a_frac_D_gt_0.9_peak'] = float(np.mean(dpk > 0.9))
    out['frac_D_gt_0.75_year2'] = float(np.mean(dy2 > 0.75)); out['frac_D_gt_0.5_year2'] = float(np.mean(dy2 > 0.5))
    north, south = dy2[cl > 30], dy2[cl < 0]
    out['H107b_median_north30'] = float(np.median(north)); out['H107b_median_south'] = float(np.median(south))
    out['H107b_diff_pts'] = float(100*(np.median(north)-np.median(south)))
    A = np.c_[np.ones_like(cl), np.abs(cl), cl, cl**2]
    res = np.linalg.lstsq(A, dy2, rcond=None)[0]; pred = A @ res
    out['H107b_R2_lat'] = float(1 - np.sum((dy2-pred)**2)/np.sum((dy2-dy2.mean())**2))
    # 群の置き方
    if sh: valid = valid & (LAT < 0)
    ii, jj = np.where(valid); ccell = cid[ii, jj]
    if popw:
        ncell = {c: np.sum(ccell == c) for c in set(ccell)}
        wcell = np.array([(pops[c] or 0)/ncell[c] for c in ccell], float)
    else:
        wcell = W[ii, jj].copy()
    wcell = wcell/wcell.sum()
    Dcell = D[:5, ii, jj].T  # (cell, 5 年)
    Dmax = Dcell.max(1)
    by_c = {c: np.where(ccell == c)[0] for c in set(ccell)}
    by_r = {}
    for c in by_c: by_r.setdefault(subreg[c], []).append(c)
    N = 60000
    def sample(mode):
        T = np.empty((N, 3), int); k = 0
        while k < N:
            a = rng.choice(len(ii), p=wcell); c0 = ccell[a]
            if mode == 'world':
                T[k] = [a] + list(rng.choice(len(ii), 2, p=wcell)); k += 1
            elif mode == 'country':
                cand = by_c[c0]; T[k] = [a] + list(rng.choice(cand, 2)); k += 1
            else:
                cs = [c for c in by_r[subreg[c0]] if c != c0]
                if len(cs) < 2: continue
                c1, c2 = rng.choice(cs, 2, replace=False)
                T[k] = [a, rng.choice(by_c[c1]), rng.choice(by_c[c2])]; k += 1
        return T
    trip = {m: sample(m) for m in ['country', 'region', 'world']}
    logit = lambda x: np.log(np.clip(x, 1e-6, 1-1e-6)/(1-np.clip(x, 1e-6, 1-1e-6)))
    P_after = 0.745  # 094：空白の中央値 3 年で P(空白>L_store)
    grid = {}
    for Ds in [0.75, 0.9, 0.95]:
        for sg in [0.5, 1.0]:
            key = f'Dstar{Ds}_sg{sg}'; g = {}
            for m, T in trip.items():
                eps = rng.normal(0, sg, T.shape)
                thr = logit(Ds) + eps
                halt = logit(Dmax[T]) > thr                          # 主：止まる→094 の 0.745
                # データの版：減りが閾値を越える年が 2 年以上続くと主食の種が尽きる（L_store 1.3 年）
                over = logit(Dcell[T]) > thr[..., None]
                run = np.zeros(T.shape, int); cur = np.zeros(T.shape, int)
                for t in range(5):
                    cur = np.where(over[..., t], cur+1, 0); run = np.maximum(run, cur)
                lossD = run >= 2
                q = halt.mean(); pl = q*P_after
                PG = float(np.mean(halt.all(1))*P_after**3)       # 0.745 は群ごとに独立
                ql = lossD.mean(); PGd = float(np.mean(lossD.all(1)))
                pos = lambda P, p: float((np.log(P)-3*np.log(p))/(np.log(p)-3*np.log(p))) if P > 0 and p > 0 else None
                g[m] = dict(q_halt=float(q), P_loss=float(pl), P_G3=PG, indep=float(pl**3), full=float(pl), pos=pos(PG, pl),
                            q_lossD=float(ql), P_G3_D=PGd, pos_D=pos(PGd, ql) if PGd > 0 else None)
            grid[key] = g
    out['triples'] = grid
    # 距離ごとの φ（主 D*=0.9、群のばらつきなし）
    h = (Dmax > 0.9).astype(float); p = h.mean()
    x = np.deg2rad(LAT[ii, jj]); y = np.deg2rad(LON[ii, jj])
    a = rng.choice(len(ii), 40000, p=wcell); b = rng.choice(len(ii), 40000, p=wcell)
    dist = 6371*np.arccos(np.clip(np.sin(x[a])*np.sin(x[b])+np.cos(x[a])*np.cos(x[b])*np.cos(y[a]-y[b]), -1, 1))
    bins = [0, 250, 500, 1000, 2000, 4000, 8000, 20100]; phi = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (dist >= lo) & (dist < hi); ha, hb = h[a][s], h[b][s]
        pa, pb = ha.mean(), hb.mean()
        phi.append(dict(lo=lo, hi=hi, n=int(s.sum()), phi=float((np.mean(ha*hb)-pa*pb)/np.sqrt(pa*(1-pa)*pb*(1-pb))) if 0 < pa < 1 and 0 < pb < 1 else None))
    out['phi_by_distance_Dstar0.9'] = phi; out['p_halt_cell_Dstar0.9'] = float(np.sum(wcell*h))
    return out, Dmax, ii, jj
R['rain'], Dmax, ii, jj = analyse(RAIN, 'rain')
R['irr'], _, _, _ = analyse(IRR, 'irr')
R['rain_popw'], _, _, _ = analyse(RAIN, 'rain', popw=True)
R['rain_south_popw'], _, _, _ = analyse(RAIN, 'rain', popw=True, sh=True)
# 094 との比較の基準
R['ref094'] = dict(indep_G3=0.0044, full=0.164, log_mid=float(np.sqrt(0.0044*0.164)))
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
r = R['rain']; m = r['triples']['Dstar0.9_sg0.5']
print('cells', r['n_cells'], 'countries', r['n_countries'])
print('H107a frac D>0.9 year2 %.3f peak %.3f ; >0.75 y2 %.3f ; >0.5 y2 %.3f' % (r['H107a_frac_D_gt_0.9_year2'], r['H107a_frac_D_gt_0.9_peak'], r['frac_D_gt_0.75_year2'], r['frac_D_gt_0.5_year2']))
print('H107b north30 %.3f south %.3f diff %.1f pts R2 %.2f' % (r['H107b_median_north30'], r['H107b_median_south'], r['H107b_diff_pts'], r['H107b_R2_lat']))
for k, g in r['triples'].items():
    print(k, ' | '.join(f"{mm}: q {v['q_halt']:.3f} PG {v['P_G3']:.4f} ind {v['indep']:.5f} pos {v['pos'] if v['pos'] is None else round(v['pos'],2)} | D: q {v['q_lossD']:.3f} PG {v['P_G3_D']:.4f}" for mm, v in g.items()))
for nm in ['rain_popw', 'rain_south_popw']:
    print(nm, {mm: (round(v['q_halt'], 3), round(v['P_G3'], 4), round(v['indep'], 5), None if v['pos'] is None else round(v['pos'], 2), round(v['P_G3_D'], 4)) for mm, v in R[nm]['triples']['Dstar0.9_sg0.5'].items()}, 'p_cell', round(R[nm]['p_halt_cell_Dstar0.9'], 3))
print('irr main', {mm: round(v['P_G3'], 4) for mm, v in R['irr']['triples']['Dstar0.9_sg0.5'].items()})
print('phi', [(b['lo'], b['hi'], None if b['phi'] is None else round(b['phi'], 2)) for b in r['phi_by_distance_Dstar0.9']], 'p', round(r['p_halt_cell_Dstar0.9'], 3))
# 図
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
grid = np.full(LAT.shape, np.nan); grid[ii, jj] = Dmax
im = ax[0].pcolormesh(np.where(lon > 180, lon-360, lon), lat, grid, cmap='magma_r', vmin=0, vmax=1, shading='auto')
# 経度を並べ替えて描く
o = np.argsort(np.where(lon > 180, lon-360, lon)); ax[0].clear()
im = ax[0].pcolormesh(np.where(lon > 180, lon-360, lon)[o], lat, grid[:, o], cmap='magma_r', vmin=0, vmax=1, shading='auto')
plt.colorbar(im, ax=ax[0], label='peak yield loss (rainfed 4 crops, yr 1-5)'); ax[0].set_title('150 Tg: peak loss per cell (CLM5, Xia 2022 OSF)')
modes = ['country', 'region', 'world']; xs = np.arange(3)
for j, key in enumerate(['Dstar0.75_sg0.5', 'Dstar0.9_sg0.5', 'Dstar0.9_sg1.0', 'Dstar0.95_sg0.5']):
    ax[1].plot(xs, [r['triples'][key][mm]['P_G3'] for mm in modes], 'o-', label=key)
ax[1].axhline(0.0044, ls='--', c='gray'); ax[1].axhline(0.164, ls=':', c='gray'); ax[1].set_yscale('log')
ax[1].set_xticks(xs, ['same country', 'same subregion\n(3 countries)', 'world random']); ax[1].set_ylabel('P(all 3 groups lose staple)'); ax[1].legend(fontsize=7)
ax[1].set_title('regional loss, G=3 (dashed: 094 indep / full)')
plt.tight_layout(); plt.savefig('fig_halt_correlation.png', dpi=110)
