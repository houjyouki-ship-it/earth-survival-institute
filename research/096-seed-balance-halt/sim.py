"""研究096：農耕の停止の規則を種の収支で置き直す（open_problems 234）
入力：Xia et al. 2022 の OSF（https://osf.io/yrbse/）の CLM5 の格子の収量（150 Tg と対照2本、095 の写し）、
NE 110m の国境、公開の播種量（FAO i2800e07、Odisha の州の種子の頁、agritricks）。
注：出す値は模型の中の割合で、生存確率でも復興確率でもない。乱数は Monte Carlo のみ（seed 96）。"""
import json, numpy as np, netCDF4 as nc
from matplotlib.path import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rng = np.random.default_rng(96)
S = 'sources/'
L = lambda f: np.ma.filled(nc.Dataset(S+f)['yield'][:].astype(float), np.nan)
Y = L('nw150.nc'); C = np.concatenate([L('ctrl03.nc'), L('ctrl03m02.nc')])
lat = np.asarray(nc.Dataset(S+'nw150.nc')['lat'][:]); lon = np.asarray(nc.Dataset(S+'nw150.nc')['lon'][:])
CROPS = {'maize': 1, 'rice': 4, 'soy': 7, 'wheat': 10}          # 天水（095 と同じ列）
# 播種量 t/ha【事実】主：Odisha 表（maize 15, wheat 125, paddy 50）、FAO i2800e07（大豆 80-100→90、米の移植 40-50）
SEED = {'main': dict(maize=0.015, rice=0.05, soy=0.09, wheat=0.125),
        # 多めの版：FAO の米の直播の上限 250、大豆 100、agritricks の maize hybrid 25、小麦 125（agritricks 上限）
        'high': dict(maize=0.025, rice=0.25, soy=0.10, wheat=0.125)}
SMR_ODISHA = dict(maize=100, rice=70, wheat=20)                 # 【事実】Odisha の種の増殖比
# 国の割り当て（095 と同じ）
gj = json.load(open(S+'ne_110m_admin_0_countries.geojson'))
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
# 095 と同じ有効な格子（4 作物の比の平均が 5 年とも有限）
with np.errstate(all='ignore'):
    RAT = np.nanmean(np.stack([np.where(grown[k], np.clip(Y[:, j]/CM[k], 0, 1.5), np.nan) for k, j in CROPS.items()]), 0)
valid = np.all(np.isfinite(RAT[:5]), 0) & (cid >= 0)
ii, jj = np.where(valid); ccell = cid[ii, jj]; wA = W[ii, jj]/W[ii, jj].sum()
nc_ = {c: np.sum(ccell == c) for c in set(ccell)}
wP = np.array([(pops[c] or 0)/nc_[c] for c in ccell], float); wP /= wP.sum()
Dmax095 = (1-RAT[:5, ii, jj]).max(0)
R = {'n_cells': int(valid.sum())}
# H108a：対照の収量÷播種量（面積の重みの中央値、作物を作る格子）
def wmed(x, w):
    o = np.argsort(x); c = np.cumsum(w[o]); return float(x[o][np.searchsorted(c, c[-1]/2)])
R['H108a_ctrl_yield_over_seed'] = {}
for k in CROPS:
    m = grown[k][ii, jj]
    R['H108a_ctrl_yield_over_seed'][k] = dict(main=wmed(CM[k][ii, jj][m]/SEED['main'][k], wA[m]), high=wmed(CM[k][ii, jj][m]/SEED['high'][k], wA[m]),
                                             ctrl_t_ha=wmed(CM[k][ii, jj][m], wA[m]), odisha_SMR=SMR_ODISHA.get(k))

def seed_loss(yld, s, f, xmin, T, noise=None, reserve=0.0):
    """yld: (T, n) t/ha。S_{t+1}=min(1, f*S_t*m_t)。蓄え reserve（播く量の年数）で S を 1 まで補い、
    余りが出た年は蓄えを reserve まで戻す。S<xmin で喪失。"""
    Sx = np.ones(yld.shape[1:]); lost = np.zeros(yld.shape[1:], bool); Rv = np.full(yld.shape[1:], reserve)
    for t in range(T):
        m = yld[t]/s
        if noise is not None: m = m*noise
        H = f*Sx*np.nan_to_num(m, nan=0.0)
        surplus = np.maximum(H-1.0, 0); Rv = np.minimum(reserve, Rv+surplus)
        Sx = np.minimum(1.0, H); top = np.minimum(Rv, 1.0-Sx); Sx = Sx+top; Rv = Rv-top
        lost |= Sx < xmin
    return lost

def cell_losses(seedset='main', f=0.5, xmin=0.01, T=5, d_hardy=0.3, noise=None, reserve=0.0, YY=None):
    YY = Y if YY is None else YY
    out = {}
    for k, j in CROPS.items():
        y = YY[:T, j][:, ii, jj]
        l = seed_loss(y, SEED[seedset][k], f, xmin, T, noise, reserve)
        out[k] = np.where(grown[k][ii, jj], l, True)          # 作らない作物は切り替え先にならない
    gm = np.stack([grown[k][ii, jj] for k in CROPS])
    Lk = np.stack([out[k] for k in CROPS])
    single = np.where(gm.sum(0) > 0, np.sum(Lk & gm, 0)/np.maximum(gm.sum(0), 1), 1.0)   # (i) 1 作物ずつの平均
    switch = Lk.all(0)                                                                     # (ii) 4 作物のどれかが残れば保つ
    # (iii) 寒さに強い作物【推測】：大麦の減り＝0.7×小麦の減り、収量は小麦の対照×(1−減り)、播種量 0.10（agritricks）
    jw = CROPS['wheat']; cw = CM['wheat'][ii, jj]
    with np.errstate(all='ignore'):
        Dw = 1-np.clip(YY[:T, jw][:, ii, jj]/cw, 0, 1.5)
    yb = cw*(1-(1-d_hardy)*Dw)
    lb = np.where(grown['wheat'][ii, jj], seed_loss(yb, 0.10, f, xmin, T, noise, reserve), True)
    hardy = switch & lb
    return dict(single=single, switch=switch.astype(float), hardy=hardy.astype(float), per_crop=out)

# 3 群の置き方（095 と同じ考え方）
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
def summarise(cl, w, TT):
    o = {}
    for nm in ['single', 'switch', 'hardy']:
        x = cl[nm]; o[nm] = dict(q=float(np.sum(w*x)))
        for m, T in TT.items():
            # single は確率として独立に引く。switch/hardy は格子で決まる 0/1
            o[nm]['PG3_'+m] = float(np.mean(np.prod(x[T], 1)))
    return o
VER = {}
for tag, kw in {'main': {}, 'f0.25': dict(f=0.25), 'f0.9': dict(f=0.9), 'xmin0.001': dict(xmin=0.001), 'xmin0.1': dict(xmin=0.1),
                'seedhigh': dict(seedset='high'), 'T10': dict(T=10), 'hardy_d0': dict(d_hardy=0.0), 'hardy_d0.6': dict(d_hardy=0.6),
                'reserve1': dict(reserve=1.0), 'reserve2': dict(reserve=2.0),
                # 対照（平時の気候）に同じ規則を当てた偽陽性：対照 03 の最初の 5 年と 03m02 の最初の 5 年
                'CTRL03_main': dict(YY=C[:14]), 'CTRL03m02_main': dict(YY=C[14:]),
                'CTRL03_reserve1': dict(YY=C[:14], reserve=1.0), 'CTRL03m02_reserve1': dict(YY=C[14:], reserve=1.0),
                'CTRL03_reserve2': dict(YY=C[:14], reserve=2.0)}.items():
    cl = cell_losses(**kw); VER[tag] = summarise(cl, wA, TA)
    if tag == 'main':
        CLmain = cl; VER['main_popw'] = summarise(cl, wP, TP)
        R['per_crop_q_main'] = {k: float(np.sum(wA[grown[k][ii, jj]]*v[grown[k][ii, jj]])/np.sum(wA[grown[k][ii, jj]])) for k, v in cl['per_crop'].items()}
# 群ごとの局地のばらつき（収量×対数正規 σ=0.3・0.6）
for sg in [0.3, 0.6]:
    qs = []
    for rep in range(20):
        cl = cell_losses(noise=np.exp(rng.normal(0, sg, len(ii)) - sg**2/2)); qs.append([np.sum(wA*cl[n]) for n in ['single', 'switch', 'hardy']])
    VER[f'noise{sg}'] = dict(zip(['single_q', 'switch_q', 'hardy_q'], map(float, np.mean(qs, 0))))
R['versions'] = VER
# 記述：095 の規則で止まる（Dmax>0.9）が種の収支では主食を保つ格子（switch）
h095 = Dmax095 > 0.9
R['desc_095halt_but_seed_ok_area'] = float(np.sum(wA*(h095 & (CLmain['switch'] == 0))))
R['desc_095ok_but_seed_lost_area'] = float(np.sum(wA*(~h095 & (CLmain['switch'] == 1))))
R['p_halt095_area'] = float(np.sum(wA*h095))
# 収量がちょうど 0 の年がある格子の割合（各作物）
R['frac_zero_year_area'] = {k: float(np.sum(wA[grown[k][ii, jj]]*(np.nanmin(Y[:5, j][:, ii, jj], 0)[grown[k][ii, jj]] < 1e-3))/np.sum(wA[grown[k][ii, jj]])) for k, j in CROPS.items()}
R['ref095'] = dict(P_loss=[0.53, 0.55], PG3=dict(country=0.233, region=0.182, world=0.163))
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
print('cells', R['n_cells'])
print('H108a', json.dumps(R['H108a_ctrl_yield_over_seed']))
print('per crop q main', {k: round(v, 3) for k, v in R['per_crop_q_main'].items()})
print('zero-yield year frac', {k: round(v, 3) for k, v in R['frac_zero_year_area'].items()})
for tag, v in VER.items():
    print(tag, json.dumps({n: ({kk: round(vv, 4) for kk, vv in d.items()} if isinstance(d, dict) else round(d, 4)) for n, d in v.items()}))
print('095 halt area %.3f ; 095halt&seedOK %.3f ; 095ok&seedLost %.3f' % (R['p_halt095_area'], R['desc_095halt_but_seed_ok_area'], R['desc_095ok_but_seed_lost_area']))
# 図
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); o = np.argsort(np.where(lon > 180, lon-360, lon))
g = np.full(LAT.shape, np.nan); g[ii, jj] = CLmain['switch'] + CLmain['hardy']*0  # 0/1
g2 = np.full(LAT.shape, np.nan); g2[ii, jj] = np.where(CLmain['switch'] == 1, np.where(CLmain['hardy'] == 1, 2, 1), 0)
im = ax[0].pcolormesh(np.where(lon > 180, lon-360, lon)[o], lat, g2[:, o], cmap='RdYlGn_r', vmin=0, vmax=2, shading='auto')
ax[0].set_title('seed balance (f=0.5): 0 kept, 1 lost unless hardy crop, 2 lost'); plt.colorbar(im, ax=ax[0])
tags = ['main', 'f0.25', 'f0.9', 'xmin0.1', 'seedhigh', 'T10', 'reserve1', 'reserve2', 'CTRL03_main', 'CTRL03_reserve1']; xs = np.arange(len(tags))
for n, mk in [('single', 'o'), ('switch', 's'), ('hardy', '^')]:
    ax[1].plot(xs, [VER[t][n]['q'] for t in tags], mk+'-', label=n)
ax[1].axhspan(0.53, 0.55, color='gray', alpha=0.3, label='095 P_loss'); ax[1].set_xticks(xs, tags, rotation=30); ax[1].set_ylabel('per-group staple loss q'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('fig_seed_balance.png', dpi=110)
