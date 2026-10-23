"""研究102：核の冬の気候での芋の疫病の好適な月（open_problems 246）。親の固定指示で、芋の疫病と核の冬の気温の系列は本回で閉じる。
入力：101 の計算（そのまま再実行、100・099・098・097・096 の差0の assert を含む）。CESM（NWDC の OSF https://osf.io/e28gq/ の nw_ur_150_07・nw_cntrl_03）
の TS と PRECC＋PRECL（097・098 で取得済み）。湿度は出力にない。
資料【事実】：UC IPM「Models: Late Blight of Potato」の Hyre の規則（https://ipm.ucanr.edu/DISEASE/DATABASE/potatolateblight.html 、sources/ に保存）、
Wikipedia「Phytophthora infestans」（10 °C 超・湿度 75〜80% 超が 2 日以上、Beaumont/Smith の期間、101 の sources/）。
注：出す値は模型の中の割合で、生存確率でも復興確率でもない。乱数は 101 と同じ順のあと、本回の Monte Carlo だけ seed 102。"""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ME0 = os.path.dirname(os.path.abspath(__file__)); D101 = os.path.join(ME0, '../101-gmult-lowtail')
s101 = open(os.path.join(D101, 'sim.py')).read()
s101 = s101.replace("ME0 = os.path.dirname(os.path.abspath(__file__))", f"ME0 = {D101!r}")
exec(s101[:s101.rindex("json.dump(R, open('results.json'")])
ME = ME0 = os.path.dirname(os.path.abspath(__file__)); os.chdir(ME)
R101 = R; R = {}
R['repro101'] = dict(q0=R101['versions']['main_no_noise']['q'], q_noise=R101['versions']['noise_main']['q'], q_blight_all=R101['versions']['blight_all_years_075']['q'])
assert abs(R['repro101']['q0']-0.47619) < 1e-5 and abs(R['repro101']['q_noise']-0.48032) < 1e-5 and abs(R['repro101']['q_blight_all']-0.56189) < 1e-5, R['repro101']
print('repro101', R['repro101'])

# ---------------- 疫病の好適な月（Hyre の月の版） ----------------
# T150, TCT：作物年の月の TS（°C）、(年, 12, 96, 144)。P150y, PCTy：作物年の月の雨（mm/月）。SEAS：作物年の 4 か月。
def blight_years(T, P, pmin=90.0, dtr=10.0, tlo_mode='hyre', nmin=2, seas=SEAS):
    Ts = T[:, seas]; Ps = P[:, seas]
    tok = (Ts < 25.5) & ((Ts-dtr/2 >= 7.2) if tlo_mode == 'hyre' else (Ts >= 10.0))
    fav = tok & (Ps >= pmin)
    return fav.sum(1) >= nmin, fav                     # (年, 96, 144)
VERS = {'main': {}, 'p45': dict(pmin=45.0), 'p60': dict(pmin=60.0), 'n1': dict(nmin=1), 't10': dict(tlo_mode='t10'),
        'dtr6': dict(dtr=6.0), 'dtr14': dict(dtr=14.0), 'p45_n1': dict(pmin=45.0, nmin=1)}
# 平時の作物年は 20 年。150 Tg は 5 年（096 の Y[:5] と同じ作物年 5〜9）。
LP = 0.75
def lost_cell(Lpc, gb=GBm, gp=GPm, nb=1.0, np_=1.0, f=0.5):
    lb = seed_loss(np.where(BOK, gb*MB*nb, 0.0), 1.0, f, 0.01, 5, 0.0)
    lp = seed_loss(np.where(POK, gp*MP*np_*(1-Lpc), 0.0), 1.0, f, 0.01, 5, 0.0)
    return SWL & lb & lp
assert abs(summ(lost_cell(np.zeros(MP.shape)).astype(float))['q']-R['repro101']['q0']) < 1e-12
lat_g = lat; lon_g = lon
def box(la0, la1, lo0, lo1):
    m = np.zeros((96, 144), bool); m[np.ix_((lat_g >= la0) & (lat_g <= la1), (lon_g >= lo0) & (lon_g <= lo1))] = True; return m
IRL_UK = box(50.0, 58.0, 350.0, 360.0) | box(50.0, 58.0, 0.0, 1.5)
NL_BE = box(50.5, 53.5, 3.0, 7.0)
# 陸の目安：097 の有効な格子（作物のある格子）を使う
LAND = np.zeros((96, 144), bool); LAND[ii, jj] = True
potc = POK.any(0)                                       # 150 Tg で芋が一度でも実る格子
wpot = wA*potc; wpot = wpot/wpot.sum()
wgain = wA*gain; wgain = wgain/wgain.sum()
rng = np.random.default_rng(102)
OUT = {}
for k, v in VERS.items():
    B150, F150 = blight_years(T150, P150y, **v); BCT, FCT = blight_years(TCT, PCTy, **v)
    b150 = cells(B150).astype(float); bct = cells(BCT).astype(float)        # (年, ncell)
    o = dict(p_bl_150_gain=float(np.sum(wgain*b150.mean(0))), p_bl_ctl_gain=float(np.sum(wgain*bct.mean(0))),
             p_bl_150_potcells=float(np.sum(wpot*b150.mean(0))), p_bl_ctl_potcells=float(np.sum(wpot*bct.mean(0))),
             p_bl_ctl_allcrop=float(np.sum(wA*bct.mean(0))), p_bl_150_allcrop=float(np.sum(wA*b150.mean(0))),
             ctl_IRL_UK_land=float(BCT[:, IRL_UK & LAND].mean()) if (IRL_UK & LAND).any() else None,
             ctl_IRL_UK_all=float(BCT[:, IRL_UK].mean()), ctl_NL_BE=float(BCT[:, NL_BE].mean()),
             w150_IRL_UK_all=float(B150[:, IRL_UK].mean()))
    # q：150 Tg の実際の疫病の年（格子×年）で芋を 0.75 減らす
    o['q_actual'] = summ(lost_cell(LP*b150).astype(float))
    # 版：各格子の年の確率 p（150 Tg の 5 年の平均）で独立に引く
    p = b150.mean(0); acc = np.zeros(SWL.shape)
    for _ in range(200): acc += lost_cell(LP*(rng.random(MP.shape) < p[None, :]))
    o['q_indep_draw'] = summ(acc/200)
    # 版：平時の確率で引く（核の冬が疫病を変えなかったら）
    pc = bct.mean(0); acc = np.zeros(SWL.shape)
    for _ in range(200): acc += lost_cell(LP*(rng.random(MP.shape) < pc[None, :]))
    o['q_ctlprob_draw'] = summ(acc/200)
    OUT[k] = o
    if k == 'main': B150m, BCTm, F150m, FCTm, b150m, bctm = B150, BCT, F150, FCT, b150, bct
R['versions'] = OUT
M = OUT['main']
# 原因の分解（主の規則、得の格子、季節の月の単位）：気温の条件と雨の条件のどちらで落ちるか
def decomp(T, P, w):
    Ts = cells(T[:, SEAS].transpose(1, 0, 2, 3).reshape(4*T.shape[0], 96, 144)); Ps = cells(P[:, SEAS].transpose(1, 0, 2, 3).reshape(4*P.shape[0], 96, 144))
    tok = (Ts < 25.5) & (Ts-5 >= 7.2); pok = Ps >= 90
    f = lambda x: float(np.sum(w*x.mean(0)))
    return dict(temp_ok=f(tok), rain_ok=f(pok), both=f(tok & pok), too_hot=f(Ts >= 25.5), too_cold=f(Ts-5 < 7.2),
                median_T=float(np.median(Ts[:, w > 0])), median_P=float(np.median(Ps[:, w > 0])))
R['decomp_gain'] = dict(w150=decomp(T150, P150y, wgain), ctl=decomp(TCT, PCTy, wgain))
R['decomp_potcells'] = dict(w150=decomp(T150, P150y, wpot), ctl=decomp(TCT, PCTy, wpot))
# 平時の誤検出の参照：疫病の年が平時で全面（毎年）の格子の割合
R['ctl_every_year_frac_gain'] = float(np.sum(wgain*(bctm.mean(0) >= 0.999)))
# 揺れつき（101 の σ）＋主の実際の疫病の年
sig = R101['sigmas_used']['main']; acc = np.zeros(SWL.shape)
for _ in range(200):
    zb = rng.standard_normal(MB.shape); zp = rng.standard_normal(MP.shape)
    acc += lost_cell(LP*b150m, nb=np.exp(sig*zb-sig**2/2), np_=np.exp(sig*zp-sig**2/2))
R['q_actual_noise'] = summ(acc/200)
R['q_blight_frac_sweep'] = {str(L): summ(lost_cell(L*b150m).astype(float))['q'] for L in (0.25, 0.5, 0.75, 1.0)}
R['H114a'] = dict(p150=M['p_bl_150_gain'], pctl=M['p_bl_ctl_gain'], supported=bool(M['p_bl_150_gain'] < M['p_bl_ctl_gain']))
R['H114b'] = dict(q=M['q_actual']['q'], supported=bool(0.476 <= M['q_actual']['q'] <= 0.50))
R['H114c'] = dict(ctl_IRL_UK=M['ctl_IRL_UK_all'], ctl_gain=M['p_bl_ctl_gain'], supported=bool(M['ctl_IRL_UK_all'] >= 0.5 and M['p_bl_ctl_gain'] < 0.5))
R['H114d'] = dict(q_p45=OUT['p45']['q_actual']['q'], supported=bool(OUT['p45']['q_actual']['q'] <= 0.53))
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in R.items(): print(k, json.dumps(v, ensure_ascii=False))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
ext = [lon_g[0], lon_g[-1], lat_g[0], lat_g[-1]]
for a, B, t in [(ax[0], BCTm, 'control: blight-year fraction (Hyre monthly)'), (ax[1], B150m, '150 Tg yrs 5-9: blight-year fraction')]:
    im = a.imshow(np.where(LAND, B.mean(0), np.nan), origin='lower', extent=ext, aspect='auto', vmin=0, vmax=1, cmap='viridis'); a.set_title(t, fontsize=9)
plt.colorbar(im, ax=ax[1])
ks = list(OUT); ax[2].bar(np.arange(len(ks))-0.2, [OUT[k]['p_bl_ctl_gain'] for k in ks], 0.4, label='ctrl gain cells')
ax[2].bar(np.arange(len(ks))+0.2, [OUT[k]['p_bl_150_gain'] for k in ks], 0.4, label='150 Tg gain cells')
ax2 = ax[2].twinx(); ax2.plot(range(len(ks)), [OUT[k]['q_actual']['q'] for k in ks], 'k.-', label='q'); ax2.set_ylim(0.45, 0.58); ax2.set_ylabel('q')
ax[2].set_xticks(range(len(ks)), ks, rotation=45, fontsize=8); ax[2].set_ylabel('blight-year fraction'); ax[2].legend(fontsize=7, loc='upper left')
plt.tight_layout(); plt.savefig('fig_blight.png', dpi=110)
