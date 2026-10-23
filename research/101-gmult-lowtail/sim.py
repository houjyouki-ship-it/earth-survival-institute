"""研究101：崩壊後の増殖比が前近代の下端に落ちる条件（open_problems 243）。
入力：100 の計算（そのまま再実行、099・098・097・096 の差0の assert を含む）。
資料【事実は report.md と sources/】：BAHS Medieval Crop Yields Database（Campbell、Titow・Farmer の計算を含む、
https://www.bahs.org.uk/crop-yields-database/download-data/ の Barley.xls・Wheat.xls、荘園×年の粗い増殖比）、
Wikipedia「Great Famine (Ireland)」「Highland Potato Famine」（疫病の年の損）。
注：出す値は模型の中の割合で、生存確率でも復興確率でもない。乱数は 100 と同じ順のあと、本回の Monte Carlo だけ seed 101。"""
import os, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ME0 = os.path.dirname(os.path.abspath(__file__)); D100 = os.path.join(ME0, '../100-gmult-eto-pm')
s100 = open(os.path.join(D100, 'sim.py')).read()
s100 = s100.replace("ME = os.path.dirname(os.path.abspath(__file__))", f"ME = {D100!r}")
exec(s100[:s100.rindex("json.dump(R, open('results.json'")])
ME = ME0; os.chdir(ME)                                   # 100 の注意：exec の後で自分の ME に戻す
R100 = R; R = {}
R['repro100'] = dict(q_main=R100['versions']['pm_gsrc_MAIN'], q_lo=R100['versions']['pm_g_lo_2.5_1.25'])
assert abs(R['repro100']['q_main']-0.47619) < 1e-5 and abs(R['repro100']['q_lo']-0.63512) < 1e-5, R['repro100']
print('repro100', R['repro100'])

# ---------------- (2) BAHS の荘園×年の増殖比 ----------------
def load_bahs(fn, sheet):
    v = pd.read_excel(os.path.join('sources', fn), sheet_name=sheet, header=None)
    hdr = v.iloc[1].tolist(); end = 2
    while end < v.shape[1] and isinstance(hdr[end], str) and hdr[end].strip() != 'beg year': end += 1
    yrs = pd.to_numeric(v.iloc[2:, 0], errors='coerce').values; n = int(np.argmax(~np.isfinite(yrs))) if (~np.isfinite(yrs)).any() else len(yrs)
    X = v.iloc[2:2+n, 2:end].apply(pd.to_numeric, errors='coerce').values.astype(float)
    return yrs[:n].astype(int), X, end-2
def stats(yrs, X, minobs=10):
    X = np.where(X > 0, X, np.nan); L = np.log(X); cnt = np.isfinite(L).sum(0); L = L[:, cnt >= minobs]; Xk = X[:, cnt >= minobs]
    dev = L-np.nanmean(L, 0)                                    # 主：荘園の平均からの対数の偏差
    roll = pd.DataFrame(L).rolling(11, center=True, min_periods=5).mean().values; dev_r = L-roll   # 版：11 年の移動平均から
    yr_eff = np.nanmean(dev, 1); yr_n = np.isfinite(dev).sum(1); ye = yr_eff[yr_n >= 10]      # 年の共通の成分（10 荘園以上の年）
    fin = Xk[np.isfinite(Xk)]
    # 5 年の窓の最悪の年÷荘園の中央値（窓に 5 年そろう荘園のみ）
    w = []
    med = np.nanmedian(Xk, 0)
    for j in range(Xk.shape[1]):
        x = Xk[:, j]
        for s in range(len(x)-4):
            seg = x[s:s+5]
            if np.isfinite(seg).all(): w.append(seg.min()/med[j])
    w = np.array(w)
    return dict(n_manors=int(Xk.shape[1]), n_obs=int(fin.size), median=float(np.median(fin)), q05=float(np.percentile(fin, 5)), q10=float(np.percentile(fin, 10)),
                frac_lt2=float(np.mean(fin < 2)), frac_lt1_5=float(np.mean(fin < 1.5)), frac_lt1=float(np.mean(fin < 1)),
                sigma_manor_mean=float(np.nanstd(dev)), sigma_roll11=float(np.nanstd(dev_r)), sigma_year_common=float(np.std(ye)), n_years_common=int(ye.size),
                sigma_by_manor_q25_q75=[float(np.nanpercentile(np.nanstd(dev, 0), 25)), float(np.nanpercentile(np.nanstd(dev, 0), 75))],
                win5_worst_over_median=dict(n=int(w.size), median=float(np.median(w)), q10=float(np.percentile(w, 10)), frac_lt_0_5=float(np.mean(w < 0.5)))), Xk, dev
yb, Xb, nb = load_bahs('Barley.xls', 'Sheet1'); yw, Xw, nw = load_bahs('Wheat.xls', 'Yields')
SB, Xbk, devb = stats(yb, Xb); SWh, Xwk, devw = stats(yw, Xw)
R['bahs'] = dict(barley=SB, wheat=SWh, years_barley=[int(yb[0]), int(yb[-1])], years_wheat=[int(yw[0]), int(yw[-1])])
R['H113a'] = dict(sigma_barley=SB['sigma_manor_mean'], sigma_wheat=SWh['sigma_manor_mean'], supported=bool(SB['sigma_manor_mean'] >= 0.2 and SWh['sigma_manor_mean'] >= 0.2))
# 【事実：Wikipedia】Ireland 1845 の損 1/3〜1/2（作付面積）、1846 は収穫の 3/4、1847 は平年の収量だが植えた量が少ない、1848 は平年の 2/3。Highlands 1846 は小作地域の約 3/4 で不作。
BL = dict(ireland_1845=(1/3, 1/2), ireland_1846=0.75, ireland_1848=1/3, highlands_1846_area=0.75)
R['H113b'] = dict(losses=BL, max_loss=0.75, supported=True)

# ---------------- (1) 臨界の増殖比 g* ----------------
def lost_cells(gb, gp, noise_b=None, noise_p=None, f=0.5, Lp=None):
    wb = water_ratio(1.15, 550, True); wp = water_ratio(1.1, 600, True)
    return None
# 100 の q_new の中身（PM、light fit）を格子ごとの喪失の配列で返す形に
keep = Lrs; Lrs = ER
MB = water_ratio(1.15, 550, True)[0]*light_ratio('barley', 'fit'); MP = water_ratio(1.1, 600, True)[0]*light_ratio('potato', 'fit')
Lrs = keep
SWL = switch_lost(Y)
def lost(gb, gp, nb=1.0, np_=1.0, f=0.5, Lp=None):
    lp_mult = 1.0 if Lp is None else (1-np.asarray(Lp))[:, None]
    lb = seed_loss(np.where(BOK, gb*MB*nb, 0.0), 1.0, f, 0.01, 5, 0.0)
    lp = seed_loss(np.where(POK, gp*MP*np_*lp_mult, 0.0), 1.0, f, 0.01, 5, 0.0)
    return SWL & lb & lp
assert abs(summ(lost(GBm, GPm).astype(float))['q']-R['repro100']['q_main']) < 1e-12
GG = np.exp(np.linspace(np.log(1.0), np.log(200.0), 600))
LOSTG = np.array([lost(g, g) for g in GG])                     # 共通の g
alive = ~LOSTG; gstar = np.where(alive.any(0), GG[np.argmax(alive, 0)], np.inf)
rescuable = SWL & np.isfinite(gstar)
def wmed(x, w):
    o = np.argsort(x); c = np.cumsum(w[o])/w[o].sum(); return float(x[o][np.searchsorted(c, 0.5)])
wr = wA*rescuable
R['gstar'] = dict(frac_switch_lost=float(np.sum(wA*SWL)), frac_rescuable=float(wr.sum()), frac_never=float(np.sum(wA*(SWL & ~np.isfinite(gstar)))),
                  wmedian_gstar_rescuable=wmed(gstar[rescuable], wA[rescuable]),
                  wq25_q75=[float(np.percentile(gstar[rescuable], 25)), float(np.percentile(gstar[rescuable], 75))],
                  frac_rescuable_gstar_le={str(t): float(np.sum(wr*(gstar <= t))/wr.sum()) for t in (2, 2.5, 3, 4.01, 5.48, 10, 20)})
R['H113c'] = dict(wmedian=R['gstar']['wmedian_gstar_rescuable'], supported=bool(2.5 <= R['gstar']['wmedian_gstar_rescuable'] <= 4.01))
R['q_curve_common_g'] = {f'{g:.3g}': float(np.sum(wA*LOSTG[i])) for i, g in enumerate(GG) if i % 40 == 0}
# 解析の境目：m≡1（平年）で S は f·g≥1 なら保たれ、f·g<1 なら (f·g)^t で減る。5 年で 0.01 を割るのは f·g < 0.01^(1/5)=0.398
R['analytic'] = dict(g_break_even_f05=2.0, g_5yr_loss_m1_f05=float(0.01**0.2/0.5),
                     note='m_t≡m なら喪失は f·g·m < 0.398 ⇔ g·m < 0.796（f=0.5）。最初の年の S=1 から 5 回の掛け算。')

# ---------------- (2') 年ごとの揺れ ----------------
rng = np.random.default_rng(101); NMC = 200
def q_noise(sig, corr=0.0, gb=GBm, gp=GPm, f=0.5, w=wA, TT=TA):
    acc = np.zeros(SWL.shape)
    for _ in range(NMC):
        zb = rng.standard_normal(MB.shape); zp = zb if corr == 1 else rng.standard_normal(MP.shape)
        acc += lost(gb, gp, np.exp(sig*zb-sig**2/2), np.exp(sig*zp-sig**2/2), f=f)
    return summ(acc/NMC, w, TT)
sig_main = SB['sigma_manor_mean']; sig_lo = min(SB['sigma_year_common'], SWh['sigma_year_common']); sig_hi = max(SB['sigma_manor_mean'], SWh['sigma_manor_mean'], SB['sigma_roll11'], SWh['sigma_roll11'])
V = {}
V['main_no_noise'] = summ(lost(GBm, GPm).astype(float))
V['noise_main'] = q_noise(sig_main); V['noise_lo_common'] = q_noise(sig_lo); V['noise_hi'] = q_noise(sig_hi); V['noise_main_corr1'] = q_noise(sig_main, corr=1)
V['noise_main_f04'] = q_noise(sig_main, f=0.4); V['noise_main_f06'] = q_noise(sig_main, f=0.6)
V['nonoise_f04'] = summ(lost(GBm, GPm, f=0.4).astype(float)); V['nonoise_f06'] = summ(lost(GBm, GPm, f=0.6).astype(float))
V['noise_main_popw'] = q_noise(sig_main, w=wP, TT=TP)
R['sigmas_used'] = dict(main=sig_main, lo=sig_lo, hi=sig_hi)
R['H113d'] = dict(q0=V['main_no_noise']['q'], q_noise=V['noise_main']['q'], delta=V['noise_main']['q']-V['main_no_noise']['q'],
                  supported=bool(V['noise_main']['q']-V['main_no_noise']['q'] >= 0.03))
# ---------------- (3) 疫病 ----------------
V['blight_all_years_075'] = summ(lost(GBm, GPm, Lp=[0.75]*5).astype(float))
V['blight_year1_075'] = summ(lost(GBm, GPm, Lp=[0.75, 0, 0, 0, 0]).astype(float))
V['blight_ireland_seq'] = summ(lost(GBm, GPm, Lp=[5/12, 0.75, 0.0, 1/3, 0.0]).astype(float))     # 登録外：1845〜1849 の並び（1849 は 0 と置く）
V['potato_absent'] = summ(lost(GBm, 0.0).astype(float))                                             # 参考：芋なし
V['blight_all_noise'] = q_noise(sig_main, gp=GPm*0.25)                                              # 登録外：全年の疫病＋揺れ
R['H113e'] = dict(q=V['blight_all_years_075']['q'], supported=bool(V['blight_all_years_075']['q'] >= 0.55))
# 平時の偽陽性（揺れつき）：100 の ctrl_false5 は揺れなし。揺れを入れた版（登録外）
def ctrl_false5_noise(gb, gp, sig, nmc=50):
    Pc = cells(PCTy[:, SEAS].sum(1)); erc = ECT/np.where(ECT.mean(0) > 0, ECT.mean(0), np.nan); erc = np.nan_to_num(erc, nan=1.0)
    def wr(ky, etm, y):
        et = etm*erc[y]; fy = np.clip(1-ky*(1-np.minimum(1, Pc[y]/et)), 0, 1); fc = np.clip(1-ky*(1-np.minimum(1, Pabs_ctl_s/etm)), 0, 1)
        return np.where(fc > 0.05, np.minimum(1, fy/np.maximum(fc, 1e-9)), fy)
    out = []
    for s0 in range(16):
        mb = np.array([wr(1.15, 550, s0+t) for t in range(5)]); mp = np.array([wr(1.1, 600, s0+t) for t in range(5)])
        a = 0
        for _ in range(nmc):
            nb_ = np.exp(sig*rng.standard_normal(mb.shape)-sig**2/2); np2 = np.exp(sig*rng.standard_normal(mp.shape)-sig**2/2)
            a += float(np.sum(wg*(seed_loss(gb*mb*nb_, 1.0, 0.5, 0.01, 5, 0.0) & seed_loss(gp*mp*np2, 1.0, 0.5, 0.01, 5, 0.0))))
        out.append(a/nmc)
    return float(np.mean(out))
R['ctrl_false_5yr_noise'] = dict(no_noise=R100['ctrl_false_5yr']['gsrc'], noise_main=ctrl_false5_noise(GBm, GPm, sig_main))
R['versions'] = {k: (v if isinstance(v, dict) else v) for k, v in V.items()}
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in R.items(): print(k, json.dumps(v, ensure_ascii=False))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
fb = Xbk[np.isfinite(Xbk)]; fw = Xwk[np.isfinite(Xwk)]
ax[0].hist([fb, fw], bins=np.arange(0, 15, 0.25), label=[f'barley n={fb.size}', f'wheat n={fw.size}'], histtype='step')
ax[0].axvline(2.0, color='k', ls='--', lw=0.8); ax[0].set_xlabel('gross yield / seed (manor-year)'); ax[0].legend(fontsize=8); ax[0].set_title('BAHS 1211-1491')
qc = np.array([np.sum(wA*LOSTG[i]) for i in range(len(GG))])
ax[1].plot(GG, qc); ax[1].set_xscale('log'); ax[1].axvline(GBm, color='C0', ls=':', lw=0.8); ax[1].axvline(GPm, color='C1', ls=':', lw=0.8); ax[1].axvline(2.5, color='gray', ls=':')
ax[1].set_xlabel('common g (barley=potato)'); ax[1].set_ylabel('q'); ax[1].set_title('q vs yield ratio')
labs = ['no noise', 'noise', 'noise lo', 'noise hi', 'corr1', 'f0.4 n', 'f0.6 n', 'blight all', 'blight y1', 'Ireland seq', 'no potato']
keys = ['main_no_noise', 'noise_main', 'noise_lo_common', 'noise_hi', 'noise_main_corr1', 'noise_main_f04', 'noise_main_f06', 'blight_all_years_075', 'blight_year1_075', 'blight_ireland_seq', 'potato_absent']
ax[2].bar(range(len(keys)), [V[k]['q'] for k in keys], color='C2'); ax[2].set_xticks(range(len(keys)), labs, rotation=45, fontsize=8); ax[2].set_ylabel('q')
plt.tight_layout(); plt.savefig('fig_gmult_lowtail.png', dpi=110)
