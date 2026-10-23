"""研究099：作物の雨と光への収量の応答の形を資料で決め、q を置き直す（open_problems 239＋240）
入力：098 の計算（そのまま再実行、097・096 の差0の assert を含む）。
資料【事実】：FAO Irrigation Water Management Training Manual 3 の Table 14（https://www.fao.org/4/s2022e/s2022e07.htm ）
  季節の作物の水の要求 Barley/Oats/Wheat 450-650 mm、Potato 500-700 mm。
  FAO 33 の季節の Ky（FAO i2800e の Table 1、https://www.fao.org/4/i2800e/i2800e02.pdf ）Spring wheat 1.15、Winter wheat 1.05、Potato 1.1（大麦の値は表にないので小麦で代用）。
  遮光：Yadav et al. 2017 J. Agrometeorol. 19(4):323-328（New Delhi の小麦、分げつ期から成熟まで遮光。粒の収量の減り 2014-15 は 20/35/50/75% 遮光で 23/48/68/81%、2015-16 は 75/50/35% で 88/74/58%）
  https://journal.agrimetassociation.org/index.php/jam/article/download/599/499 、
  Schulz et al. 2019 Agronomy 9(6):330（Germany のジャガイモ、50% 遮光で塊茎の乾物の減り 19-44%、2015 は出芽から収穫まで 10.11 MJ m-2 d-1 でも減りなし）
  https://www.mdpi.com/2073-4395/9/6/330 。
注：出す値は模型の中の割合で、生存確率でも復興確率でもない。乱数は 097・098 と同じ（seed 96、同じ順）。"""
import os, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ME = os.path.dirname(os.path.abspath(__file__)); D98 = os.path.join(ME, '../098-lowlat-loss-light-rain')
s98 = open(os.path.join(D98, 'sim.py')).read()
s98 = s98.replace("HERE = os.path.dirname(os.path.abspath(__file__))", f"HERE = {D98!r}")
exec(s98[:s98.index("json.dump(R, open('results.json'")])
os.chdir(ME)
R98 = R; R = {}
R['repro098'] = dict(q_phys05=R98['qphys']['thr0.5']['q'], q_prop_product=R98['explore_prop']['product']['q'])
assert abs(R['repro098']['q_phys05']-0.6454131434002193) < 1e-12 and abs(R['repro098']['q_prop_product']-0.4335467742785113) < 1e-12, R['repro098']

# ---- 遮光の試験から光の応答の指数 γ（収量比 = L^γ）を最小二乗（log）で ----
wheat_pts = [(0.80, 0.77), (0.65, 0.52), (0.50, 0.32), (0.25, 0.19), (0.25, 0.12), (0.50, 0.26), (0.65, 0.42)]  # (光の比, 収量比) Yadav 2017
pot_pts = [(0.50, 0.81), (0.50, 0.56)]                                                                          # Schulz 2019 の 50% 遮光の両端
fitg = lambda P: float(np.sum([np.log(y)*np.log(x) for x, y in P])/np.sum([np.log(x)**2 for x, y in P]))
g_w = fitg(wheat_pts); g_p = fitg(pot_pts); g_p_lo = np.log(0.81)/np.log(0.5); g_p_hi = np.log(0.56)/np.log(0.5)
r50 = [y for x, y in wheat_pts+pot_pts if x == 0.5]
R['light_fit'] = dict(gamma_wheat=g_w, gamma_potato=g_p, gamma_potato_range=[g_p_lo, g_p_hi], yield_ratio_at_50pct_shade=r50,
                      H111b_median_ratio_50pct=float(np.median(r50)), proportional=0.5)

# ---- 絶対の光（季節の FSDS、W/m2）。ジャガイモの飽和版：S_sat = 10.11 MJ/m2/d ≈ 117 W/m2 ----
S_s = np.full((5, len(ii)), np.nan); S_ctl = np.full(len(ii), np.nan)
for k in range(len(ii)):
    nm = names[ccell[k]]
    if nm is None: continue
    a = fs_crop(f150[[F150.index.get_loc(nm)]], 5, sh_c[k])[0]; c = fs_crop(fct[[FCT.index.get_loc(nm)]], 20, sh_c[k])[0]
    S_s[:, k] = a[:, SEAS].mean(1); S_ctl[k] = c[:, SEAS].mean()
SSAT = 10.11e6/86400
R['desc_abs_light'] = dict(median_ctl_Wm2_gain=float(np.nanmedian(S_ctl[gain])), median_y2_Wm2_gain=float(np.nanmedian(S_s[1][gain])), S_sat_Wm2=SSAT,
                           frac_gain_y2_below_Ssat=float(np.sum(wg*(np.nan_to_num(S_s[1], nan=999) < SSAT))))

# ---- 水：FAO 33 の Ky の式。季節の雨（4 か月）÷ ETm、ETm150 = ETm_ctl × L_r（版で比例させない） ----
Lrs = np.minimum(1, np.nan_to_num(Lr_s, nan=1.0))
def water_ratio(ky, etm, scale_et=True):
    et150 = etm*(Lrs if scale_et else 1.0)
    f150_ = np.clip(1-ky*(1-np.minimum(1, Pabs_s/et150)), 0, 1)
    fct_ = np.clip(1-ky*(1-np.minimum(1, Pabs_ctl_s/etm)), 0, 1)
    return np.where(fct_ > 0.05, np.minimum(1, f150_/np.maximum(fct_, 1e-9)), f150_), f150_, fct_
Pabs_ctl_s = cells(PCTy[:, SEAS].sum(1)).mean(0)
def light_ratio(crop, kind):
    if kind == 'prop': return Lrs
    if kind == 'fit': return Lrs**(g_w if crop == 'barley' else g_p)
    if kind == 'fit_lo': return Lrs**(g_w if crop == 'barley' else g_p_lo)
    if kind == 'fit_hi': return Lrs**(g_w if crop == 'barley' else g_p_hi)
    if kind == 'sat':   # 大麦は fit、ジャガイモは絶対の飽和
        if crop == 'barley': return Lrs**g_w
        s150 = np.minimum(1, np.nan_to_num(S_s, nan=999)/SSAT); sc = np.minimum(1, np.nan_to_num(S_ctl, nan=999)/SSAT)
        return np.minimum(1, s150/sc)
def q_resp(ky_b=1.15, ky_p=1.1, et_b=550, et_p=600, scale_et=True, light='fit', gmult=10.0, w=wA, TT=TA, ret_m=False):
    mb = water_ratio(ky_b, et_b, scale_et)[0]*light_ratio('barley', light)
    mp = water_ratio(ky_p, et_p, scale_et)[0]*light_ratio('potato', light)
    lb = seed_loss(np.where(BOK, gmult*mb, 0.0), 1.0, 0.5, 0.01, 5, 0.0); lp = seed_loss(np.where(POK, gmult*mp, 0.0), 1.0, 0.5, 0.01, 5, 0.0)
    out = summ((switch_lost(Y) & lb & lp).astype(float), w, TT)
    return (out, mb, mp) if ret_m else out
main, MB, MP = q_resp(ret_m=True)
R['q_main'] = main
V = {}
V['light_prop'] = q_resp(light='prop'); V['light_sat_potato'] = q_resp(light='sat')
V['light_fit_potato_lo'] = q_resp(light='fit_lo'); V['light_fit_potato_hi'] = q_resp(light='fit_hi')
V['no_et_scaling'] = q_resp(scale_et=False); V['no_et_scaling_light_prop'] = q_resp(scale_et=False, light='prop')
V['ky_low_1.05'] = q_resp(ky_b=1.05); V['etm_low_450_500'] = q_resp(et_b=450, et_p=500); V['etm_high_650_700'] = q_resp(et_b=650, et_p=700)
V['gmult5'] = q_resp(gmult=5.0); V['gmult20'] = q_resp(gmult=20.0); V['popw'] = q_resp(w=wP, TT=TP)
V['best_case'] = q_resp(ky_b=1.05, et_b=450, et_p=500, light='sat', gmult=20.0)
V['worst_case'] = q_resp(et_b=650, et_p=700, scale_et=False, light='fit', gmult=5.0)
R['versions'] = V
R['H111c'] = dict(q=main['q'], midpoint=(0.6454+0.4335)/2)
wmin_b = water_ratio(1.15, 550)[0].min(0); wmin_p = water_ratio(1.1, 600)[0].min(0)
R['H111d'] = dict(frac_gain_worst_water_lt01_barley=float(np.sum(wg*(wmin_b < 0.1))), frac_gain_worst_water_lt01_potato=float(np.sum(wg*(wmin_p < 0.1))),
                  frac_gain_worst_water_lt01_both=float(np.sum(wg*((wmin_b < 0.1) & (wmin_p < 0.1)))),
                  median_worst_water_barley=float(np.median(wmin_b[gain])), median_worst_water_potato=float(np.median(wmin_p[gain])))
R['desc_m'] = dict(median_worst_m_barley=float(np.median(MB.min(0)[gain])), median_worst_m_potato=float(np.median(MP.min(0)[gain])),
                   frac_gain_kept_main=float(np.sum(wg*~(seed_loss(np.where(BOK, 10*MB, 0.0), 1.0, 0.5, 0.01, 5, 0.0) & seed_loss(np.where(POK, 10*MP, 0.0), 1.0, 0.5, 0.01, 5, 0.0)))),
                   median_ctl_season_rain_mm=float(np.median(Pabs_ctl_s[gain])), frac_gain_ctl_water_factor_lt05_barley=float(np.sum(wg*(water_ratio(1.15, 550)[2] < 0.5))))
# 平時の偽陽性：対照の各年に同じ規則を当てる（得の格子）
def ctrl_false(ky=1.15, etm=550):
    out = []
    for y in range(20):
        Pc = cells(PCTy[y:y+1, SEAS].sum(1))[0]
        fy = np.clip(1-ky*(1-np.minimum(1, Pc/etm)), 0, 1); fc = np.clip(1-ky*(1-np.minimum(1, Pabs_ctl_s/etm)), 0, 1)
        m = np.where(fc > 0.05, np.minimum(1, fy/np.maximum(fc, 1e-9)), fy)
        out.append(float(np.sum(wg*(5*m < 0.01*2))))   # 1 年で種が 1/50 を割る
    return float(np.mean(out))
R['ctrl_false_barley_oneyear'] = ctrl_false()
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in R.items(): print(k, json.dumps(v, ensure_ascii=False))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
xs = np.linspace(0.05, 1, 50)
ax[0].plot(xs, xs, 'k--', lw=0.8, label='proportional'); ax[0].plot(xs, xs**g_w, label=f'wheat fit γ={g_w:.2f}'); ax[0].plot(xs, xs**g_p, label=f'potato fit γ={g_p:.2f}')
ax[0].scatter(*zip(*wheat_pts), c='C0'); ax[0].scatter(*zip(*pot_pts), c='C1'); ax[0].set_xlabel('light ratio'); ax[0].set_ylabel('yield ratio'); ax[0].legend(fontsize=8); ax[0].set_title('shading trials')
ax[1].hist([MB.min(0)[gain], MP.min(0)[gain]], bins=20, label=['barley', 'potato']); ax[1].axvline(0.2*0.01*10, color='k', lw=0.5)
ax[1].set_xlabel('worst-year yield multiplier m (water×light)'); ax[1].legend(fontsize=8); ax[1].set_title('gain cells')
labs = ['098 thr', '098 prop', 'main', 'light prop', 'pot sat', 'no ET scale', 'gmult5', 'gmult20', 'best', 'worst']
vals = [0.6454, 0.4335, main['q'], V['light_prop']['q'], V['light_sat_potato']['q'], V['no_et_scaling']['q'], V['gmult5']['q'], V['gmult20']['q'], V['best_case']['q'], V['worst_case']['q']]
ax[2].bar(range(len(vals)), vals, color=['gray', 'gray']+['C2']*8); ax[2].axhline(R['H111c']['midpoint'], color='k', ls='--', lw=0.8)
ax[2].set_xticks(range(len(vals)), labs, rotation=45, fontsize=8); ax[2].set_ylabel('per-group staple loss q')
plt.tight_layout(); plt.savefig('fig_crop_response.png', dpi=110)
