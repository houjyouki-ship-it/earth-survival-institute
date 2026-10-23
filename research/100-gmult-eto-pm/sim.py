"""研究100：耐寒の作物の増殖比（241）と核の冬の蒸発の需要（242、FAO 56 の Penman-Monteith）で 099 の q を置き直す。
入力：099 の計算（そのまま再実行、098・097・096 の差0の assert を含む）、NWDC の OSF（https://osf.io/e28gq/）の CESM の月の TS
（097 で取得した nw_ur_150_07・nw_cntrl_03 の cam.h0）、098 の FSDS の国の月平均。
資料【事実は report.md の表と sources/】：増殖比は Wikipedia（Agriculture in the Middle Ages、Crop yield）、AICPMIP と APSSCA の種子の基準、
FAO i1127e（種芋 2〜3 t/ha）、Wolaita の調査（Agric & Food Secur 2017, 6:43）、Shibe & Anbes（Debre Birhan の麦芽大麦）。
FAO 56 の式：https://www.fao.org/4/x0490e/x0490e00.htm （式 6・7・8・11・13・21〜25・37〜40、欠測の手順：露点＝Tmin、u2＝2 m/s）。
注：出す値は模型の中の割合で、生存確率でも復興確率でもない。乱数は 099 と同じ（seed 96、同じ順）。"""
import os, json, numpy as np, netCDF4 as nc
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ME = os.path.dirname(os.path.abspath(__file__)); D99 = os.path.join(ME, '../099-crop-response-shape')
s99 = open(os.path.join(D99, 'sim.py')).read()
s99 = s99.replace("ME = os.path.dirname(os.path.abspath(__file__))", f"ME = {D99!r}")
exec(s99[:s99.rindex("json.dump(R, open('results.json'")])
ME = os.path.dirname(os.path.abspath(__file__)); os.chdir(ME)
R99 = R; R = {}
R['repro099'] = dict(q_main=R99['q_main']['q'], q_no_et=R99['versions']['no_et_scaling']['q'])
assert abs(R['repro099']['q_main']-0.46277) < 1e-5 and abs(R['repro099']['q_no_et']-0.50671) < 1e-5, R['repro099']
print('repro099', R['repro099'])

# ---------------- 242：FAO 56 の Penman-Monteith の ETo ----------------
ND = '/workspace/nwdata/scratch/alpine/joco6825/dwinstead_data/'
def load_x(run, years, v):
    A = []
    for y in years:
        for m in range(1, 13):
            d = nc.Dataset(f'{ND}{run}/{run}.cam.h0.{y:04d}-{m:02d}.nc'); A.append(d[v][0].astype(np.float32)); d.close()
    return np.array(A).reshape(len(years), 12, 96, 144)-273.15
# 【事実：ファイルの属性】150 Tg の TSMX・TSMN は「出力期間（月）の最大・最小」（cell_methods: maximum）で日の最高・最低の月平均ではなく、
# 対照の TSMX・TSMN は cell_methods: mean で TS と同じ値（差 0）。よって日較差は資料から取れず、登録の Tmax/Tmin＝TSMX/TSMN は使えない（逸脱）。
# 代わりに TS（地表の温度、097 の T150・TCT）に日較差 DTR を置く：主は両方 10 °C、版は 6・14 °C、DTR を季節の日射の比に比例（150 Tg のみ縮む）。
def attr_check():
    d = nc.Dataset(f'{ND}nw_cntrl_03/nw_cntrl_03.cam.h0.0005-07.nc'); c = d['TSMX'].cell_methods; dx = float(np.abs(np.array(d['TSMX'][0])-np.array(d['TSMN'][0])).max()); d.close()
    d = nc.Dataset(f'{ND}nw_ur_150_07/nw_ur_150_07.cam.h0.0006-07.nc'); c2 = d['TSMX'].cell_methods; d.close(); return dict(ctl_cell_methods=c, ctl_max_abs_TSMX_minus_TSMN=dx, w150_cell_methods=c2)
R['tsmx_attr'] = attr_check()
TS150c = T150[..., ii, jj]; TSCTc = TCT[..., ii, jj]                       # (years, 12, ncell) °C
sea = lambda a: a[:, SEAS]                                                  # 4 か月
latc = np.deg2rad(lat[ii]); MIDDOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
def Ra_month(k):   # 作物年の月 k の月中の日の Ra（MJ/m2/d）、FAO 56 式 21〜25
    cal = np.where(sh_c, (k+6) % 12, k); J = MIDDOY[cal]
    dr = 1+0.033*np.cos(2*np.pi*J/365); dl = 0.409*np.sin(2*np.pi*J/365-1.39)
    ws = np.arccos(np.clip(-np.tan(latc)*np.tan(dl), -1, 1))
    return 24*60/np.pi*0.0820*dr*(ws*np.sin(latc)*np.sin(dl)+np.cos(latc)*np.cos(dl)*np.sin(ws))
RA = np.array([Ra_month(k) for k in range(12)])[SEAS]                       # (4, ncell)
e0 = lambda T: 0.6108*np.exp(17.27*T/(T+237.3))
def eto(Tx, Tn, Rs_W, u2=2.0, dew_off=0.0):
    Tx = np.maximum(Tx, Tn); T = (Tx+Tn)/2; es = (e0(Tx)+e0(Tn))/2; ea = e0(Tn-dew_off)
    D = 4098*e0(T)/(T+237.3)**2; g = 0.000665*101.3
    Rs = Rs_W*0.0864; Rso = 0.75*RA; Rns = 0.77*Rs
    rel = np.clip(Rs/np.maximum(Rso, 1e-6), 0.25, 1.0)
    Rnl = 4.903e-9*((Tx+273.16)**4+(Tn+273.16)**4)/2*(0.34-0.14*np.sqrt(ea))*(1.35*rel-0.35)
    Rn = Rns-Rnl
    E = (0.408*D*Rn+g*900/(T+273)*u2*(es-ea))/(D+g*(1+0.34*u2))
    return np.maximum(E, 0.0)
# 月の FSDS（国の平均）を作物年・格子へ
FS150 = np.full((5, 4, len(ii)), np.nan); FSCT = np.full((20, 4, len(ii)), np.nan)
for k in range(len(ii)):
    nm = names[ccell[k]]
    if nm is None: continue
    FS150[:, :, k] = fs_crop(f150[[F150.index.get_loc(nm)]], 5, sh_c[k])[0][:, SEAS]; FSCT[:, :, k] = fs_crop(fct[[FCT.index.get_loc(nm)]], 20, sh_c[k])[0][:, SEAS]
def eto_ratio(dtr=10.0, dtr_light=False, **kw):
    d150 = dtr*np.minimum(1, FS150/np.nanmean(FSCT, 0)) if dtr_light else dtr
    t1, t0 = sea(TS150c), sea(TSCTc)
    e150 = eto(t1+d150/2, t1-d150/2, FS150, **kw).mean(1); ect = eto(t0+dtr/2, t0-dtr/2, FSCT, **kw).mean(1)
    r = e150/np.where(ect.mean(0) > 0, ect.mean(0), np.nan)
    return np.nan_to_num(r, nan=1.0), e150, ect
ER, E150, ECT = eto_ratio()
okg = gain & np.isfinite(FS150[0, 0])
R['H112c'] = dict(median_ETo_ratio_y2_gain=float(np.median(ER[1][okg])), median_Lr_s_y2_gain=float(np.median(Lrs[1][okg])),
                  median_ETo_ratio_by_year=[float(np.median(ER[y][okg])) for y in range(5)], median_Lr_by_year=[float(np.median(Lrs[y][okg])) for y in range(5)],
                  frac_gain_ETo_ratio_lt_Lr_y2=float(np.sum(wg*(okg & (ER[1] < Lrs[1])))/np.sum(wg*okg)))
R['desc_eto'] = dict(median_ctl_ETo_mm_d=float(np.median(ECT.mean(0)[okg])), median_150_ETo_mm_d_y2=float(np.median(E150[1][okg])),
                     median_ctl_Tseason=float(np.median(sea(TSCTc).mean((0, 1))[okg])), median_150_Tseason_y2=float(np.median(sea(TS150c)[1].mean(0)[okg])),
                     frac_gain_light_matched=float(np.sum(wg*okg)))
# 校正：対照の ETo を FAO 56 の値の幅と並べる（記述）
R['desc_eto']['ctl_ETo_q10_q90'] = [float(np.percentile(ECT.mean(0)[okg], 10)), float(np.percentile(ECT.mean(0)[okg], 90))]

# ---------------- 241：増殖比の資料 ----------------
# 【事実】各値は sources/ の頁で確認したもの（収量÷播種量、重さ）。cls: premodern / modern_lowinput / seed_norm / trial_highinput
GB_DATA = [('Sussex barley 1350-99 (Brandon 1972 via WP)', 4.01, 'premodern'), ('Europe 9th c. cereals (WP Crop yield)', 2.5, 'premodern'),
           ('Low Countries 14th c. cereals (WP Crop yield)', 14.0, 'premodern'), ('Sussex wheat 1350-99', 4.34, 'premodern_other'),
           ('Sussex oats 1350-99', 2.87, 'premodern_other'), ('Debre Birhan malt barley 100 kg/ha, high input', 39.0, 'trial_highinput'),
           ('AICPMIP wheat norm', 20.0, 'seed_norm'), ('AICPMIP oat norm', 15.0, 'seed_norm')]
GP_DATA = [('AICPMIP/APSSCA potato norm 3 t -> 12.5 t', 125/30, 'seed_norm'), ('Ethiopia national 13.7 t / 2 t rec.', 13.7/2, 'modern_lowinput'),
           ('Wolaita 11.5-17.2 t / 1.59 t (mid)', 14.35/1.59, 'modern_lowinput'), ('Tigray 9.3 t / 2 t', 9.3/2, 'modern_lowinput'),
           ('S. Ethiopia improved 16.6 t / 2 t', 16.6/2, 'modern_lowinput'), ('S. Ethiopia local 2.5 t / 2 t', 2.5/2, 'modern_lowinput'),
           ('FAO typical seed 2-3 t with Ethiopia national 13.7 t', 13.7/2.5, 'modern_lowinput')]
gb_pre = [v for _, v, c in GB_DATA if c == 'premodern']; gp_low = [v for _, v, c in GP_DATA if c in ('modern_lowinput', 'seed_norm')]
R['H112a'] = dict(premodern_barley_or_cereal=gb_pre, median=float(np.median(gb_pre)), barley_only=4.01, all_values={n: v for n, v, c in GB_DATA})
R['H112b'] = dict(potato_values=gp_low, median=float(np.median(gp_low)), q25_q75=[float(np.percentile(gp_low, 25)), float(np.percentile(gp_low, 75))],
                  all_values={n: v for n, v, c in GP_DATA})

# ---------------- q の置き直し ----------------
def q_new(gb=10.0, gp=10.0, et='pm', ky_b=1.15, ky_p=1.1, et_b=550, et_p=600, light='fit', f=0.5, w=wA, TT=TA, ERx=None):
    global Lrs
    keep = Lrs
    if et == 'pm': Lrs = ERx if ERx is not None else ER      # water_ratio は Lrs で ETm を縮める
    wb = water_ratio(ky_b, et_b, et != 'none')[0]; wp = water_ratio(ky_p, et_p, et != 'none')[0]
    Lrs = keep
    mb = wb*light_ratio('barley', light); mp = wp*light_ratio('potato', light)
    lb = seed_loss(np.where(BOK, gb*mb, 0.0), 1.0, f, 0.01, 5, 0.0); lp = seed_loss(np.where(POK, gp*mp, 0.0), 1.0, f, 0.01, 5, 0.0)
    return summ((switch_lost(Y) & lb & lp).astype(float), w, TT)
chk = q_new(et='fsds'); assert abs(chk['q']-R['repro099']['q_main']) < 1e-12   # 099 の主を同じ関数で再現
GBm, GPm = R['H112a']['median'], R['H112b']['median']
V = {}
V['099_main_fsds_g10'] = chk['q']
V['pm_g10'] = q_new()['q']
V['fsds_gsrc'] = q_new(gb=GBm, gp=GPm, et='fsds')['q']
V['pm_gsrc_MAIN'] = q_new(gb=GBm, gp=GPm)['q']
V['pm_gsrc_popw'] = q_new(gb=GBm, gp=GPm, w=wP, TT=TP)['q']
V['pm_gsrc_sameSubregion'] = q_new(gb=GBm, gp=GPm)
for u in (1.0, 4.0): V[f'pm_u{u:g}_gsrc'] = q_new(gb=GBm, gp=GPm, ERx=eto_ratio(u2=u)[0])['q']
for dd in (6.0, 14.0): V[f'pm_dtr{dd:g}_gsrc'] = q_new(gb=GBm, gp=GPm, ERx=eto_ratio(dtr=dd)[0])['q']
ERL = eto_ratio(dtr_light=True)[0]; V['pm_dtr_light_gsrc'] = q_new(gb=GBm, gp=GPm, ERx=ERL)['q']
R['H112c']['median_ETo_ratio_y2_dtr_light'] = float(np.median(ERL[1][okg]))
V['pm_dew_minus2_gsrc'] = q_new(gb=GBm, gp=GPm, ERx=eto_ratio(dew_off=2.0)[0])['q']
ERmean = np.repeat(ER.mean(0, keepdims=True), 5, 0); V['pm_5yr_mean_ratio_gsrc'] = q_new(gb=GBm, gp=GPm, ERx=ERmean)['q']
V['pm_g_lo_2.5_1.25'] = q_new(gb=2.5, gp=1.25)['q']; V['pm_g_hi_14_9'] = q_new(gb=14.0, gp=9.0)['q']
V['pm_g_modern_39_9'] = q_new(gb=39.0, gp=9.0)['q']
V['pm_gsrc_f075'] = q_new(gb=GBm, gp=GPm, f=0.75)['q']                     # 登録外：飢えても種を多く残す
V['none_et_gsrc'] = q_new(gb=GBm, gp=GPm, et='none')['q']
R['versions'] = V
R['H112d'] = dict(q=V['pm_gsrc_MAIN'], range099=[0.445, 0.579], inside=bool(0.445 <= V['pm_gsrc_MAIN'] <= 0.579))
# 平時の偽陽性（登録外の確認）：対照の 20 年から連続 5 年の窓 16 個で同じ規則を回す（光 1、ETo は対照の年の比）
def ctrl_false5(gb, gp, f=0.5):
    out = []
    Pc = cells(PCTy[:, SEAS].sum(1)); erc = ECT/np.where(ECT.mean(0) > 0, ECT.mean(0), np.nan); erc = np.nan_to_num(erc, nan=1.0)
    def wr(ky, etm, y):
        et = etm*erc[y]; fy = np.clip(1-ky*(1-np.minimum(1, Pc[y]/et)), 0, 1); fc = np.clip(1-ky*(1-np.minimum(1, Pabs_ctl_s/etm)), 0, 1)
        return np.where(fc > 0.05, np.minimum(1, fy/np.maximum(fc, 1e-9)), fy)
    for s0 in range(16):
        mb = np.array([wr(1.15, 550, s0+t) for t in range(5)]); mp = np.array([wr(1.1, 600, s0+t) for t in range(5)])
        lb = seed_loss(gb*mb, 1.0, f, 0.01, 5, 0.0); lp = seed_loss(gp*mp, 1.0, f, 0.01, 5, 0.0)
        out.append(float(np.sum(wg*(lb & lp))))
    return float(np.mean(out))
R['ctrl_false_5yr'] = dict(g10=ctrl_false5(10, 10), gsrc=ctrl_false5(GBm, GPm), g_lo=ctrl_false5(2.5, 1.25))
json.dump(R, open('results.json', 'w'), indent=1, ensure_ascii=False)
for k, v in R.items(): print(k, json.dumps(v, ensure_ascii=False))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
ax[0].scatter(Lrs[1][okg], ER[1][okg], s=6); ax[0].plot([0, 1.1], [0, 1.1], 'k--', lw=0.8)
ax[0].set_xlabel('season FSDS ratio L_r (year 2)'); ax[0].set_ylabel('season PM ETo ratio (year 2)'); ax[0].set_title('gain cells: ETo vs light')
nb = [n for n, v, c in GB_DATA]; ax[1].barh(range(len(GB_DATA)), [v for _, v, _ in GB_DATA], color='C0', label='cereal')
ax[1].barh(range(len(GB_DATA), len(GB_DATA)+len(GP_DATA)), [v for _, v, _ in GP_DATA], color='C1', label='potato (weight)')
ax[1].set_yticks(range(len(GB_DATA)+len(GP_DATA)), nb+[n for n, v, c in GP_DATA], fontsize=6); ax[1].axvline(10, color='k', ls='--', lw=0.8); ax[1].set_xscale('log')
ax[1].set_xlabel('harvest / seed'); ax[1].legend(fontsize=7)
labs = ['099 main', 'PM g10', 'FSDS gsrc', 'PM gsrc (main)', 'u1', 'u4', 'dew-2', 'g lo', 'g hi', 'g modern', 'f0.75', 'no ET']
vals = [V['099_main_fsds_g10'], V['pm_g10'], V['fsds_gsrc'], V['pm_gsrc_MAIN'], V['pm_u1_gsrc'], V['pm_u4_gsrc'], V['pm_dew_minus2_gsrc'],
        V['pm_g_lo_2.5_1.25'], V['pm_g_hi_14_9'], V['pm_g_modern_39_9'], V['pm_gsrc_f075'], V['none_et_gsrc']]
ax[2].bar(range(len(vals)), vals, color=['gray']+['C2']*11); ax[2].axhspan(0.445, 0.579, color='gray', alpha=0.15)
ax[2].set_xticks(range(len(vals)), labs, rotation=45, fontsize=8); ax[2].set_ylabel('per-group staple loss q')
plt.tight_layout(); plt.savefig('fig_gmult_eto.png', dpi=110)
