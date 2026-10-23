"""研究104：廃墟の品（ガラス・レンズ・工作機械）が段の人口の閾値 N_lens・N_T2e を下げる効き（open_problems 250）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（21:41 JST）。
057 の sim.py を「# ---------- 新しい乱数」の前まで exec し（056→…→038 の乱数をそのまま使う）、
T2 到達の式の wait(N_lens)・wait(N_T2e) を、品が寿命 L の内に下がった閾値へ届く標本でだけ wait(m·N) に置く。新しい乱数は seed 20261104。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P57 = os.path.abspath(os.path.join(HERE, "..", "057-pre-reader-knowledge", "sim.py"))
src = open(P57, encoding="utf8").read(); cut = src.index("# ---------- 新しい乱数")
G = {"__file__": P57, "__name__": "sim057"}
exec(compile(src[:cut], P57, "exec"), G)
assert G["R"]["check_T2s_reproduced"] == 0.0 and G["R"]["check_V_m_eq_V_ext"] == 0.0
N, B, SRC, COMPS, agg, V_m, q_TC, FIXED = (G[k] for k in ("N", "B", "SRC", "COMPS", "agg", "V_m", "q_TC", "FIXED"))
U_lit, d01, d12, lit_mult, d_Fe, W = (G[k] for k in ("U_lit", "d01", "d12", "lit_mult", "d_Fe", "W"))
N_Fe, N_lens, N_T2e, r_pop = (G[k] for k in ("N_Fe", "N_lens", "N_T2e", "r_pop"))
R = {"study": "104-ruin-glass-thresholds", "seed": 20261104}

def parts(k):
    st = B[k]["st"]; sv = SRC[k]["sv"]; lit = SRC[k]["lit"]
    n = np.maximum(sv["nmax"], 1.0)
    wait = lambda Nk: np.maximum(0.0, np.log(Nk / n) / r_pop)
    d01e = np.where(U_lit < lit, d01 * lit_mult, d01)
    tFe = np.where(st == 6, np.maximum(d_Fe, wait(N_Fe)), 0.0)
    return st, n, wait, d01e, tFe

def T2s_mod(k, m_l=None, L_l=np.inf, m_t=None, L_t=np.inf, ok=True):
    """m_l: N_lens の倍率（下限 N_Fe）。m_t: N_T2e の倍率。品が寿命内に下がった閾値へ届く標本でだけ効く。ok: 届く範囲に必要な数がある標本"""
    if k in FIXED: return B[k]["T2s"]
    st, n, wait, d01e, tFe = parts(k)
    wl = wait(N_lens); wt = wait(N_T2e)
    if m_l is not None:
        wl2 = wait(np.maximum(m_l * N_lens, N_Fe)); wl = np.where(ok & (wl2 <= L_l), wl2, wl)
    if m_t is not None:
        wt2 = wait(m_t * N_T2e); wt = np.where(ok & (wt2 <= L_t), wt2, wt)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wl), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12, wt), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [W, np.zeros(N), t2], default=np.zeros(N))
R["check_T2s_mod_eq_B"] = max(float(np.abs(T2s_mod(k) - B[k]["T2s"]).max()) for k in COMPS)
assert R["check_T2s_mod_eq_B"] == 0.0

# ---------- 入力（事実は sources/、推測は【推測】） ----------
g = np.random.default_rng(20261104)
def lu(lo, hi, n=N): return np.exp(g.uniform(np.log(lo), np.log(hi), n))
P_NEED = (0.25, 0.80)    # 【事実の帯】老視の有病率 世界 25%（2015）、都市 25〜80%（Wikipedia Presbyopia）。眼鏡を使う割合の数値は見つからず、要る割合を代わりに使う【推測】
rho = lu(10, 300)        # 届く範囲の元の人口密度（人/km2）、103 と同じ幅【推測】
REACH = 145.0            # 073 の Binford の中央値（km）
p_need = g.uniform(*P_NEED)
p_find = lu(1e-4, 1e-1)  # 見つけて割れずに使える割合【推測】
n_need = lu(1, 100)      # 段に要るレンズの数【推測】
g_glass = g.uniform(0.05, 0.30)   # 眼鏡のうちガラスの割合：H116b の資料は「樹脂が最も多い」「ガラスは減った」の記述のみ【推測】
L_glass = lu(500, 5000)  # ガラスの光学面が使える年数【推測】。ローマのガラスが約 2,000 年残る（Wikipedia Roman glass）ことと、アルカリの多いガラスは腐食する（Glass、Glass disease）ことの間
L_plast = lu(30, 300)    # 樹脂レンズ（CR-39・PC）の使える年数【推測】（資料に数値なし）
# 鋼の工作機械：103 の ISO 9223/9224 の式、覆いの下（C1 上端〜C2）と外（C2〜C4）
B1 = 0.523
def t_life(r1, th, b=B1):
    Dlim = th / 4; t_pw = (Dlim / r1) ** (1 / b); t_lin = 20 + (Dlim / r1 - 20 ** b) / (b * 20 ** (b - 1))
    return np.where(t_pw <= 20, t_pw, t_lin)
r_sh = lu(1.3, 13.0); r_out = lu(13.0, 80.0)
shel = g.uniform(0.2, 0.6)
U_sh = g.uniform(size=N)
r_tool = np.where(U_sh < shel, r_sh, r_out)
L_tool_bulk = t_life(r_tool, 20000.0)     # 20 mm の部材の半分の厚さ（103 の定義）
L_tool_prec = t_life(r_tool, 400.0)       # 精度の面：0.1 mm の腐食で精度を失う（th/4=100 µm）【推測】
m_l_main = lu(0.1, 1.0); m_t_main = lu(0.1, 1.0)

stock = rho * np.pi * REACH ** 2 * p_need * p_find   # 届く範囲で使えるレンズの数
ok_n = stock >= n_need
R["inputs"] = {"lens_stock_q": [float(np.quantile(stock, p)) for p in (0.05, 0.5, 0.95)], "share_stock_ge_need": float(ok_n.mean()),
               "L_tool_bulk_q": [float(np.quantile(L_tool_bulk, p)) for p in (0.05, 0.5, 0.95)],
               "L_tool_prec_q": [float(np.quantile(L_tool_prec, p)) for p in (0.05, 0.5, 0.95)]}
# レンズの寿命：届く範囲に必要な数のガラスがあればガラス、なければ樹脂
L_lens = np.where(stock * g_glass >= n_need, L_glass, L_plast)
R["inputs"]["share_glass_enough"] = float((stock * g_glass >= n_need).mean())

# ---------- 診断：重い崩壊で t2 を縛るのは何か ----------
diag = {}
for k in COMPS:
    if k in FIXED: continue
    st, n, wait, d01e, tFe = parts(k); m = st >= 4
    if not m.any(): continue
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(N_lens)), 0.0)
    diag[k] = {"n": int(m.sum()), "share_wait_T2e_binds_t2": float((wait(N_T2e) >= t1 + d12)[m].mean()),
               "share_wait_lens_binds_t1": float(((wait(N_lens) >= tFe + d01e) & (st >= 5))[m].mean()),
               "median_wait_T2e": float(np.median(wait(N_T2e)[m])), "median_t1_plus_d12": float(np.median((t1 + d12)[m]))}
R["diag_binding"] = diag

D0 = {k: V_m(k, q_TC, True, B[k]["T2s"]) - V_m(k, 0.3, False, B[k]["T2s"]) for k in COMPS}
v0 = agg({"x": D0}, anchors=("A_Ord", "A_XPT_S"))
def evaluate(tag, **kw):
    D = {}; b_all = []; n_all = []
    for k in COMPS:
        T2n = T2s_mod(k, **kw)
        D[k] = V_m(k, q_TC, True, T2n) - V_m(k, 0.3, False, T2n)
        if k not in FIXED:
            m = B[k]["st"] >= 4
            b_all.append(B[k]["T2s"][m]); n_all.append(T2n[m])
    v1 = agg({"x": D}, anchors=("A_Ord", "A_XPT_S"))
    out = {}
    for a, u in (("A_Ord", "U3"), ("A_XPT_S", "U2")):
        b_ = v0[a][u]["x"]; n_ = v1[a][u]["x"]
        out[f"{a}/{u}"] = {"F_total": n_["total"] / b_["total"], "F_heavy": n_["heavy"] / b_["heavy"]}
    b6 = np.concatenate(b_all); n6 = np.concatenate(n_all)
    out["heavy_st4to6"] = {"T2_median_base": float(np.median(b6)), "T2_median_new": float(np.median(n6)),
                           "T2_median_rel_change": float(np.median(n6) / np.median(b6) - 1),
                           "share_saving_gt_10pct": float(np.mean((b6 - n6) > 0.1 * np.maximum(b6, 1e-9)))}
    R.setdefault("runs", {})[tag] = out
    return out

INF = np.inf
evaluate("lens_upper_to_NFe", m_l=0.0)                         # N_lens→N_Fe、寿命無限、数の条件なし
evaluate("lens_m0.3_inf", m_l=0.3)
evaluate("lens_main", m_l=m_l_main, L_l=L_lens, ok=ok_n)
evaluate("lens_plastic_only", m_l=m_l_main, L_l=L_plast, ok=ok_n)
evaluate("t2_m0.3_inf", m_t=0.3)
evaluate("both_m0.3_inf", m_l=0.3, m_t=0.3)
evaluate("both_m0.3_life_bulk", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_tool_bulk, ok=ok_n)
evaluate("both_m0.3_life_prec", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_tool_prec, ok=ok_n)
evaluate("both_main_life_bulk", m_l=m_l_main, L_l=L_lens, m_t=m_t_main, L_t=L_tool_bulk, ok=ok_n)
evaluate("both_main_life_prec", m_l=m_l_main, L_l=L_lens, m_t=m_t_main, L_t=L_tool_prec, ok=ok_n)
evaluate("both_m0.1_inf", m_l=0.1, m_t=0.1)
evaluate("both_m0.1_life_bulk", m_l=0.1, L_l=L_lens, m_t=0.1, L_t=L_tool_bulk, ok=ok_n)
evaluate("t2_m0.3_life_bulk", m_t=0.3, L_t=L_tool_bulk, ok=ok_n)

RU = R["runs"]; K = "A_Ord/U3"
# ---------- 仮説 ----------
H = {}
H["H116a"] = {"class": "confirmatory", "value": None, "rule": "adult corrective-lens use >= 50% in a high-income country (public source)",
              "supported": None, "note": "開いた資料に使う割合の数値なし（検定できず）。要る割合（老視 25%、都市 25〜80%、近視 欧米 30〜40%）は得た"}
H["H116b"] = {"class": "confirmatory", "value": "qualitative", "rule": "glass < 30% of spectacle lenses",
              "supported": True, "note": "数値なし。Corrective lens：'Plastic lenses are currently the most commonly prescribed lens' と 'Glass lenses have become less common'。記述での支持（弱い）"}
v = RU["lens_upper_to_NFe"]; 
H["H116c"] = {"class": "prediction", "value": {"F_heavy": v[K]["F_heavy"], "T2_med_change": v["heavy_st4to6"]["T2_median_rel_change"]},
              "rule": "lens path upper: F_heavy-1 < 0.05 and |T2 median change| < 0.10",
              "supported": bool(v[K]["F_heavy"] - 1 < 0.05 and abs(v["heavy_st4to6"]["T2_median_rel_change"]) < 0.10)}
fb = RU["both_m0.3_inf"][K]["F_heavy"]; ft = RU["t2_m0.3_inf"][K]["F_heavy"]
H["H116d"] = {"class": "prediction", "value": {"F_heavy_both": fb, "F_heavy_t2only": ft, "share_from_T2e": (ft - 1) / (fb - 1)},
              "rule": "both m=0.3 inf: F_heavy-1 > 0.10 and >=80% of gain from N_T2e", "supported": bool(fb - 1 > 0.10 and (ft - 1) / (fb - 1) >= 0.8)}
fl = RU["both_m0.3_life_bulk"][K]["F_heavy"]; fp = RU["both_m0.3_life_prec"][K]["F_heavy"]
H["H116e"] = {"class": "prediction", "value": {"ratio_bulk": (fl - 1) / (fb - 1), "ratio_prec": (fp - 1) / (fb - 1)},
              "rule": "with lifetimes (bulk steel) the gain < 50% of infinite-life gain", "supported": bool((fl - 1) / (fb - 1) < 0.5)}
R["hypotheses"] = H

# ---------- 図 ----------
names = list(RU)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
ax[0].barh(names, [RU[n_][K]["F_heavy"] for n_ in names], color=["C0" if n_.startswith("lens") else "C1" if n_.startswith("t2") else "C2" for n_ in names])
ax[0].axvline(1, color="k", lw=0.8); ax[0].set_xlabel("F_heavy (A_Ord/U3) vs 057 base"); ax[0].set_title("104: ruin optics/tools lowering stage thresholds", fontsize=9)
ax[0].tick_params(axis="y", labelsize=7)
ks = list(diag); ax[1].bar(ks, [diag[k]["share_wait_T2e_binds_t2"] for k in ks])
ax[1].set_ylabel("share of heavy-collapse samples (st>=T1-) where wait(N_T2e) binds t2"); ax[1].set_ylim(0, 1)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_ruin_glass.png"), dpi=120)

R["runtime_s"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in R if k != "runs"}, indent=1, ensure_ascii=False, default=float))
for n_, r in RU.items():
    print(f"{n_:24s} F_heavy {r[K]['F_heavy']:.4f} F_total {r[K]['F_total']:.4f} | XPT {r['A_XPT_S/U2']['F_heavy']:.4f} | T2med {r['heavy_st4to6']['T2_median_base']:.0f}->{r['heavy_st4to6']['T2_median_new']:.0f} ({r['heavy_st4to6']['T2_median_rel_change']:+.3f})")
