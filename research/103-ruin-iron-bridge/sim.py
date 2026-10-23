"""研究103：重い崩壊の後、廃墟の鉄・鋼で製鉄の段を飛ばせる年数（open_problems 249）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
057 の sim.py を「# ---------- 新しい乱数」の前まで exec し（056→…→038 の乱数をそのまま使う）、
038 の times() の tFe=max(d_Fe, wait(N_Fe)) を、廃墟の鉄が tFe まで持つ標本で 0 に置く。新しい乱数は seed 20261103。
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
N, B, SRC, COMPS, agg, V_m, V_ext, q_TC, FIXED = (G[k] for k in ("N", "B", "SRC", "COMPS", "agg", "V_m", "V_ext", "q_TC", "FIXED"))
SCEN, U_lit, d01, d12, lit_mult, d_Fe, W = (G[k] for k in ("SCEN", "U_lit", "d01", "d12", "lit_mult", "d_Fe", "W"))
N_Fe, N_lens, N_T2e, r_pop = (G[k] for k in ("N_Fe", "N_lens", "N_T2e", "r_pop"))
R = {"study": "103-ruin-iron-bridge", "seed": 20261103}

def parts(k):
    st = B[k]["st"]; sv = SRC[k]["sv"]; lit = SRC[k]["lit"]
    n = np.maximum(sv["nmax"], 1.0)
    wait = lambda Nk: np.maximum(0.0, np.log(Nk / n) / r_pop)
    d01e = np.where(U_lit < lit, d01 * lit_mult, d01)
    tFe = np.where(st == 6, np.maximum(d_Fe, wait(N_Fe)), 0.0)
    return st, n, wait, d01e, tFe

def T2s_scrap(k, bridge=None):
    if k in FIXED: return B[k]["T2s"]
    st, n, wait, d01e, tFe = parts(k)
    if bridge is not None: tFe = np.where(bridge, 0.0, tFe)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(N_lens)), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12, wait(N_T2e)), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [W, np.zeros(N), t2], default=np.zeros(N))
R["check_T2s_scrap_eq_B"] = max(float(np.abs(T2s_scrap(k) - B[k]["T2s"]).max()) for k in COMPS)
assert R["check_T2s_scrap_eq_B"] == 0.0

# ---------- 入力（事実は sources/、推測は【推測】） ----------
PC_DEV = 7085.0          # 【事実】IRP Metal Stocks in Society：先進国の1人あたり使用中の鋼 7,085 kg（Wikipedia Scrap）
DEV_FACTOR = (5, 10)     # 【事実】先進国は途上国の 5〜10 倍（IRP の報告の頁）
C_MAIN = 1.5             # 【事実】Roman Britain の鉄の生産 1.5 kg/人/年の推計（Wikipedia Roman metallurgy）
C_LO, C_HI = 0.1, 5.0    # 【事実】Han 約 0.1 kg/人（5,000 t/5千万人、同頁）、Saugus の能力 4.7〜5.5 kg/人（046）
B1, B2 = 0.523, 0.575    # 【事実】ISO 9224 の炭素鋼の時間の指数
R_OUT = (13.0, 80.0)     # 【事実の帯】ISO 9223 の初年の腐食 C2 の中〜C4 上端（µm/年）。帯の中の分布は【推測】
R_SHEL = (1.3, 13.0)     # 屋内・覆いの下：C1 上端〜C2 の中【推測】
REACH = 145.0            # 073：Binford の狩猟採集民の年の移動の中央値（km）

g = np.random.default_rng(20261103)
def lu(lo, hi, n=N): return np.exp(g.uniform(np.log(lo), np.log(hi), n))
pc = lu(PC_DEV / DEV_FACTOR[1], PC_DEV)              # 1人あたりの在庫（kg）、途上国〜先進国
rho = lu(10, 300)                                     # 届く範囲の元の人口密度（人/km2）【推測】
acc = lu(0.01, 0.3)                                   # 掘り出せて使える割合（瓦礫・コンクリートの中・火事）【推測】
w_thin = g.uniform(0.1, 0.4); w_med = g.uniform(0.2, 0.4); w_thick = 1 - w_thin - w_med   # 厚さの構成【推測】
TH = np.array([1.0, 5.0, 20.0]) * 1000                # µm（薄板・中・厚い形鋼／レール）
shel = g.uniform(0.2, 0.6)                            # 覆いの下にある割合【推測】
r_out = lu(*R_OUT); r_sh = lu(*R_SHEL)
c_pp = lu(C_LO, C_HI)

def D_iso(t, r1, b, linear=True):
    t = np.asarray(t, float)
    pw = r1 * t ** b
    if not linear: return pw
    return np.where(t <= 20, pw, r1 * (20 ** b + b * 20 ** (b - 1) * (t - 20)))

def t_life(r1, th, b, linear=True):
    """両面から腐食し、厚さの半分を失う（片面 th/4）までの年数"""
    Dlim = th / 4
    if linear:
        t_pw = (Dlim / r1) ** (1 / b)
        t_lin = 20 + (Dlim / r1 - 20 ** b) / (b * 20 ** (b - 1))
        return np.where(t_pw <= 20, t_pw, t_lin)
    return (Dlim / r1) ** (1 / b)

def scrap_time(k, b=B1, linear=True, acc_m=1.0, c_m=1.0, reach=REACH, c_fix=None, thick_ok=True):
    """T_scrap：使える在庫が尽きる（腐食で寿命が切れる、または使い切る）時刻。製鉄を始める前は人口は n0·e^{rt} で N_Fe を超えない"""
    st, n, wait, d01e, tFe = parts(k)
    stock0 = pc * rho * np.pi * reach ** 2 * acc * acc_m    # kg
    ws = [w_thin, w_med, w_thick if thick_ok else 0 * w_thick]
    lives = []
    for th in TH:
        lives.append((shel * 0 + t_life(r_out, th, b, linear), t_life(r_sh, th, b, linear)))
    tgrid = np.concatenate([[0], np.logspace(-1, np.log10(20000), 240)])
    c = (C_MAIN if c_fix else c_pp) * c_m
    Tend = np.full(N, tgrid[-1])
    alive_prev = np.ones(N, bool)
    out = np.full(N, np.nan)
    cum = np.zeros(N); tprev = 0.0
    for t in tgrid[1:]:
        S = np.zeros(N)
        for w, (lo, ls) in zip(ws, lives):
            S += stock0 * w * ((1 - shel) * (lo > t) + shel * (ls > t))
        nt = np.minimum(n * np.exp(r_pop * t), np.maximum(N_Fe, n))
        cum += c * nt * (t - tprev); tprev = t
        dead = (cum >= S) & np.isnan(out)
        out = np.where(dead, t, out)
    out = np.where(np.isnan(out), tgrid[-1], out)
    return out, stock0, tFe, st, n

def evaluate(tag, **kw):
    D = {}; info = {}
    T6 = []; TF = []; BR = []; dT2 = []; base6 = []; new6 = []
    for k in COMPS:
        if k in FIXED:
            D[k] = V_m(k, q_TC, True, B[k]["T2s"]) - V_m(k, 0.3, False, B[k]["T2s"]); continue
        ts, stock0, tFe, st, n = scrap_time(k, **kw)
        br = (st == 6) & (ts >= tFe)
        T2n = T2s_scrap(k, br)
        D[k] = V_m(k, q_TC, True, T2n) - V_m(k, 0.3, False, T2n)
        m6 = st == 6
        if m6.any():
            T6.append(ts[m6]); TF.append(tFe[m6]); BR.append(br[m6])
            base6.append(B[k]["T2s"][m6]); new6.append(T2n[m6])
        info[k] = {"share_st6": float(m6.mean()), "bridge_share_of_st6": float(br[m6].mean()) if m6.any() else None}
    D0 = {k: V_m(k, q_TC, True, B[k]["T2s"]) - V_m(k, 0.3, False, B[k]["T2s"]) for k in COMPS}
    v1 = agg({"x": D}, anchors=("A_Ord", "A_XPT_S")); v0 = agg({"x": D0}, anchors=("A_Ord", "A_XPT_S"))
    out = {"by_comp": info}
    for a, u in (("A_Ord", "U3"), ("A_XPT_S", "U2")):
        b_ = v0[a][u]["x"]; n_ = v1[a][u]["x"]
        out[f"{a}/{u}"] = {"F_total": n_["total"] / b_["total"], "F_heavy": n_["heavy"] / b_["heavy"] if b_["heavy"] > 0 else None,
                           "heavy_share_base": b_["heavy"] / b_["total"]}
    if T6:
        T6 = np.concatenate(T6); TF = np.concatenate(TF); BR = np.concatenate(BR); b6 = np.concatenate(base6); n6 = np.concatenate(new6)
        fin = b6 < 1e8
        out["st6"] = {"n": int(T6.size), "T_scrap_q": [float(np.quantile(T6, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)],
                      "tFe_q": [float(np.quantile(TF, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)],
                      "bridge_share": float(BR.mean()),
                      "T2_median_base": float(np.median(b6)), "T2_median_new": float(np.median(n6)),
                      "T2_median_rel_change": float(np.median(n6) / np.median(b6) - 1),
                      "T2_mean_saving_years": float(np.mean(b6 - n6)),
                      "share_T2_saving_gt_10pct": float(np.mean((b6 - n6) > 0.1 * np.maximum(b6, 1e-9)))}
        if tag == "main": R["_arrays"] = (T6, TF, BR, b6, n6)
    R.setdefault("runs", {})[tag] = out
    return out

# 量と腐食の切り分け（H115a・b）
k_ref = [k for k in COMPS if k not in FIXED and (B[k]["st"] == 6).any()]
R["comps_with_st6"] = k_ref
st_all = np.concatenate([B[k]["st"] for k in k_ref]); 
# 量：届く在庫 ÷ 1年の需要（n0, c）
yrs = []
for k in k_ref:
    st, n, wait, d01e, tFe = parts(k); m = st == 6
    stock0 = pc * rho * np.pi * REACH ** 2 * acc
    yrs.append((stock0 / (c_pp * np.maximum(N_Fe, n)))[m])
yrs = np.concatenate(yrs)
R["H_inputs"] = {"years_of_demand_at_N_Fe_q": [float(np.quantile(yrs, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]}
# 腐食：使える量（外と覆いの下の重み）の半分が失われる時刻（C3 の中 37.5 µm、覆いの下 0.4、厚さの構成の中央）
tt = np.logspace(0, 4, 4000)
def usable_frac(t, r1o, r1s, sh, w, b=B1):
    s = 0
    for wi, th in zip(w, TH):
        s += wi * ((1 - sh) * (t_life(r1o, th, b) > t) + sh * (t_life(r1s, th, b) > t))
    return s
uf = np.array([usable_frac(t, 37.5, np.sqrt(1.3 * 13), 0.4, (0.25, 0.3, 0.45)) for t in tt])
R["H_inputs"]["C3_half_usable_years"] = float(tt[np.argmax(uf <= 0.5)])
R["H_inputs"]["life_years_C3_outdoor"] = {f"{int(th/1000)}mm": float(t_life(37.5, th, B1)) for th in TH}
R["H_inputs"]["life_years_C3_outdoor_B2"] = {f"{int(th/1000)}mm": float(t_life(37.5, th, B2)) for th in TH}
R["H_inputs"]["life_years_C3_outdoor_powerlaw"] = {f"{int(th/1000)}mm": float(t_life(37.5, th, B1, False)) for th in TH}
R["H_inputs"]["life_years_C5_outdoor"] = {f"{int(th/1000)}mm": float(t_life(140, th, B1)) for th in TH}

main = evaluate("main")
evaluate("B2", b=B2)
evaluate("powerlaw_only", linear=False)
evaluate("acc_x0.1", acc_m=0.1)
evaluate("c_x3", c_m=3.0)
evaluate("reach_50km", reach=50.0)
evaluate("no_thick", thick_ok=False)
evaluate("worst", b=B2, acc_m=0.1, c_m=3.0, reach=50.0, thick_ok=False)

# 上限：すべての st6 で tFe=0（廃墟の鉄が無限に持つ）
D = {}
for k in COMPS:
    if k in FIXED: D[k] = V_m(k, q_TC, True, B[k]["T2s"]) - V_m(k, 0.3, False, B[k]["T2s"]); continue
    st = B[k]["st"]; T2n = T2s_scrap(k, st == 6)
    D[k] = V_m(k, q_TC, True, T2n) - V_m(k, 0.3, False, T2n)
D0 = {k: V_m(k, q_TC, True, B[k]["T2s"]) - V_m(k, 0.3, False, B[k]["T2s"]) for k in COMPS}
v1 = agg({"x": D}, anchors=("A_Ord", "A_XPT_S")); v0 = agg({"x": D0}, anchors=("A_Ord", "A_XPT_S"))
R["upper_bound_all_bridge"] = {f"{a}/{u}": {"F_total": v1[a][u]["x"]["total"] / v0[a][u]["x"]["total"],
                                             "F_heavy": v1[a][u]["x"]["heavy"] / v0[a][u]["x"]["heavy"]} for a, u in (("A_Ord", "U3"), ("A_XPT_S", "U2"))}

# なぜ効かないか：st6 で t1 を決めているのは wait(N_lens) か tFe+d01e か
bind = []
for k in k_ref:
    st, n, wait, d01e, tFe = parts(k); m = st == 6
    bind.append(((tFe + d01e) > wait(N_lens))[m])
R["st6_share_iron_path_binding_t1"] = float(np.concatenate(bind).mean())

# ---------- 仮説 ----------
H = {}
v = R["H_inputs"]["years_of_demand_at_N_Fe_q"][2]
H["H115a"] = {"class": "confirmatory", "value": v, "rule": "median stock/annual demand (at N_Fe) > 1000 years", "supported": bool(v > 1000)}
v = R["H_inputs"]["C3_half_usable_years"]
H["H115b"] = {"class": "confirmatory", "value": v, "rule": "C3: half of usable mass lost after > 100 years", "supported": bool(v > 100)}
v = main["st6"]["bridge_share"]
H["H115c"] = {"class": "prediction", "value": v, "rule": "bridge share of T00 samples > 0.8", "supported": bool(v > 0.8)}
v = {kk: main[kk]["F_total"] for kk in ("A_Ord/U3", "A_XPT_S/U2")}; vh = main["st6"]["T2_median_rel_change"]
H["H115d"] = {"class": "prediction", "value": {"F_total": v, "T2_median_rel_change_st6": vh},
              "rule": "F_total - 1 < 0.05 both and |T2 median change| < 0.10", "supported": bool(all(x - 1 < 0.05 for x in v.values()) and abs(vh) < 0.10)}
R["hypotheses"] = H

# ---------- 図 ----------
T6, TF, BR, b6, n6 = R.pop("_arrays")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].hist(np.log10(np.maximum(T6, 0.1)), bins=60, alpha=0.6, label="T_scrap (ruin iron lasts)")
ax[0].hist(np.log10(np.maximum(TF, 0.1)), bins=60, alpha=0.6, label="tFe (time to own smelting)")
ax[0].set_xlabel("log10 years"); ax[0].set_ylabel("T00 samples"); ax[0].legend(); ax[0].set_title("103: heavy collapse, pre-iron groups")
names = list(R["runs"]); vals = [R["runs"][n_]["A_Ord/U3"]["F_heavy"] for n_ in names]
ax[1].barh(names + ["upper (all bridged)"], vals + [R["upper_bound_all_bridge"]["A_Ord/U3"]["F_heavy"]])
ax[1].axvline(1, color="k", lw=0.8); ax[1].set_xlabel("F_heavy (A_Ord/U3), value ratio vs 057 base")
ax[1].set_xlim(0.98, max(vals + [1.02]) * 1.01)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_ruin_iron.png"), dpi=120)

R["runtime_s"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in R if k not in ("runs",)}, indent=1, ensure_ascii=False, default=float))
print(json.dumps({n_: {kk: vv for kk, vv in r.items() if kk != "by_comp"} for n_, r in R["runs"].items()}, indent=1, default=float))
