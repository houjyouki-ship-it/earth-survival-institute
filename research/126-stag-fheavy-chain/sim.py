"""研究126：停滞を入れた人口待ちで F_heavy の連鎖（104→108→112→…→124）を再実行する（open_problems 295）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-25 17:43 JST）。
注：F_heavy・V は生存確率や復興確率ではない。
124 の sim.py を「# ---------- 仮説」の前まで exec し 124 の table の F_heavy を差 0 で再現（assert）。
停滞の入力は 125 の results.json（V2_inputs、cases の r_all）から読む【事実の出典は 125 の sources/】。
停滞の長さの対数一様 263〜1000 年の版は【推測】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, ".."))
P124 = os.path.join(RES, "124-cam-stepper-wear", "sim.py")
src = open(P124, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
body = src[:cut]
old = "fT = t2e_two(100.0, L_bulk, PPREC * PVN[vn])"
assert body.count(old) == 1
body = body.replace(old, old + "; FTS[vn] = fT")          # 124 が使った T2e の条件（乱数込み）をそのまま取っておく
G124 = {"__file__": P124, "__name__": "sim124", "FTS": {}}
exec(compile(body, P124, "exec"), G124)
R124 = json.load(open(os.path.join(RES, "124-cam-stepper-wear", "results.json"), encoding="utf8"))
d124 = max(abs(G124["TAB"][t]["F_heavy"] - R124["table"][t]["F_heavy"]) for t in R124["table"])
assert d124 < 1e-12, d124
R125 = json.load(open(os.path.join(RES, "125-pop-recovery-history", "results.json"), encoding="utf8"))

run = G124["run"]; G1 = run.__globals__                       # 112 の名前空間
T2_from = G1["T2_from"]; G8 = T2_from.__globals__             # 108 の名前空間
N, B, SRC, COMPS, FIXED, V_m, agg, q_TC, v0 = (G1[k] for k in ("N", "B", "SRC", "COMPS", "FIXED", "V_m", "agg", "q_TC", "v0"))
N_Fe, N_lens, N_T2e, r_pop, CONS, zmin, STAGES = (G1[k] for k in ("N_Fe", "N_lens", "N_T2e", "r_pop", "CONS", "zmin", "STAGES"))
POP_W = G1["G"]["POP_W"]
U_lit, d01, lit_mult, d12, d_Fe, W = (G8[k] for k in ("U_lit", "d01", "lit_mult", "d12", "d_Fe", "W"))
KAPS, fe_d, le_d, FTS = G124["KAPS"], G124["fe_d"], G124["le_d"], G124["FTS"]
OUT = {"study": "126-stag-fheavy-chain", "check_124_table_F_maxdiff": d124, "note": "F_heavy・V は生存確率や復興確率ではない"}

# ---------- 停滞を入れた T2 と段の喪失 ----------
def T2_stag(k, st, stag, r):
    if k in FIXED: return B[k]["T2s"]
    lit = SRC[k]["lit"]; n = np.maximum(SRC[k]["sv"]["nmax"], 1.0)
    wait = lambda Nk: np.where(Nk > n, stag + np.log(Nk / n) / r, 0.0)
    d01e = np.where(U_lit < lit, d01 * lit_mult, d01)
    tFe = np.where(st == 6, np.maximum(d_Fe, wait(N_Fe)), 0.0)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wait(N_lens)), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12, wait(N_T2e)), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [W, np.zeros(N), t2], default=np.zeros(N))

def new_st_stag(k, kap, Lk, stag, r, tb_stag):
    st = B[k]["st"]; n = np.maximum(SRC[k]["sv"]["nmax"], 1.0)
    zm, _ = zmin(n, r, kap, POP_W)
    lev = np.full(N, 6); prev = np.ones(N, bool)
    for nm, code, Nk in STAGES:
        tb = np.maximum(0.0, np.log(Nk / n) / r)
        if tb_stag: tb = np.where(Nk > n, tb + stag, tb)
        keep = (zm >= np.log(Nk)) & Lk[nm](tb) & prev
        lev = np.where(keep, code, lev); prev = keep
    return np.where(CONS[k], np.minimum(st, lev), st)

def base_val(stag, r):
    D0 = {}
    for k in COMPS:
        T = T2_stag(k, B[k]["st"], stag, r)
        D0[k] = V_m(k, q_TC, True, T) - V_m(k, 0.3, False, T)
    return agg({"x": D0}, anchors=("A_Ord", "A_XPT_S"))

def run_stag(kap, Lk, stag, r, tb_stag, vb):
    D = {}; sh = {}
    for k in COMPS:
        if k in FIXED: T2n = B[k]["T2s"]
        else:
            s2 = new_st_stag(k, kap, Lk, stag, r, tb_stag); T2n = T2_stag(k, s2, stag, r)
            sh[k] = (float(np.isin(B[k]["st"], (4, 5, 6)).mean()), float(np.isin(s2, (4, 5, 6)).mean()))
        D[k] = V_m(k, q_TC, True, T2n) - V_m(k, 0.3, False, T2n)
    v1 = agg({"x": D}, anchors=("A_Ord", "A_XPT_S"))
    b_ = vb["A_Ord"]["U3"]["x"]; n_ = v1["A_Ord"]["U3"]["x"]
    return {"F_heavy": n_["heavy"] / b_["heavy"], "F_total": n_["total"] / b_["total"],
            "base_heavy": b_["heavy"], "new_heavy": n_["heavy"], "base_total": b_["total"], "new_total": n_["total"],
            "U3_heavy_rel_drop": 1 - sh["U3"][1] / sh["U3"][0]}

# 検算：停滞 0・元の r で 108 の T2_from と B の T2s、112 の v0、124 の table を差 0
Z0 = np.zeros(N)
OUT["check_T2_stag0"] = max(float(np.abs(T2_stag(k, B[k]["st"], Z0, r_pop) - B[k]["T2s"]).max()) for k in COMPS)
assert OUT["check_T2_stag0"] == 0.0
VB0 = base_val(Z0, r_pop)
OUT["check_v0"] = abs(VB0["A_Ord"]["U3"]["x"]["heavy"] - v0["A_Ord"]["U3"]["x"]["heavy"])
assert OUT["check_v0"] < 1e-12
DESIGNS = list(FTS)
LK = {vn: {"Fe": fe_d, "lens": le_d, "T2e": FTS[vn]} for vn in DESIGNS}
chk = 0.0
for vn in DESIGNS:
    for kn in KAPS:
        o = run_stag(KAPS[kn], LK[vn], Z0, r_pop, False, VB0)
        chk = max(chk, abs(o["F_heavy"] - R124["table"][f"{kn}|{vn}"]["F_heavy"]))
OUT["check_124_via_run_stag"] = chk
assert chk < 1e-12, chk

# ---------- 125 と同じ乱数・同じ入力 ----------
g5 = np.random.default_rng(20261125)
U_case = g5.uniform(size=N); U_stag = g5.uniform(size=N); U_pick = g5.uniform(size=N)
V2 = R125["V2_inputs"]
def stag_draw(p, vals):
    vals = np.asarray(vals, float); pick = vals[np.minimum((U_pick * len(vals)).astype(int), len(vals) - 1)]
    return np.where(U_stag < p, pick, 0.0)
def from_list(vals):
    vals = np.asarray(vals, float); return vals[np.minimum((U_case * len(vals)).astype(int), len(vals) - 1)]
rall_ni = [c["r_all"] for c in R125["cases"] if c["r_all"] is not None and not c["immig"]]
VARS = {
 "V0": (Z0, r_pop),
 "V2a": (stag_draw(V2["p_nr"], V2["stag_years"]), r_pop),
 "V2b": (stag_draw(V2["p_nr_no_immig"], V2["stag_years_no_immig"]), r_pop),
 "V2b_logU263_1000": (np.where(U_stag < V2["p_nr_no_immig"], np.exp(np.log(263) + U_pick * (np.log(1000) - np.log(263))), 0.0), r_pop),
 "V3b": (Z0, np.maximum(from_list(rall_ni), 1e-9)),
}
OUT["inputs"] = {"p_V2a": V2["p_nr"], "p_V2b": V2["p_nr_no_immig"], "stag_years": V2["stag_years"], "rall_no_immig": rall_ni,
                 "share_stag_V2b": float((VARS["V2b"][0] > 0).mean())}
TAB = {}
for vname, (stag, r) in VARS.items():
    vb = base_val(stag, r)
    for form in ("W", "WT"):
        if vname in ("V0", "V3b") and form == "WT": continue   # 停滞なしでは W と WT は同じ
        for vn in DESIGNS:
            for kn in KAPS:
                TAB[f"{vname}|{form}|{kn}|{vn}"] = run_stag(KAPS[kn], LK[vn], stag, r, form == "WT", vb)
    print(vname, "done", round(time.time() - t0), "s", flush=True)
OUT["table"] = TAB

# ---------- 仮説の判定（登録どおり） ----------
SL = "slow_Tasmania"; BEST = "D2rz|none"
f = lambda v, fm, d=BEST, kn=SL: TAB[f"{v}|{fm}|{kn}|{d}"]["F_heavy"]
H = {}
f0 = f("V0", "W"); f2b = f("V2b", "W")
H["H138a"] = {"class": "confirmatory", "value": {"F_V0": f0, "F_V2b_W": f2b, "rel": f2b / f0 - 1},
              "rule": "F_V2b_W/F_V0 - 1 >= +0.10（向き上で登録）", "supported": bool(f2b / f0 - 1 >= 0.10),
              "moved_10pct_any_direction": bool(abs(f2b / f0 - 1) >= 0.10)}
rank_ok = {}
for key in sorted({k.rsplit("|", 2)[0] for k in TAB if f"|{SL}|" in k}):
    v_, fm = key.split("|")[:2]
    a, b, c = f(v_, fm, "D2rz|none"), f(v_, fm, "D1|main"), f(v_, fm, "D4|main")
    rank_ok[f"{v_}|{fm}"] = {"D2rz": a, "D1": b, "D4": c, "ok": bool(a >= b >= c)}
H["H138b"] = {"class": "confirmatory", "value": rank_ok, "rule": "全ての版・形で D2rz ≥ D1 ≥ D4（遅い κ）", "supported": all(x["ok"] for x in rank_ok.values())}
dW = {v_: abs(f(v_, "W") - f0) for v_ in ("V2a", "V2b", "V2b_logU263_1000")}
dWT = {v_: abs(f(v_, "WT") - f0) for v_ in ("V2a", "V2b", "V2b_logU263_1000")}
H["H138c"] = {"class": "prediction", "value": {"dW": dW, "dWT": dWT}, "rule": "全ての停滞の版で |ΔF|_WT > |ΔF|_W", "supported": all(dWT[v_] > dW[v_] for v_ in dW)}
b0 = TAB[f"V0|W|{SL}|{BEST}"]["base_heavy"]; b2 = TAB[f"V2b|W|{SL}|{BEST}"]["base_heavy"]
H["H138d"] = {"class": "prediction", "value": {"base_heavy_V0": b0, "base_heavy_V2b": b2, "rel": b2 / b0 - 1}, "rule": "rel <= -0.19", "supported": bool(b2 / b0 - 1 <= -0.19)}
f3 = f("V3b", "W")
H["H138e"] = {"class": "prediction", "value": {"dF_V3b": abs(f3 - f0), "dF_V2b": abs(f2b - f0)}, "rule": "|ΔF|_V3b < |ΔF|_V2b", "supported": bool(abs(f3 - f0) < abs(f2b - f0))}
OUT["hypotheses"] = H

# 重い崩壊分の価値の絶対量（新しい側、遅い κ・D2rz）の V0 比：提言に使う
OUT["new_heavy_rel_V0"] = {k.split("|")[0] + "|" + k.split("|")[1]: TAB[k]["new_heavy"] / TAB[f"V0|W|{SL}|{BEST}"]["new_heavy"] - 1
                           for k in TAB if k.endswith(f"|{SL}|{BEST}")}
OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
keys = [k for k in rank_ok]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
x = np.arange(len(keys)); w = 0.27
for i, d in enumerate(("D2rz", "D1", "D4")):
    ax[0].bar(x + (i - 1) * w, [rank_ok[k][d] for k in keys], w, label=d)
ax[0].axhline(f0, color="k", lw=0.7, ls="--"); ax[0].set_xticks(x); ax[0].set_xticklabels(keys, rotation=40, ha="right", fontsize=8)
ax[0].set_ylabel("F_heavy (slow kappa, A_Ord/U3)"); ax[0].legend(); ax[0].set_title("126: F_heavy with post-collapse stagnation", fontsize=9)
kk = [k for k in OUT["new_heavy_rel_V0"]]
ax[1].bar(kk, [OUT["new_heavy_rel_V0"][k] for k in kk], color="C3", label="new heavy value vs V0")
ax[1].bar(kk, [TAB[f"{k}|{SL}|{BEST}"]["base_heavy"] / b0 - 1 for k in kk], color="none", edgecolor="k", label="base heavy value vs V0")
ax[1].set_xticklabels(kk, rotation=40, ha="right", fontsize=8); ax[1].set_ylabel("relative change"); ax[1].legend(fontsize=8)
ax[1].set_title("heavy-collapse value of the intervention (not a survival probability)", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_stag_fheavy.png"), dpi=120)

json.dump(OUT, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ("check_124_table_F_maxdiff", "check_T2_stag0", "check_v0", "check_124_via_run_stag", "inputs")}, ensure_ascii=False, default=float))
for k, o in TAB.items():
    if f"|{SL}|" in k: print(k, "F_heavy", round(o["F_heavy"], 4), "base_h", round(o["base_heavy"], 5), "new_h", round(o["new_heavy"], 5), "drop", round(o["U3_heavy_rel_drop"], 3))
for k, o in TAB.items():
    if f"|{SL}|" not in k and k.endswith(BEST): print(k, "F_heavy", round(o["F_heavy"], 4))
for h, v in H.items(): print(h, v["class"], v["supported"], json.dumps(v["value"], ensure_ascii=False, default=float)[:400])
print("runtime", round(OUT["runtime_s"]), "s")
