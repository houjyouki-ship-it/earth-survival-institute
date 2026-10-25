"""研究127：停滞と段の喪失の相関（open_problems 298）と地平 H の感度（299）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-25 19:47 JST）。
注：F_heavy・V は生存確率や復興確率ではない。
126 の sim.py を「# ---------- 125 と同じ乱数」の前まで exec（126 の assert 4 つを通る）し、
H=1000 で 126 の V0・V2b の table を差 0 で再現（assert）。停滞の入力の出典は 125 の sources/。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
P126 = os.path.join(RES, "126-stag-fheavy-chain", "sim.py")
src = open(P126, encoding="utf8").read(); body = src[:src.index("\n# ---------- 125 と同じ乱数")]
G = {"__file__": P126, "__name__": "sim126"}
exec(compile(body, P126, "exec"), G)
print("126 prefix exec", round(time.time() - t0), "s", flush=True)
R126 = json.load(open(os.path.join(RES, "126-stag-fheavy-chain", "results.json"), encoding="utf8"))
R125 = G["R125"]; N = G["N"]; B = G["B"]; COMPS = G["COMPS"]; FIXED = G["FIXED"]
run_stag, base_val, KAPS, LK, DESIGNS, r_pop = (G[k] for k in ("run_stag", "base_val", "KAPS", "LK", "DESIGNS", "r_pop"))
Vg = G["V_m"].__globals__; assert float(Vg["H"]) == 1000.0
OUT = {"study": "127-stag-corr-horizon", "note": "F_heavy・V は生存確率や復興確率ではない"}

g5 = np.random.default_rng(20261125)
U_case = g5.uniform(size=N); U_stag = g5.uniform(size=N); U_pick = g5.uniform(size=N)
V2 = R125["V2_inputs"]; vals = np.asarray(V2["stag_years_no_immig"], float)
pick = vals[np.minimum((U_pick * len(vals)).astype(int), len(vals) - 1)]
P0 = V2["p_nr_no_immig"]; assert abs(P0 - 0.6) < 1e-12
lost = {k: np.isin(B[k]["st"], (4, 5, 6)) for k in COMPS}
share = {k: float(lost[k].mean()) for k in COMPS}
OUT["share_lost"] = share

def stag_of(pfun):
    """成分ごとの停滞の配列。U_stag・pick は全成分で共通（126 と同じ）。pfun(k) -> 標本ごとの p"""
    return {k: np.where(U_stag < pfun(k), pick, 0.0) for k in COMPS}
def p_mid(k):
    s = share[k]; lo = 0.0 if s >= 1 else max(0.0, (P0 - 0.8 * s) / (1 - s))
    return np.where(lost[k], 0.8, lo)
CVER = {
 "V0":    {k: np.zeros(N) for k in COMPS},
 "C0":    stag_of(lambda k: np.full(N, P0)),
 "Ccond": stag_of(lambda k: np.where(lost[k], P0, 0.0)),
 "Cmid":  stag_of(p_mid),
 "Cmass": stag_of(lambda k: np.where(lost[k], min(1.0, P0 / max(share[k], 1e-12)), 0.0)),
 "C1":    stag_of(lambda k: np.where(lost[k], 1.0, 0.0)),
}
OUT["p_in_lost_U3"] = {"C0": P0, "Ccond": P0, "Cmid": 0.8, "Cmass": min(1.0, P0 / share["U3"]), "C1": 1.0}
OUT["p_marginal_U3"] = {v: float((CVER[v]["U3"] > 0).mean()) for v in CVER}

# 126 の関数は stag を全成分共通の配列で受ける。成分ごとの stag を渡すため、k ごとに呼び分ける薄い包み
T2_stag, new_st_stag, V_m, agg, q_TC = (G[k] for k in ("T2_stag", "new_st_stag", "V_m", "agg", "q_TC"))
def base_val_k(S, r):
    D0 = {}
    for k in COMPS:
        T = T2_stag(k, B[k]["st"], S[k], r); D0[k] = V_m(k, q_TC, True, T) - V_m(k, 0.3, False, T)
    return agg({"x": D0}, anchors=("A_Ord", "A_XPT_S"))
def run_k(kap, Lk, S, r, tb, vb):
    D = {}
    for k in COMPS:
        if k in FIXED: T2n = B[k]["T2s"]
        else: s2 = new_st_stag(k, kap, Lk, S[k], r, tb); T2n = T2_stag(k, s2, S[k], r)
        D[k] = V_m(k, q_TC, True, T2n) - V_m(k, 0.3, False, T2n)
    v1 = agg({"x": D}, anchors=("A_Ord", "A_XPT_S")); b_ = vb["A_Ord"]["U3"]["x"]; n_ = v1["A_Ord"]["U3"]["x"]
    return {"F_heavy": n_["heavy"] / b_["heavy"], "base_heavy": b_["heavy"], "new_heavy": n_["heavy"],
            "F_total": n_["total"] / b_["total"]}

SL = "slow_Tasmania"; MAIN3 = ("D2rz|none", "D1|main", "D4|main")
TAB = {}
for Hh in (1000.0, 1500.0, 2000.0):
    Vg["H"] = Hh
    for vname, S in CVER.items():
        forms = ("W", "WT") if vname in ("C0", "Cmass") else ("W",)
        vb = base_val_k(S, r_pop)
        for fm in forms:
            for vn in DESIGNS:
                for kn in KAPS:
                    if kn != SL and vn != "D2rz|none": continue
                    if kn == SL and vn not in MAIN3: continue
                    TAB[f"H{int(Hh)}|{vname}|{fm}|{kn}|{vn}"] = run_k(KAPS[kn], LK[vn], S, r_pop, fm == "WT", vb)
    print("H", Hh, "done", round(time.time() - t0), "s", flush=True)
Vg["H"] = 1000.0
OUT["table"] = TAB

# 検算：H=1000 の V0・C0 を 126 の V0・V2b と差 0、Ccond は C0 と差 0（登録の構造的な予想）
chk = 0.0
for k, o in TAB.items():
    if not k.startswith("H1000|"): continue
    _, v, fm, kn, vn = k.split("|", 4)
    m = {"V0": "V0", "C0": "V2b"}.get(v)
    if m and f"{m}|{fm}|{kn}|{vn}" in R126["table"]:
        chk = max(chk, abs(o["F_heavy"] - R126["table"][f"{m}|{fm}|{kn}|{vn}"]["F_heavy"]))
OUT["check_126_maxdiff"] = chk; assert chk < 1e-12, chk
dc = max(abs(TAB[k]["F_heavy"] - TAB[k.replace("|Ccond|", "|C0|")]["F_heavy"]) for k in TAB if "|Ccond|" in k)
OUT["Ccond_minus_C0_maxdiff"] = dc

f = lambda h, v, fm="W", d="D2rz|none", kn=SL: TAB[f"H{h}|{v}|{fm}|{kn}|{d}"]
HYP = {}
fm_ = f(1000, "Cmass")["F_heavy"]; fc = f(1000, "C0")["F_heavy"]
HYP["H139a"] = {"class": "confirmatory", "value": {"F_C0": fc, "F_Cmass": fm_, "rel": fm_ / fc - 1}, "rule": "rel >= +0.05", "supported": bool(fm_ / fc - 1 >= 0.05)}
up = {h: f(h, "C0")["F_heavy"] - f(h, "V0")["F_heavy"] for h in (1000, 1500, 2000)}
HYP["H139b"] = {"class": "confirmatory", "value": up, "rule": "up(H2000) < 0.5*up(H1000)", "supported": bool(up[2000] < 0.5 * up[1000])}
rk = {}
for k in TAB:
    if f"|{SL}|D2rz|none" in k:
        pre = k.rsplit("|", 3)[0]
        a, b, c = (TAB[f"{pre}|{SL}|{d}"]["F_heavy"] for d in MAIN3)
        rk[pre] = {"D2rz": a, "D1": b, "D4": c, "ok": bool(a >= b >= c)}
HYP["H139c"] = {"class": "confirmatory", "value": rk, "rule": "全ての版・H で D2rz ≥ D1 ≥ D4", "supported": all(x["ok"] for x in rk.values())}
b1, b2 = f(1000, "V0")["base_heavy"], f(2000, "V0")["base_heavy"]
HYP["H139d"] = {"class": "prediction", "value": {"base_H1000": b1, "base_H2000": b2, "rel": b2 / b1 - 1}, "rule": "rel < +0.10", "supported": bool(b2 / b1 - 1 < 0.10)}
OUT["hypotheses"] = HYP
OUT["amounts_slow_D2rz_W"] = {f"H{h}|{v}": {"F": f(h, v)["F_heavy"], "base": f(h, v)["base_heavy"], "new": f(h, v)["new_heavy"],
                               "new_rel_V0_H1000": f(h, v)["new_heavy"] / f(1000, "V0")["new_heavy"] - 1}
                              for h in (1000, 1500, 2000) for v in CVER}
OUT["life_quantiles"] = [float(np.quantile(Vg["life"], q)) for q in (0.1, 0.5, 0.9)]
OUT["runtime_s"] = time.time() - t0

# 図
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
vv = list(CVER); x = np.arange(len(vv)); w = 0.27
for i, h in enumerate((1000, 1500, 2000)):
    ax[0].bar(x + (i - 1) * w, [f(h, v)["F_heavy"] for v in vv], w, label=f"H={h}")
    ax[1].bar(x + (i - 1) * w, [f(h, v)["new_heavy"] / f(1000, "V0")["new_heavy"] - 1 for v in vv], w, label=f"H={h}")
for a, t in zip(ax, ("F_heavy (slow kappa, D2rz, W)", "new heavy value vs V0@H1000 (not a survival prob.)")):
    a.set_xticks(x); a.set_xticklabels(vv); a.set_title("127: " + t, fontsize=9); a.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_corr_horizon.png"), dpi=120)

json.dump(OUT, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ("check_126_maxdiff", "Ccond_minus_C0_maxdiff", "p_in_lost_U3", "p_marginal_U3", "life_quantiles")}, ensure_ascii=False, default=float))
for k, o in OUT["amounts_slow_D2rz_W"].items(): print(k, {a: round(b, 5) for a, b in o.items()})
for k, o in TAB.items():
    if f"|{SL}|" not in k: print(k, "F_heavy", round(o["F_heavy"], 4))
    elif "|WT|" in k: print(k, "F_heavy", round(o["F_heavy"], 4))
for h, v in HYP.items(): print(h, v["class"], v["supported"], json.dumps(v["value"], ensure_ascii=False, default=float)[:300])
print("runtime", round(OUT["runtime_s"]), "s")
