"""研究066：現代の学校で教わる読み書きが学校なしで続くか（064 の重み w、open_problems 172）。
計算のみ、ネット接続なし（sources/wb_literacy.json は fetch_wb.py で取得済み）。事前登録：PREREGISTRATION.md。
065 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し 065 の主を差0で再現してから、
L = w·L_broad + (1−w)·L_narrow の w を事例から置き直す。新しい乱数は seed 20261066。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P65 = os.path.abspath(os.path.join(HERE, "..", "065-primer-readability", "sim.py"))
src = open(P65, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G65 = {"__file__": P65, "__name__": "sim065"}
exec(compile(src[:cut], P65, "exec"), G65)
r65 = json.load(open(os.path.join(HERE, "..", "065-primer-readability", "results.json"), encoding="utf8"))
G64 = G65["G64"]
N = G65["N"]; ver = G65["ver"]; get = G65["get"]; ratio_all = G65["ratio_all"]; rho1 = G65["rho1"]; r_reach = G65["r_reach"]
S_ind = G65["S_ind"]; f_i = G65["f_i"]; taup_061 = G65["taup_061"]; lam = G65["lam"]; R65 = G65["R65"]; pp_fn = G65["pp_fn"]
C_T = G65["C_T"]; C_P = G65["C_P"]; C_pr = G65["C_pr"]; qf = G65["qf"]
Ln_m = G64["Ln_m"]; Lb_m = G64["Lb_m"]; mix = G64["mix"]; w_064 = G64["w_mod"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"
R = {"study": "066-literacy-without-school", "seed": 20261066}
PPM = pp_fn()
def rho66(Lv, ppf=PPM, n_p=100, K=20):
    def f(T):
        pc = (1 - Lv) * np.exp(-lam * T)
        return 1 - (1 - rho1) * (1 - r_reach * R65(T, pc, ppf) * S_ind(T, n_p, K, f_i, taup_061))
    return f
chk, _ = ratio_all(ver(rho=rho66(mix(Ln_m, Lb_m, w_064))))
dd = max(abs(chk[k][p] - r65["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_065_main"] = {"max_abs_diff": dd, "reproduced": dd < 1e-12}

# ---------- 資料：世界銀行 ----------
WB = json.load(open(os.path.join(HERE, "sources", "wb_literacy.json"), encoding="utf8"))
def series(ind, iso):
    return sorted((int(y), v) for c, y, v in WB[ind]["rows"] if c == iso)
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
def wb_case(c, pre_slack=0):
    s = series("SE.ADT.1524.LT.ZS", c["iso"]); end = c["end"] if c["end"] is not None else c["start"] + 10
    pre = [x for x in s if x[0] <= c["start"] + pre_slack]; post = [x for x in s if x[0] >= end]
    if not pre or not post: return None
    p0, p1 = pre[-1], post[0]
    a = dict(series("UIS.LR.AG25T64", c["iso"])).get(p1[0])
    return {"iso": c["iso"], "start": c["start"], "end": c["end"], "pre": list(p0), "post": list(p1), "g": p1[1] / p0[1],
            "adult25_64_post": a, "youth_over_adult_post": (p1[1] / a) if a else None}
WBs = [x for x in (wb_case(c) for c in CJ["conflicts"]) if x]
WBa = [x for x in (wb_case(c, 3) for c in CJ["conflicts"]) if x]
gs = np.array([x["g"] for x in WBs]); ga = np.array([x["g"] for x in WBa])
R["wb"] = {"strict": WBs, "alt": WBa, "strict_g_median": float(np.median(gs)), "strict_g_min_max": [float(gs.min()), float(gs.max())],
           "alt_g_median": float(np.median(ga)), "n_strict": len(WBs), "n_alt": len(WBa),
           "excluded_strict": [c["iso"] for c in CJ["conflicts"] if c["iso"] not in [x["iso"] for x in WBs]],
           "youth_gt_adult_share_strict": float(np.mean([x["youth_over_adult_post"] > 1 for x in WBs if x["youth_over_adult_post"]]))}
yc = [1.0 if x["g"] >= 0.9 else (0.5 if x["g"] >= 0.5 else 0.0) for x in WBs]

# ---------- 資料：局面 ----------
EP = CJ["episodes"]
def tally(sel, yk="y"):
    s = [c for c in EP if sel(c)]; return len(s), float(sum(c[yk] for c in s))
SEL = {"main": (lambda c: True, "y"), "alt": (lambda c: True, "y_alt"), "collapse_only": (lambda c: c["group"] == "collapse", "y"),
       "deep_only": (lambda c: c["depth"] == "deep", "y"), "deep_regime": (lambda c: c["depth"] in ("deep", "regime"), "y"),
       "built_only": (lambda c: c["group"] == "built", "y"), "relig1": (lambda c: c["relig"] == 1, "y"), "relig0": (lambda c: c["relig"] < 1, "y")}
NY = {nm: tally(f, yk) for nm, (f, yk) in SEL.items()}
NY["with_conflicts"] = (NY["main"][0] + len(yc), NY["main"][1] + sum(yc))
R["tally"] = {nm: {"n": n, "y": y, "mean_jeffreys": (0.5 + y) / (1 + n)} for nm, (n, y) in NY.items()}
R["relig_raw_means"] = {"relig1": NY["relig1"][1] / NY["relig1"][0], "relig_lt1": NY["relig0"][1] / NY["relig0"][0]}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261066)
WV = {nm: g.beta(0.5 + y, 0.5 + n - y, N) for nm, (n, y) in NY.items()}
WV["u01"] = w_064
R["w_draws"] = {nm: {"mean": float(v.mean()), "q05_50_95": [float(x) for x in np.quantile(v, [0.05, 0.5, 0.95])]} for nm, v in WV.items()}

# ---------- 版 ----------
RT = {}; DV = {}
for nm, wv in WV.items():
    RT[nm], DV[nm] = ratio_all(ver(rho=rho66(mix(Ln_m, Lb_m, wv))))
d = max(abs(RT["u01"][k][p] - r65["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_u01_equals_065"] = {"max_abs_diff": d}
RT["065_main"] = r65["ratio_to_V0"]["main"]; RT["064_main"] = r65["ratio_to_V0"]["064_main"]
R["ratio_to_V0"] = RT
R["scan_w"] = []
for wv in [0.0, 0.25, 0.5, 0.75, 1.0]:
    rr, _ = ratio_all(ver(rho=rho66(mix(Ln_m, Lb_m, np.full(N, wv)))))
    R["scan_w"].append({"w": wv, M1: rr[M1]["heavy"], "A_XPT_S/U2": rr["A_XPT_S/U2"]["heavy"]})
# 局面を越える確率 1−L の平均
R["keep_mean"] = {nm: float(1 - mix(Ln_m, Lb_m, wv).mean()) for nm, wv in WV.items()}

# ---------- 費用あたり（冊子＋primer の合計、065 と同じ式） ----------
R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for p in ("total", "heavy"):
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; R["CE"][key][p] = {}
        for nm in WV:
            dall = DV[nm][a][u]["TC"][p] - vtc
            R["CE"][key][p][nm] = qf(dall / vt * C_T / (C_P(100) + C_pr)) if dall > 0 else [0.0, 0.0, 0.0]
        R["CE"][key][p]["065_main"] = r65["CE"][key][p]["main"]["booklet_plus_primer"]

# ---------- 仮説 ----------
H = {}; h = lambda nm, k=M1: RT[nm][k]["heavy"]; TM = R["tally"]; h65 = r65["ratio_to_V0"]["main"][M1]["heavy"]
v = TM["main"]["mean_jeffreys"]; H["H78a"] = {"value": v, "rule": "main w Jeffreys mean in [0.4,0.8]", "supported": 0.4 <= v <= 0.8}
v = [TM["collapse_only"]["mean_jeffreys"], TM["main"]["mean_jeffreys"]]; H["H78b"] = {"value": v, "rule": "collapse_only <= main - 0.1", "supported": v[0] <= v[1] - 0.1}
v = R["wb"]["strict_g_median"]; H["H78c"] = {"value": v, "rule": "WB strict median g >= 1.0", "supported": v >= 1.0}
v = h("main") - h65; H["H78d"] = {"value": [h("main"), h65, v], "rule": "|main heavy - 065 main| < 0.01", "supported": abs(v) < 0.01}
v = [h("deep_regime"), h65]; H["H78e"] = {"value": v, "rule": "deep_regime heavy <= 065 main - 0.005", "supported": v[0] <= v[1] - 0.005}
vs = [h(n) for n in ("main", "alt", "collapse_only", "deep_only", "deep_regime", "built_only", "with_conflicts", "relig1", "relig0")]
H["H78f"] = {"value": [min(vs), max(vs), max(vs) - min(vs)], "rule": "range of heavy across versions <= 0.03", "supported": max(vs) - min(vs) <= 0.03}
v = R["wb"]["youth_gt_adult_share_strict"]; H["H78g"] = {"value": v, "rule": "share youth>adult25-64 in post survey >= 0.8", "supported": v >= 0.8}
v = R["relig_raw_means"]["relig1"] - R["relig_raw_means"]["relig_lt1"]; H["H78h"] = {"value": v, "rule": "relig1 raw mean - relig<1 >= 0.3", "supported": v >= 0.3}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for i, x in enumerate(sorted(WBs, key=lambda x: x["g"])):
    axs[0].plot([0, 1], [x["pre"][1], x["post"][1]], "o-", color="C3" if x["g"] < 0.9 else "C0")
    axs[0].text(1.02, x["post"][1], f'{x["iso"]} {x["pre"][0]}->{x["post"][0]} g={x["g"]:.2f}', fontsize=7, va="center")
axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(["last survey before war", "first survey after war"]); axs[0].set_xlim(-0.1, 1.7)
axs[0].set_ylabel("youth (15-24) literacy, % (World Bank / UIS)"); axs[0].set_title("War-time schooling and next-cohort literacy (strict rule, 9 cases)", fontsize=8)
nm_e = [c["id"] for c in EP]; ye = [c["y"] for c in EP]; cl = ["C2" if c["relig"] == 1 else ("C1" if c["relig"] == 0.5 else "C3") for c in EP]
ax2 = axs[1]; ax2.barh(range(len(EP)), ye, color=cl); ax2.set_yticks(range(len(EP))); ax2.set_yticklabels([f'{c["id"]} [{c["depth"]}]' for c in EP], fontsize=7)
ax2.set_xlabel("y: broad reading carried on without state schools (green: lay religious reading, orange: partial, red: none)")
ax2.set_title(f'episodes: main w mean {TM["main"]["mean_jeffreys"]:.2f}, collapse-only {TM["collapse_only"]["mean_jeffreys"]:.2f}', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_cases_wb.png"), dpi=130); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
names = ["main", "alt", "collapse_only", "deep_only", "deep_regime", "built_only", "with_conflicts", "relig1", "relig0", "u01", "064_main"]
yy = np.arange(len(names))
axs[0].barh(yy - 0.2, [RT[n][M1]["heavy"] for n in names], height=0.4, color=["k" if n in ("u01", "064_main") else "C0" for n in names], label="A_Ord/U3")
axs[0].barh(yy + 0.2, [RT[n]["A_XPT_S/U2"]["heavy"] for n in names], height=0.4, color="C1", alpha=0.7, label="A_XPT_S/U2")
axs[0].set_yticks(yy); axs[0].set_yticklabels([f'{n} (w={R["w_draws"][n]["mean"]:.2f})' if n in R["w_draws"] else n for n in names], fontsize=7)
axs[0].set_xlim(0.75, 0.92); axs[0].axvline(h65, color="k", ls=":", lw=0.7); axs[0].set_xlabel("heavy-collapse TC / V0 (primer included)"); axs[0].legend(fontsize=7)
sw = R["scan_w"]
axs[1].plot([r["w"] for r in sw], [r[M1] for r in sw], "o-", label="A_Ord/U3"); axs[1].plot([r["w"] for r in sw], [r["A_XPT_S/U2"] for r in sw], "s--", label="A_XPT_S/U2")
for nm, c in (("main", "C2"), ("deep_regime", "C3"), ("relig1", "C4")):
    axs[1].axvline(R["w_draws"][nm]["mean"], color=c, ls=":", label=f"{nm} w mean")
axs[1].set_xlabel("w (constant): weight that modern literacy behaves 'broad'"); axs[1].set_ylabel("heavy TC / V0"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions_scan.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check", R["check_065_main"], R["check_u01_equals_065"])
print("wb", json.dumps({k: v for k, v in R["wb"].items() if k not in ("strict", "alt")}))
for x in WBs: print(" ", x)
print("tally", json.dumps(R["tally"])); print("relig", R["relig_raw_means"])
print("w_draws", json.dumps({k: round(v["mean"], 3) for k, v in R["w_draws"].items()})); print("keep", json.dumps({k: round(v, 3) for k, v in R["keep_mean"].items()}))
for nm in RT: print(nm, {k: {p: round(RT[nm][k][p], 4) for p in PARTS} for k in KEYS})
print("scan_w", json.dumps(R["scan_w"]))
print("CE total", json.dumps(R["CE"][M1]["total"])); print("CE heavy", json.dumps(R["CE"][M1]["heavy"]))
for k, x in H.items(): print(k, x["supported"], x["value"])
print("runtime", R["runtime_s"])
