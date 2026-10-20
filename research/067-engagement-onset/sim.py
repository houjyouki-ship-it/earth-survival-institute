"""研究067：崩壊の後の社会が古い文書に取り組み始める時点（open_problems 175）と、教える人なしに文書から読み始めた例（174）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
066 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し 066 の主を差0で再現してから、
primer の経路の時間の原点を「学問の段 P（T2 の Δ 年前）」に置く二段の式に替える。新しい乱数は seed 20261067。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P66 = os.path.abspath(os.path.join(HERE, "..", "066-literacy-without-school", "sim.py"))
src = open(P66, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G66 = {"__file__": P66, "__name__": "sim066"}
exec(compile(src[:cut], P66, "exec"), G66)
r66 = json.load(open(os.path.join(HERE, "..", "066-literacy-without-school", "results.json"), encoding="utf8"))
r65 = json.load(open(os.path.join(HERE, "..", "065-primer-readability", "results.json"), encoding="utf8"))
G65 = G66["G65"]
N = G66["N"]; ver = G66["ver"]; get = G66["get"]; ratio_all = G66["ratio_all"]; rho66 = G66["rho66"]
mix = G66["mix"]; Ln_m = G66["Ln_m"]; Lb_m = G66["Lb_m"]; WV = G66["WV"]; qf = G66["qf"]
C_T = G66["C_T"]; C_P = G66["C_P"]; C_pr = G66["C_pr"]
k_rec = G65["k_rec"]; k_cont = G65["k_cont"]; tau_key = G65["tau_key"]; tau_unk = G65["tau_unk"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"
R = {"study": "067-engagement-onset", "seed": 20261067}
L66 = mix(Ln_m, Lb_m, WV["main"])

# ---------- 資料 ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
CO = CJ["corpora"]
reg = [c for c in CO if c["loss"] is not None and c["onset"] is not None]
x = np.array([c["loss"] for c in reg], float); y = np.array([c["onset"] for c in reg], float)
def slope(xx, yy): return float(np.polyfit(xx, yy, 1)[0])
gb = np.random.default_rng(20261067)
bs = []
for _ in range(10000):
    i = gb.integers(0, len(x), len(x))
    if np.ptp(x[i]) > 0: bs.append(slope(x[i], y[i]))
bs = np.array(bs)
R["stage_vs_time"] = {"n": len(reg), "ids": [c["id"] for c in reg], "slope": slope(x, y), "slope_q05_95": [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))],
                      "sd_onset_year": float(y.std(ddof=1)), "sd_lag": float((y - x).std(ddof=1)), "ratio_sd": float(y.std(ddof=1) / (y - x).std(ddof=1)),
                      "onset_range": [float(y.min()), float(y.max())], "loss_range": [float(x.min()), float(x.max())],
                      "lags": {c["id"]: c["onset"] - c["loss"] for c in reg}}
# 見えていた文書の 1600年より前の曝露
VIS = [c for c in CO if c["visible"] and c["loss"] is not None and c["loss"] < 1600]
exp_sys = float(sum(1600 - c["loss"] for c in VIS)); k_sys = sum(1 for c in VIS if c["onset"] is not None and c["onset"] < 1600)
exp_any = float(sum((c["attempt_any"] if (c["attempt_any"] is not None and c["attempt_any"] < 1600) else 1600) - c["loss"] for c in VIS))
k_any = sum(1 for c in VIS if c["attempt_any"] is not None and c["attempt_any"] < 1600)
R["premodern"] = {"ids": [c["id"] for c in VIS], "n": len(VIS), "exposure_sys": exp_sys, "k_sys": k_sys, "exposure_any": exp_any, "k_any": k_any,
                  "tau_any_jeffreys_mean_rate": exp_any / (0.5 + k_any), "tau_sys_jeffreys_mean_rate": exp_sys / (0.5 + k_sys),
                  "onset_after_1600": {c["id"]: c["onset"] for c in VIS}}
# 学問の段の社会が新しく手にした文書
NEW = [c for c in CO if (not c["visible"]) and c["access"] >= 1850]
def lagB(c, alt=False):
    on = c["onset"] if c["onset"] is not None else c["onset_ub"]
    acc = c.get("access_alt", c["access"]) if alt else c["access"]
    return on - acc
lb = np.array([lagB(c) for c in NEW], float); lba = np.array([lagB(c, True) for c in NEW], float)
R["new_access"] = {"ids": [c["id"] for c in NEW], "lags": lb.tolist(), "median": float(np.median(lb)), "lags_alt": lba.tolist(), "median_alt": float(np.median(lba)),
                   "share_le10": float(np.mean(lb <= 10))}
NE = CJ["near174"]
R["near174"] = {"strict": int(sum(c["strict"] for c in NE)), "n_candidates": len(NE),
                "cases": [{k: c.get(k) for k in ("id", "nonscholar", "no_teacher", "unknown_script", "docs_only", "years", "success")} for c in NE]}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261067)
lam_any = g.gamma(0.5 + k_any, 1 / exp_any, N)
lam_sys = g.gamma(0.5 + k_sys, 1 / exp_sys, N)
R["draws"] = {nm: {"mean": float(v.mean()), "q05_50_95": [float(q) for q in np.quantile(v, [0.05, 0.5, 0.95])]} for nm, v in
              {"lam_any": lam_any, "lam_sys": lam_sys, "tau_any": 1 / lam_any, "tau_sys": 1 / lam_sys}.items()}

def two_phase(T, lam, tk_pre, tk, Delta, W):
    """崩壊から T 年で T2。P は T2 の Delta 年前（0 より前なら 0）。P の前は率 lam で取り組み tk_pre で読む。P の後は tk で読む。T+W までに読めた確率。"""
    D = T + W
    TP = np.zeros_like(T) if np.isinf(Delta) else np.maximum(T - Delta, 0.0)
    if np.isinf(Delta) and W == 0:
        return 1 - np.exp(-T / tk)            # 065/066 の式そのもの
    a = 1 / tk_pre - lam
    pre = (1 - np.exp(-lam * TP)) - lam * np.exp(-D / tk_pre) * (np.exp(a * TP) - 1) / a
    post = np.exp(-lam * TP) * (1 - np.exp(-(D - TP) / tk))
    return pre + post
def pp67(Delta=0.0, W=0.0, lam=None, tk_pre=None):
    lv = lam_any if lam is None else lam; tp = tau_key if tk_pre is None else tk_pre
    return lambda T: k_rec * k_cont * two_phase(T, lv, tp, tau_key, Delta, W)

# ---------- 版 ----------
PV = {"e066": pp67(Delta=np.inf), "nopr": None, "main": pp67(), "sys": pp67(lam=lam_sys), "pre_unkeyed": pp67(tk_pre=tau_unk),
      "lead25": pp67(25.0), "lead50": pp67(50.0), "lead100": pp67(100.0), "lead200": pp67(200.0), "lead400": pp67(400.0),
      "W20": pp67(W=20.0), "W50": pp67(W=50.0)}
RT = {}; DV = {}
for nm, f in PV.items():
    RT[nm], DV[nm] = ratio_all(ver(rho=rho66(L66, ppf=f)))
d = max(abs(RT["e066"][k][p] - r66["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_066_main"] = {"max_abs_diff": d, "reproduced": d < 1e-12}
d2 = max(abs(RT["nopr"][k][p] - RT["nopr"][k][p]) for k in KEYS for p in PARTS)
RT["066_main"] = r66["ratio_to_V0"]["main"]; RT["065_main"] = r66["ratio_to_V0"]["065_main"]; RT["064_main"] = r66["ratio_to_V0"]["064_main"]
R["ratio_to_V0"] = RT
# p_pr（重い崩壊の T での平均）
B = G65["B"]; hmU = np.isin(B["U3"]["st"], (4, 5)); TU = np.maximum(B["U3"]["T2s"], 0)
R["p_pr_at_T_U3_heavy"] = {nm: float(f(TU)[hmU].mean()) for nm, f in PV.items() if f is not None}
R["T2s_heavy_U3_median"] = float(np.median(TU[hmU]))
# Δ の走査
R["scan_Delta"] = []
for Dl in [0, 25, 50, 100, 200, 400, 800, 1600]:
    rr, _ = ratio_all(ver(rho=rho66(L66, ppf=pp67(float(Dl)))))
    R["scan_Delta"].append({"Delta": Dl, M1: rr[M1]["heavy"], "A_XPT_S/U2": rr["A_XPT_S/U2"]["heavy"]})

# ---------- 費用あたり ----------
R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for p in ("total", "heavy"):
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; v0 = DV["nopr"][a][u]["TC"][p]; R["CE"][key][p] = {}
        for nm in PV:
            if nm == "nopr": continue
            dpr = DV[nm][a][u]["TC"][p] - v0; dall = DV[nm][a][u]["TC"][p] - vtc
            R["CE"][key][p][nm] = {"primer_increment": qf(dpr / vt * C_T / C_pr) if dpr > 0 else [0.0, 0.0, 0.0],
                                   "booklet_plus_primer": qf(dall / vt * C_T / (C_P(100) + C_pr)) if dall > 0 else [0.0, 0.0, 0.0]}
        R["CE"][key][p]["064_booklet"] = r65["CE"][key][p]["064_booklet"]

# ---------- 仮説 ----------
H = {}; h = lambda nm, k=M1: RT[nm][k]["heavy"]; h66 = r66["ratio_to_V0"]["main"][M1]["heavy"]
v = R["stage_vs_time"]["slope"]; H["H79a"] = {"value": [v] + R["stage_vs_time"]["slope_q05_95"], "rule": "slope onset~loss < 0.2", "supported": v < 0.2}
v = R["new_access"]["median"]; H["H79b"] = {"value": v, "rule": "median onset lag after new access <= 10 y", "supported": v <= 10}
v = [k_sys, exp_sys]; H["H79c"] = {"value": v, "rule": "0 systematic onsets before 1600 and exposure >= 5000", "supported": k_sys == 0 and exp_sys >= 5000}
v = [h("main"), h66]; H["H79d"] = {"value": v, "rule": "main heavy <= 066 main - 0.03", "supported": v[0] <= v[1] - 0.03}
sc = R["scan_Delta"]; half = (h66 + h("nopr")) / 2; dh = None
for i in range(1, len(sc)):
    if sc[i - 1][M1] < half <= sc[i][M1]:
        dh = sc[i - 1]["Delta"] + (half - sc[i - 1][M1]) / (sc[i][M1] - sc[i - 1][M1]) * (sc[i]["Delta"] - sc[i - 1]["Delta"]); break
if dh is None and sc[0][M1] >= half: dh = 0.0
R["Delta_half"] = dh
H["H79e"] = {"value": [dh, half], "rule": "Delta recovering half of (066 main - nopr) <= 100", "supported": dh is not None and dh <= 100}
v = R["near174"]["strict"]; H["H79f"] = {"value": v, "rule": "strict teacherless non-scholar cases >= 3", "supported": v >= 3}
v = [R["CE"][M1]["total"]["main"]["primer_increment"][1], R["CE"][M1]["total"]["064_booklet"][1]]
H["H79g"] = {"value": v, "rule": "main primer increment CE (total, median) < booklet-only", "supported": v[0] < v[1]}
v = h("W20") - h("main"); H["H79h"] = {"value": v, "rule": "W20 - main >= 0.01", "supported": v >= 0.01}
for xx in H.values(): xx["supported"] = bool(xx["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for c in CO:
    if c["loss"] is None: continue
    on = c["onset"] if c["onset"] is not None else c.get("onset_ub")
    if on is None: continue
    col = "C0" if c["visible"] else "C1"; mk = "o" if c["onset"] is not None else "v"
    axs[0].plot([c["loss"], c["access"]], [on, on], "-", color=col, lw=0.6, alpha=0.6)
    axs[0].plot(c["loss"], on, mk, color=col); axs[0].text(c["loss"] + 30, on + 8, c["id"], fontsize=6)
xs = np.linspace(-2000, 1700, 10)
axs[0].plot(xs, xs + np.median(y - x), "k:", lw=0.8, label="time-tied (onset = loss + median lag)")
axs[0].axhline(np.median(y), color="C2", ls="--", lw=0.8, label=f"stage-tied (median onset {np.median(y):.0f})")
axs[0].set_xlabel("year writing became unreadable (loss)"); axs[0].set_ylabel("year of first systematic engagement (onset)")
axs[0].set_title(f"onset vs loss: slope {R['stage_vs_time']['slope']:.3f} (90% {bs.min()*0+R['stage_vs_time']['slope_q05_95'][0]:.3f}..{R['stage_vs_time']['slope_q05_95'][1]:.3f}); blue visible, orange buried", fontsize=7)
axs[0].legend(fontsize=7); axs[0].set_ylim(1550, 2000)
Ds = [r["Delta"] for r in sc]
axs[1].semilogx([max(dd, 10) for dd in Ds], [r[M1] for r in sc], "o-", label="A_Ord/U3 heavy")
axs[1].semilogx([max(dd, 10) for dd in Ds], [r["A_XPT_S/U2"] for r in sc], "s--", label="A_XPT_S/U2 heavy")
axs[1].axhline(h66, color="k", ls=":", lw=0.8, label="066 main (engage from collapse)")
axs[1].axhline(h("nopr"), color="C3", ls=":", lw=0.8, label="no primer")
axs[1].set_xlabel("Delta: years before T2 when the society reaches the 'scholarly' stage (0 plotted at 10)")
axs[1].set_ylabel("heavy-collapse TC / V0"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_onset_scan.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4.6))
names = ["nopr", "main", "sys", "pre_unkeyed", "lead25", "lead50", "lead100", "lead200", "lead400", "W20", "W50", "e066"]
yy = np.arange(len(names))
ax.barh(yy - 0.2, [RT[n][M1]["heavy"] for n in names], height=0.4, label="A_Ord/U3")
ax.barh(yy + 0.2, [RT[n]["A_XPT_S/U2"]["heavy"] for n in names], height=0.4, color="C1", alpha=0.7, label="A_XPT_S/U2")
ax.set_yticks(yy); ax.set_yticklabels([f"{n} (p_pr {R['p_pr_at_T_U3_heavy'].get(n, 0):.2f})" for n in names], fontsize=7)
ax.set_xlim(0.75, 0.9); ax.axvline(h66, color="k", ls=":", lw=0.7); ax.set_xlabel("heavy-collapse TC / V0"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check", R["check_066_main"])
print("stage_vs_time", json.dumps(R["stage_vs_time"]))
print("premodern", json.dumps(R["premodern"]))
print("new_access", json.dumps(R["new_access"]))
print("near174 strict", R["near174"]["strict"])
print("draws", json.dumps({k: [round(q, 6) for q in v["q05_50_95"]] for k, v in R["draws"].items()}))
print("p_pr", json.dumps({k: round(v, 4) for k, v in R["p_pr_at_T_U3_heavy"].items()}), "T2s med", R["T2s_heavy_U3_median"])
for nm in RT: print(nm, {k: {p: round(RT[nm][k][p], 4) for p in PARTS} for k in KEYS})
print("scan", json.dumps(R["scan_Delta"])); print("Delta_half", R["Delta_half"])
print("CE total", json.dumps({k: v for k, v in R["CE"][M1]["total"].items()}))
print("CE heavy", json.dumps({k: v for k, v in R["CE"][M1]["heavy"].items()}))
for k, xx in H.items(): print(k, xx["supported"], xx["value"])
print("runtime", R["runtime_s"])
