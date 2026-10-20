"""研究068：学問の段（文献学の段 R1／読めない文字への取り組み R2）と読取装置の段（L）の順（open_problems 179）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
067 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し 067 の主を差0で再現してから、
primer の経路の取り組みを「崩壊〜R1：λ_any、R1〜R2：λ1、R2 の後：すぐ」の三区間の率に替える。新しい乱数は seed 20261068。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P67 = os.path.abspath(os.path.join(HERE, "..", "067-engagement-onset", "sim.py"))
src = open(P67, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G67 = {"__file__": P67, "__name__": "sim067"}
exec(compile(src[:cut], P67, "exec"), G67)
r67 = json.load(open(os.path.join(HERE, "..", "067-engagement-onset", "results.json"), encoding="utf8"))
r65 = G67["r65"]
N = G67["N"]; ver = G67["ver"]; get = G67["get"]; ratio_all = G67["ratio_all"]; rho66 = G67["rho66"]; L66 = G67["L66"]
qf = G67["qf"]; C_T = G67["C_T"]; C_P = G67["C_P"]; C_pr = G67["C_pr"]; pp67 = G67["pp67"]
k_rec = G67["k_rec"]; k_cont = G67["k_cont"]; tau_key = G67["tau_key"]; lam_any = G67["lam_any"]
B = G67["G65"]["B"]; hmU = np.isin(B["U3"]["st"], (4, 5)); TU = np.maximum(B["U3"]["T2s"], 0)
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"
R = {"study": "068-scholarly-vs-lens-order", "seed": 20261068}

# ---------- 資料 ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
SO = CJ["societies"]
def d1(s, alt=False):
    r1 = (s["R1_alt"] if (alt and s["R1_alt"] is not None) else s["R1"])
    if s["L_censored"]: return s["L_lb"] - r1, True
    L = (s["L_alt"] if (alt and s["L_alt"] is not None) else s["L"])
    return L - r1, False
D1 = {s["id"]: d1(s)[0] for s in SO}; D1alt = {s["id"]: d1(s, True)[0] for s in SO}; CENS = {s["id"]: d1(s)[1] for s in SO}
nonEU_point = [s["id"] for s in SO if s["region"] == "nonEurope" and not s["L_censored"]]
R["lead_R1"] = {"Delta1": D1, "Delta1_alt": D1alt, "censored": CENS, "positive": int(sum(v > 0 for v in D1.values())), "n": len(SO),
                "nonEurope_point_pairs": nonEU_point, "median_uncensored_lower_bound": float(np.median(list(D1.values())))}
eu = [s for s in SO if s["id"] == "Europe_Italy"][0]
R["europe"] = {"R2_minus_L": eu["R2"] - eu["L"], "R2alt_minus_L": eu["R2_alt"] - eu["L"], "R1_to_R2": eu["R2"] - eu["R1"], "R1_to_R2_alt": eu["R2_alt"] - eu["R1"],
               "eyeglasses_minus_R1": eu["eyeglasses"] - eu["R1"]}
def expo(end_key="end_preL", half=False, count_eu_event=False):
    E = 0.0; k = 0; parts = {}
    for s in SO:
        x = s["exposure"]
        if x is None: continue
        end = x.get(end_key, x["end_preL"])
        if count_eu_event and s["id"] == "Europe_Italy": end = s["R2"]; k += 1
        e = (end - x["start"]) * (0.5 if half else 1.0); parts[s["id"]] = e; E += e; k += x["k_sys_preL"]
    return E, k, parts
E1, k1, parts1 = expo()
R["R1_exposure"] = {"main": {"E": E1, "k": k1, "parts": parts1}, "abb1600": dict(zip(("E", "k", "parts"), expo("end_alt"))),
                    "k1": dict(zip(("E", "k", "parts"), expo(count_eu_event=True))), "exp_half": dict(zip(("E", "k", "parts"), expo(half=True)))}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261068)
ids = [s["id"] for s in SO]
pick = g.integers(0, len(ids), N)
def Dl_arr(dd, cens_mult=1.0):
    v = np.array([dd[i] * (cens_mult if CENS[i] else 1.0) for i in ids], float); return v[pick]
D1_main = Dl_arr(D1); D1_alt = Dl_arr(D1alt); D1_c2 = Dl_arr(D1, 2.0)
zstd = g.standard_gamma(0.5, N)          # 共通の乱数：λ1 = Gamma(0.5+k)/E は k=0 なら zstd/E
zstd1 = g.standard_gamma(1.5, N)         # k=1 の版
def lam1(E, k): return (zstd if k == 0 else zstd1) / E
L1 = {"main": lam1(E1, k1), "abb1600": lam1(R["R1_exposure"]["abb1600"]["E"], R["R1_exposure"]["abb1600"]["k"]),
      "k1": lam1(R["R1_exposure"]["k1"]["E"], R["R1_exposure"]["k1"]["k"]), "exp_half": lam1(R["R1_exposure"]["exp_half"]["E"], R["R1_exposure"]["exp_half"]["k"])}
R["draws"] = {nm: {"tau1_q05_50_95": [float(q) for q in np.quantile(1 / v, [0.05, 0.5, 0.95])]} for nm, v in L1.items()}
R["draws"]["tau_key_median"] = float(np.median(tau_key)); R["draws"]["tau_any_median"] = float(np.median(1 / lam_any))
R["draws"]["Delta1_main_q"] = [float(q) for q in np.quantile(D1_main, [0.05, 0.5, 0.95])]

def seg(S, a, b, lam, D, tk):
    """[a,b) で率 lam で取り組み、取り組んでから時定数 tk で読む。D までに読めた確率への寄与と b での生存。安定な形。"""
    Lg = b - a; c = 1 / tk - lam
    part = (1 - np.exp(-lam * Lg)) - lam * (np.exp(-(D - b) / tk - lam * Lg) - np.exp(-(D - a) / tk)) / c
    return S * part, S * np.exp(-lam * Lg)
def three_phase(T, D1v, lam1v, D2v, W=0.0, fast=False, tk=None):
    tk = tau_key if tk is None else tk
    D = T + W
    T1 = np.clip(T - D1v, 0.0, D)
    T2p = np.clip(T - D2v, T1, D)
    S = np.ones_like(T)
    p0, S = seg(S, 0.0, T1, lam_any, D, tk)
    if fast:
        return p0 + S * (1 - np.exp(-(D - T1) / tk))
    p1, S = seg(S, T1, T2p, lam1v, D, tk)
    return p0 + p1 + S * (1 - np.exp(-(D - T2p) / tk))
def pp68(D1v=None, lamv=None, D2=None, W=0.0, fast=False):
    D1v = D1_main if D1v is None else D1v; lamv = L1["main"] if lamv is None else lamv
    D2 = float(eu["L"] - eu["R2"]) if D2 is None else D2
    return lambda T: k_rec * k_cont * three_phase(T, D1v, lamv, D2, W, fast)

# ---------- 版 ----------
PV = {"e067": pp67(), "nopr": None,
      "chk_D0": pp68(D1v=np.zeros(N), D2=-1e9),
      "main": pp68(), "R1_fast": pp68(fast=True), "k1": pp68(lamv=L1["k1"]), "abb1600": pp68(lamv=L1["abb1600"]),
      "exp_half": pp68(lamv=L1["exp_half"]), "R1alt": pp68(D1v=D1_alt), "cens2x": pp68(D1v=D1_c2), "eur_only": pp68(D1v=np.full(N, float(D1["Europe_Italy"]))),
      "W20": pp68(W=20.0), "W20_R2alt": pp68(W=20.0, D2=float(eu["L"] - eu["R2_alt"])), "R1_fast_cens2x": pp68(D1v=D1_c2, fast=True)}
RT = {}; DV = {}
for nm, f in PV.items():
    RT[nm], DV[nm] = ratio_all(ver(rho=rho66(L66, ppf=f)))
d = max(abs(RT["e067"][k][p] - r67["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_067_main"] = {"max_abs_diff": d, "reproduced": d < 1e-12}
d2 = max(abs(RT["chk_D0"][k][p] - r67["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_threephase_D0_equals_067"] = {"max_abs_diff": d2, "ok": d2 < 1e-9}
RT["067_main"] = r67["ratio_to_V0"]["main"]; RT["066_main"] = r67["ratio_to_V0"]["066_main"]
R["ratio_to_V0"] = RT
R["p_pr_at_T_U3_heavy"] = {nm: float(f(TU)[hmU].mean()) for nm, f in PV.items() if f is not None}
# τ1 の走査（Δ1 は main の抽出）
R["scan_tau1"] = []
for t1 in [65, 100, 300, 1000, 3000, 10000]:
    rr, _ = ratio_all(ver(rho=rho66(L66, ppf=pp68(lamv=np.full(N, 1.0 / t1)))))
    R["scan_tau1"].append({"tau1": t1, M1: rr[M1]["heavy"], "A_XPT_S/U2": rr["A_XPT_S/U2"]["heavy"]})

# ---------- 費用あたり ----------
R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for p in ("total", "heavy"):
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; v0 = DV["nopr"][a][u]["TC"][p]; R["CE"][key][p] = {}
        for nm in PV:
            if nm in ("nopr", "chk_D0"): continue
            dpr = DV[nm][a][u]["TC"][p] - v0; dall = DV[nm][a][u]["TC"][p] - vtc
            R["CE"][key][p][nm] = {"primer_increment": qf(dpr / vt * C_T / C_pr) if dpr > 0 else [0.0, 0.0, 0.0],
                                   "booklet_plus_primer": qf(dall / vt * C_T / (C_P(100) + C_pr)) if dall > 0 else [0.0, 0.0, 0.0]}
        R["CE"][key][p]["064_booklet"] = r65["CE"][key][p]["064_booklet"]

# ---------- 仮説 ----------
H = {}; h = lambda nm, k=M1: RT[nm][k]["heavy"]; h67 = r67["ratio_to_V0"]["main"][M1]["heavy"]
v = [R["lead_R1"]["positive"], R["lead_R1"]["n"]]; H["H80a"] = {"value": v, "rule": "Delta1 > 0 in all societies", "supported": v[0] == v[1]}
v = [k1, E1]; H["H80b"] = {"value": v, "rule": "k1 == 0 and E1 >= 1000", "supported": k1 == 0 and E1 >= 1000}
v = R["europe"]["R2_minus_L"]; H["H80c"] = {"value": v, "rule": "0 <= R2 - L <= 50 (Europe)", "supported": 0 <= v <= 50}
v = [h("main"), h67]; H["H80d"] = {"value": v, "rule": "main heavy <= 067 main + 0.01", "supported": v[0] <= v[1] + 0.01}
v = [h("R1_fast"), h67]; H["H80e"] = {"value": v, "rule": "R1_fast heavy >= 067 main + 0.025", "supported": v[0] >= v[1] + 0.025}
v = [R["draws"]["main"]["tau1_q05_50_95"][1], R["draws"]["tau_key_median"]]; H["H80f"] = {"value": v, "rule": "median 1/lambda1 >= 10 * median tau_key", "supported": v[0] >= 10 * v[1]}
v = [R["CE"][M1]["total"]["main"]["primer_increment"][1], R["CE"][M1]["total"]["064_booklet"][1]]
H["H80g"] = {"value": v, "rule": "main primer increment CE (total, median) < booklet-only", "supported": v[0] < v[1]}
v = len(nonEU_point); H["H80h"] = {"value": v, "rule": "non-European societies with both point years >= 3", "supported": v >= 3}
for xx in H.values(): xx["supported"] = bool(xx["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for i, s in enumerate(SO):
    y = len(SO) - i
    r1 = s["R1"]; Lv = s["L"] if not s["L_censored"] else s["L_lb"]
    axs[0].plot([r1, Lv], [y, y], "-", color="C0" if not s["L_censored"] else "C7", lw=2)
    axs[0].plot(r1, y, "o", color="C2"); axs[0].plot(Lv, y, ">" if s["L_censored"] else "s", color="C3")
    if s["exposure"] is not None:
        x = s["exposure"]; axs[0].plot([x["start"], x["end_preL"]], [y - 0.18, y - 0.18], ":", color="C1", lw=2)
    if s["R2"] is not None: axs[0].plot(s["R2"], y, "*", color="k", ms=10)
    axs[0].text(r1, y + 0.18, f"{s['id']}  Δ1={D1[s['id']]}{'+' if CENS[s['id']] else ''}" + (" (diffused L)" if s["diffused"] else ""), fontsize=7)
axs[0].set_yticks([]); axs[0].set_xlabel("year")
axs[0].set_title("green: R1 philology onset; red: lens (telescope-grade) L (> = censored at 1608);\norange dotted: exposure to unread script before L (0 systematic engagements); star: R2", fontsize=7)
axs[1].semilogx([r["tau1"] for r in R["scan_tau1"]], [r[M1] for r in R["scan_tau1"]], "o-", label="A_Ord/U3 heavy (Δ1 from 5 societies)")
axs[1].semilogx([r["tau1"] for r in R["scan_tau1"]], [r["A_XPT_S/U2"] for r in R["scan_tau1"]], "s--", label="A_XPT_S/U2 heavy")
axs[1].axhline(h67, color="k", ls=":", lw=0.8, label="067 main (Δ=0)"); axs[1].axhline(RT["066_main"][M1]["heavy"], color="C2", ls=":", lw=0.8, label="066 main")
axs[1].axhline(h("nopr"), color="C3", ls=":", lw=0.8, label="no primer")
axs[1].axvline(R["draws"]["main"]["tau1_q05_50_95"][1], color="C1", lw=0.8, label="posterior median τ1 (main)")
axs[1].set_xlabel("τ1: time constant of engagement once a society is at the philology stage R1 (years)"); axs[1].set_ylabel("heavy-collapse TC / V0"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_order_and_tau1.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 4.8))
names = ["nopr", "e067", "main", "k1", "abb1600", "exp_half", "R1alt", "cens2x", "eur_only", "W20", "W20_R2alt", "R1_fast", "R1_fast_cens2x"]
yy = np.arange(len(names))
ax.barh(yy - 0.2, [RT[n][M1]["heavy"] for n in names], height=0.4, label="A_Ord/U3")
ax.barh(yy + 0.2, [RT[n]["A_XPT_S/U2"]["heavy"] for n in names], height=0.4, color="C1", alpha=0.7, label="A_XPT_S/U2")
ax.set_yticks(yy); ax.set_yticklabels([f"{n} (p_pr {R['p_pr_at_T_U3_heavy'].get(n, 0):.2f})" for n in names], fontsize=7)
ax.set_xlim(0.78, 0.89); ax.axvline(h67, color="k", ls=":", lw=0.7); ax.set_xlabel("heavy-collapse TC / V0"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check067", R["check_067_main"], "check3", R["check_threephase_D0_equals_067"])
print("lead_R1", json.dumps(R["lead_R1"], ensure_ascii=False)); print("europe", R["europe"])
print("R1_exposure", json.dumps(R["R1_exposure"])); print("draws", json.dumps(R["draws"]))
print("p_pr", json.dumps({k: round(v, 4) for k, v in R["p_pr_at_T_U3_heavy"].items()}))
for nm in RT: print(nm, {k: {p: round(RT[nm][k][p], 4) for p in PARTS} for k in KEYS})
print("scan", json.dumps(R["scan_tau1"]))
print("CE total", json.dumps({k: v for k, v in R["CE"][M1]["total"].items()}))
print("CE heavy", json.dumps({k: v for k, v in R["CE"][M1]["heavy"].items()}))
for k, xx in H.items(): print(k, xx["supported"], xx["value"])
print("runtime", R["runtime_s"])
