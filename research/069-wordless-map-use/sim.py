"""研究069：文字を読まずに使える座標の割合 p_map（open_problems 180＋176）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
068 の sim.py を行頭の「# ---------- 版 ----------」の前まで exec し、067・068 の主を p_map なしで差0で再現してから、
重い崩壊で文字が切れた標本の「使える」確率を pd = 1 − (1 − p_dec)(1 − p_pr(T))(1 − p_map) に替える。新しい乱数は seed 20261069。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P68 = os.path.abspath(os.path.join(HERE, "..", "068-scholarly-vs-lens-order", "sim.py"))
src = open(P68, encoding="utf8").read(); cut = src.index("\n# ---------- 版 ----------\n")
G68 = {"__file__": P68, "__name__": "sim068"}
exec(compile(src[:cut], P68, "exec"), G68)
r68 = json.load(open(os.path.join(HERE, "..", "068-scholarly-vs-lens-order", "results.json"), encoding="utf8"))
G67 = G68["G67"]; G66 = G67["G66"]; G65 = G67["G65"]
N = G68["N"]; ver = G68["ver"]; get = G68["get"]; ratio_all = G68["ratio_all"]; L66 = G68["L66"]; qf = G68["qf"]
C_T = G68["C_T"]; C_P = G68["C_P"]; C_pr = G68["C_pr"]; pp67 = G68["pp67"]; pp68 = G68["pp68"]; r65 = G68["r65"]; r67 = G68["r67"]
hmU = G68["hmU"]; TU = G68["TU"]
lam = G66["lam"]; rho1 = G66["rho1"]; r_reach = G66["r_reach"]; S_ind = G66["S_ind"]; f_i = G66["f_i"]; taup_061 = G66["taup_061"]
heavy_mask = G65["heavy_mask"]; tau_dec = G65["tau_dec"]
KEYS = ["A_Ord/U3", "A_XPT_S/U2"]; PARTS = ("total", "heavy", "light"); M1 = "A_Ord/U3"
R = {"study": "069-wordless-map-use", "seed": 20261069}

def R69(T, pc, ppf, pm):
    """065 の R65 に地図の経路を足す。pm=None なら R65 と同じ演算の順で同じ値。"""
    hm = heavy_mask(T); pd = 1 - np.exp(-T / tau_dec)
    if ppf is not None: pd = 1 - (1 - pd) * (1 - ppf(T))
    if pm is not None: pd = 1 - (1 - pd) * (1 - pm)
    read = pc + (1 - pc) * pd
    return np.where(hm, read, 1.0)
def rho69(Lv, ppf, pm=None, n_p=100, K=20):
    def f(T):
        pc = (1 - Lv) * np.exp(-lam * T)
        return 1 - (1 - rho1) * (1 - r_reach * R69(T, pc, ppf, pm) * S_ind(T, n_p, K, f_i, taup_061))
    return f

# ---------- 検査：067・068 の主を差0で再現 ----------
RT = {}; DV = {}
RT["chk067"], DV["chk067"] = ratio_all(ver(rho=rho69(L66, pp67())))
RT["chk0"], DV["chk0"] = ratio_all(ver(rho=rho69(L66, pp68())))
d67 = max(abs(RT["chk067"][k][p] - r67["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
d68 = max(abs(RT["chk0"][k][p] - r68["ratio_to_V0"]["main"][k][p]) for k in KEYS for p in PARTS)
R["check_067_main"] = {"max_abs_diff": d67, "reproduced": d67 < 1e-12}
R["check_068_main"] = {"max_abs_diff": d68, "reproduced": d68 < 1e-12}

# ---------- 資料 ----------
CJ = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
MP, LM, PI, HI = CJ["map"], CJ["landmark"], CJ["picto"], CJ["hist"]
corr = lambda p, c: max(0.0, (p - c) / (1 - c))
R["evidence"] = {"map": [{"id": c["id"], "p": c["p"], "chance": c["chance"], "corrected": corr(c["p"], c["chance"]), "n": c["n"]} for c in MP],
                 "landmark": [{"id": c["id"], "p": c["p"], "corrected": corr(c["p"], c["chance"])} for c in LM],
                 "picto": [{"id": c["id"], "p": c["p"], "low": c["low"], "n": c["n"]} for c in PI],
                 "hist": [{"id": c["id"], "y": c["y"]} for c in HI],
                 "n_numeric_sources_opened": len(CJ["switch_rule_count"]["sources"])}
mc = [corr(c["p"], c["chance"]) for c in MP]; lc = [corr(c["p"], c["chance"]) for c in LM]; pv = [c["p"] for c in PI]
R["evidence"]["summary"] = {"map_corrected_median": float(np.median(mc)), "map_corrected_mean": float(np.mean(mc)), "map_raw_mean": float(np.mean([c["p"] for c in MP])),
                            "landmark_corrected_mean": float(np.mean(lc)), "picto_median": float(np.median(pv)), "picto_mean": float(np.mean(pv)),
                            "picto_low_mean": float(np.mean([c["p"] for c in PI if c["low"]])), "hist_y": float(sum(c["y"] for c in HI)), "hist_n": len(HI)}

# ---------- 新しい乱数 ----------
g = np.random.default_rng(20261069)
def draw(cases, correct=True, raw_out=False):
    i = g.integers(0, len(cases), N)
    p = np.array([c["p"] for c in cases])[i]; n = np.array([c["n"] if c.get("n") else 30 for c in cases], float)[i]
    x = g.beta(0.5 + p * n, 0.5 + (1 - p) * n)
    if not correct: return x
    ch = np.array([c["chance"] for c in cases])[i]
    k = np.clip((x - ch) / (1 - ch), 0.0, 1.0)
    return (k, x) if raw_out else k
k_match, k_match_raw = draw(MP, raw_out=True)
k_land = draw(LM)
k_unsch = draw([MP[0]])
k_con = draw(PI, correct=False)
k_con_low = draw([c for c in PI if c["low"]], correct=False)
hy = R["evidence"]["summary"]["hist_y"]; k_hist = g.beta(0.5 + hy, 0.5 + len(HI) - hy, N)
k_geo = g.uniform(0.5, 1.0, N); k_geo_low = g.uniform(0.2, 0.6, N)
PM = {"main": k_match * k_con * k_geo, "raw": k_match_raw * k_con * k_geo, "landmark": k_land * k_con * k_geo, "unschooled": k_unsch * k_con * k_geo,
      "p_low": k_match * k_con_low * k_geo, "match_only": k_match * k_geo, "geo1": k_match * k_con, "geo_low": k_match * k_con * k_geo_low,
      "hist": k_match * k_hist * k_geo, "best": k_land}
R["draws"] = {nm: {"mean": float(v.mean()), "q05_50_95": [float(q) for q in np.quantile(v, [0.05, 0.5, 0.95])]} for nm, v in
              {"k_match": k_match, "k_match_raw": k_match_raw, "k_landmark": k_land, "k_unschooled": k_unsch, "k_concept": k_con, "k_concept_low": k_con_low,
               "k_hist": k_hist, "k_geo": k_geo, **{"p_map_" + k: v for k, v in PM.items()}}.items()}
R["p_map_main_mean"] = float(PM["main"].mean())

# ---------- 版 ----------
VS = {nm: (pp68(), pm) for nm, pm in PM.items()}
VS["nopr"] = (None, None); VS["map_nopr"] = (None, PM["main"]); VS["map_R1fast"] = (pp68(fast=True), PM["main"]); VS["R1fast"] = (pp68(fast=True), None)
for nm, (ppf, pm) in VS.items():
    RT[nm], DV[nm] = ratio_all(ver(rho=rho69(L66, ppf, pm)))
RT["068_main"] = r68["ratio_to_V0"]["main"]; RT["067_main"] = r67["ratio_to_V0"]["main"]; RT["066_main"] = r67["ratio_to_V0"]["066_main"]
R["ratio_to_V0"] = RT
# 重い崩壊の T（U3）での「読めない標本で使える」確率（primer と地図を合わせた分）
pr68 = pp68()(TU)
R["use_prob_U3_heavy"] = {"primer_068": float(pr68[hmU].mean()), "map_main": float(PM["main"][hmU].mean()),
                          "primer_or_map_main": float((1 - (1 - pr68) * (1 - PM["main"]))[hmU].mean())}
R["scan_pmap"] = []
for pmv in [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]:
    rr, _ = ratio_all(ver(rho=rho69(L66, pp68(), np.full(N, pmv))))
    R["scan_pmap"].append({"p_map": pmv, M1: rr[M1]["heavy"], "A_XPT_S/U2": rr["A_XPT_S/U2"]["heavy"], "total_" + M1: rr[M1]["total"]})

# ---------- 費用あたり（TAB=1、065〜068 と同じ式。地図の費用は primer と同じ抽出 C_pr【推測】） ----------
C_map = C_pr
R["CE"] = {}
for key in KEYS:
    a, u = key.split("/"); R["CE"][key] = {}
    for p in ("total", "heavy"):
        vt = get("main", key)["TAB"][p]; vtc = get("main", key)["TC"][p]; R["CE"][key][p] = {}
        base68 = DV["chk0"][a][u]["TC"][p]; base_np = DV["nopr"][a][u]["TC"][p]
        for nm in list(PM) + ["map_R1fast"]:
            b = DV["R1fast"][a][u]["TC"][p] if nm == "map_R1fast" else base68
            dm = DV[nm][a][u]["TC"][p] - b; dall = DV[nm][a][u]["TC"][p] - vtc
            R["CE"][key][p][nm] = {"map_increment": qf(dm / vt * C_T / C_map) if dm > 0 else [0.0, 0.0, 0.0],
                                   "booklet_primer_map": qf(dall / vt * C_T / (C_P(100) + C_pr + C_map)) if dall > 0 else [0.0, 0.0, 0.0]}
        dm = DV["map_nopr"][a][u]["TC"][p] - base_np; dall = DV["map_nopr"][a][u]["TC"][p] - vtc
        R["CE"][key][p]["map_nopr"] = {"map_increment_vs_nopr": qf(dm / vt * C_T / C_map) if dm > 0 else [0.0, 0.0, 0.0],
                                       "booklet_map": qf(dall / vt * C_T / (C_P(100) + C_map)) if dall > 0 else [0.0, 0.0, 0.0]}
        dpr = base68 - base_np
        R["CE"][key][p]["primer_increment_068"] = qf(dpr / vt * C_T / C_pr) if dpr > 0 else [0.0, 0.0, 0.0]
        R["CE"][key][p]["064_booklet"] = r65["CE"][key][p]["064_booklet"]

# ---------- 仮説 ----------
H = {}; h = lambda nm, k=M1: RT[nm][k]["heavy"]; h68 = r68["ratio_to_V0"]["main"][M1]["heavy"]; S = R["evidence"]["summary"]
v = S["map_corrected_median"]; H["H81a"] = {"value": v, "rule": "median corrected map rate >= 0.40 (known at registration)", "supported": v >= 0.40}
v = [S["landmark_corrected_mean"], S["map_corrected_mean"]]; H["H81b"] = {"value": v, "rule": "landmark corrected mean >= map corrected mean + 0.25 (known)", "supported": v[0] >= v[1] + 0.25}
v = S["picto_median"]; H["H81c"] = {"value": v, "rule": "picto median in [0.5, 0.8] (known)", "supported": 0.5 <= v <= 0.8}
v = [h("main"), h68, h("main") - h68]; H["H81d"] = {"value": v, "rule": "main heavy - 068 main >= 0.010", "supported": v[2] >= 0.010}
H["H81e"] = {"value": v, "rule": "main heavy - 068 main <= 0.030", "supported": v[2] <= 0.030}
ce = R["CE"][M1]["total"]
v = [ce["main"]["map_increment"][1], ce["primer_increment_068"][1]]; H["H81f"] = {"value": v, "rule": "map increment CE (total, median) > 068 primer increment CE", "supported": v[0] > v[1]}
v = [ce["main"]["map_increment"][1], ce["064_booklet"][1]]; H["H81g"] = {"value": v, "rule": "map increment CE (total, median) < booklet-only (064)", "supported": v[0] < v[1]}
v = [h("landmark"), h("main")]; H["H81h"] = {"value": v, "rule": "landmark heavy >= main + 0.005", "supported": v[0] >= v[1] + 0.005}
VN = ["main", "raw", "landmark", "unschooled", "p_low", "match_only", "geo1", "geo_low", "hist", "map_nopr", "map_R1fast", "best"]
vs = [h(n) for n in VN]; H["H81i"] = {"value": [min(vs), max(vs), max(vs) - min(vs)], "rule": "range of heavy across versions <= 0.04", "supported": max(vs) - min(vs) <= 0.04}
v = R["evidence"]["n_numeric_sources_opened"]; H["H81j"] = {"value": v, "rule": "numeric comprehension sources opened >= 5 (else switch to B)", "supported": v >= 5}
for xx in H.values(): xx["supported"] = bool(xx["supported"])
R["hypotheses"] = H

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
rows = [(c["id"], c["p"], corr(c["p"], c["chance"]), "C0") for c in MP] + [(c["id"], c["p"], corr(c["p"], c["chance"]), "C2") for c in LM] + \
       [(c["id"], c["p"], c["p"], "C1") for c in PI] + [(c["id"], c["y"], c["y"], "C7") for c in HI]
yy = np.arange(len(rows))[::-1]
for (nm, raw, cv, col), y in zip(rows, yy):
    axs[0].plot([raw], [y], "o", mfc="none", color=col); axs[0].plot([cv], [y], "o", color=col); axs[0].plot([min(raw, cv), max(raw, cv)], [y, y], "-", color=col, lw=0.8)
axs[0].set_yticks(yy); axs[0].set_yticklabels([r[0] for r in rows], fontsize=7); axs[0].set_xlim(-0.02, 1.02)
axs[0].set_xlabel("rate (open: as reported; filled: chance-corrected)")
axs[0].set_title("blue: map tasks (unschooled / pre-literate); green: landmark trials;\norange: pictogram comprehension (low-literacy / workers); grey: ancient wordless maps recognised as maps", fontsize=7)
sc = R["scan_pmap"]
axs[1].plot([r["p_map"] for r in sc], [r[M1] for r in sc], "o-", label="A_Ord/U3 heavy")
axs[1].plot([r["p_map"] for r in sc], [r["A_XPT_S/U2"] for r in sc], "s--", label="A_XPT_S/U2 heavy")
axs[1].axhline(h68, color="k", ls=":", lw=0.8, label="068 main (no map)"); axs[1].axhline(RT["066_main"][M1]["heavy"], color="C2", ls=":", lw=0.8, label="066 main")
axs[1].axvline(R["p_map_main_mean"], color="C1", lw=0.8, label="p_map main (mean)")
axs[1].set_xlabel("p_map: probability that the wordless map part is usable (constant)"); axs[1].set_ylabel("heavy-collapse TC / V0"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_evidence_scan.png"), dpi=130); plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 5))
names = ["nopr", "chk0", "main", "raw", "landmark", "unschooled", "p_low", "match_only", "geo1", "geo_low", "hist", "map_nopr", "R1fast", "map_R1fast", "best"]
yy = np.arange(len(names))
ax.barh(yy - 0.2, [RT[n][M1]["heavy"] for n in names], height=0.4, label="A_Ord/U3")
ax.barh(yy + 0.2, [RT[n]["A_XPT_S/U2"]["heavy"] for n in names], height=0.4, color="C1", alpha=0.7, label="A_XPT_S/U2")
lab = lambda n: f"{n} (p_map {PM[n].mean():.2f})" if n in PM else (f"{n} (p_map {PM['main'].mean():.2f})" if n.startswith("map_") else n)
ax.set_yticks(yy); ax.set_yticklabels([lab(n) for n in names], fontsize=7)
ax.set_xlim(0.78, 0.90); ax.axvline(h68, color="k", ls=":", lw=0.7); ax.set_xlabel("heavy-collapse TC / V0"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_versions.png"), dpi=130); plt.close(fig)

R["runtime_s"] = time.time() - t0
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float, ensure_ascii=False)
print("check067", R["check_067_main"], "check068", R["check_068_main"])
print("evidence", json.dumps(R["evidence"]["summary"]))
print("draws", json.dumps({k: [round(x, 3) for x in v["q05_50_95"]] + [round(v["mean"], 3)] for k, v in R["draws"].items()}))
print("use_prob", json.dumps(R["use_prob_U3_heavy"]))
for nm in RT: print(nm, {k: {p: round(RT[nm][k][p], 4) for p in PARTS} for k in KEYS})
print("scan", json.dumps(R["scan_pmap"]))
print("CE total", json.dumps(R["CE"][M1]["total"])); print("CE heavy", json.dumps(R["CE"][M1]["heavy"]))
print("CE XPT total", json.dumps({k: R["CE"]["A_XPT_S/U2"]["total"][k] for k in ("main", "primer_increment_068", "064_booklet")}))
for k, xx in H.items(): print(k, xx["supported"], xx["value"])
print("runtime", R["runtime_s"])
