"""研究089 第2部：短い尺度の β（episodes.py）を 075 の群の点の重み pop^e に入れて重い崩壊の TC/V0 を出す。
085 と同じ手順：075 の sim.py を「PP = G74["PP"]」の前まで exec（070・074・075 の差0の検査を含む）、085 の e=0.6615 の mesh35・mesh200 を差0で再現。
計算のみ、新しい乱数なし。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"""
import json, os, time, subprocess, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
subprocess.run([sys.executable, os.path.join(HERE, "episodes.py")], check=True, stdout=subprocess.DEVNULL)
EPI = json.load(open(os.path.join(HERE, "episodes.json"), encoding="utf8"))["episodes"]
p75 = os.path.abspath(os.path.join(RS, "075-reach-to-tc-copies", "sim.py")); s = open(p75, encoding="utf8").read()
G = {"__file__": p75, "__name__": "sim075"}; exec(compile(s[: s.index('PP = G74["PP"]')], p75, "exec"), G)
r75 = json.load(open(os.path.join(RS, "075-reach-to-tc-copies", "results.json"), encoding="utf8"))
r85 = json.load(open(os.path.join(RS, "085-post-collapse-urban-elasticity", "results.json"), encoding="utf8"))
R = {"study": "089-short-scale-urban-beta", "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
RT, M1, N, GMAX, run = G["RT"], G["M1"], G["N"], G["GMAX"], G["run"]
G74 = G["G74"]; PP = G74["PP"]; MESH = list(G74["mesh"]); SITES = dict(PP); PLP = list(PP) + ["mesh35"]; SITES["mesh35"] = MESH[:35]
wpts = G74["wpts"]; U_geo = G["U_geo"]; GEO3 = G["GEO3"]; NPT = G["NPT"]; UPT = G["UPT"]; PT0 = G["PT"].copy()
h = lambda nm: RT[nm][M1]["heavy"]
def PT_e(e):
    PT = np.zeros((N, GMAX), int)
    for j, g in enumerate(GEO3):
        cdf = np.cumsum(wpts(g, e)); cdf /= cdf[-1]; m = U_geo == j
        PT[m] = np.minimum(np.searchsorted(cdf, UPT[m], side="right"), NPT - 1)
    return PT
def run_e(nm, S, e): G["PT"] = PT_e(e); run(nm, S); G["PT"] = PT0
# 検査：075（e=1）と 085（e=0.6615）
assert (PT_e(1.0) == PT0).all()
for n in (35, 200): run_e(f"mesh{n}_e0.6615", MESH[:n], 0.6615)
d85 = max(abs(h(f"mesh{n}_e0.6615") - r85["heavy_TC"][f"mesh{n}_e0.6615"]) for n in (35, 200))
R["check_085"] = d85; assert d85 < 1e-12, d85
BJ = round(EPI["JP_all"]["beta"], 4); BD = round(EPI["DE_1939_1950_all"]["beta"], 4)
BJlo, BJhi = [round(x, 4) for x in EPI["JP_all"]["ci95"]]
ES = sorted(set([BJ, BD, BJlo, BJhi, round(EPI["JP_excl_Kanto_Tokai"]["beta"], 4)]))
R["e_grid"] = ES; R["beta_used"] = {"JP_1940_1947": BJ, "DE_1939_1950": BD, "JP_ci95": [BJlo, BJhi]}
for e in ES:
    for n in (35, 200): run_e(f"mesh{n}_e{e}", MESH[:n], e)
for p in PP: run_e(f"{p}_e{BJ}", SITES[p], BJ)
R["heavy_TC"] = {k: h(k) for k in RT if k != "chk070"}
R["e_scan"] = {str(e): {"mesh35": h(f"mesh35_e{e}"), "mesh200": h(f"mesh200_e{e}"), "gain_35_200": h(f"mesh200_e{e}") - h(f"mesh35_e{e}")} for e in ES}
R["ref"] = {"075_e1_mesh35": r75["ratio_to_V0"]["mesh35"][M1]["heavy"], "085_e0.6615_mesh35": r85["heavy_TC"]["mesh35_e0.6615"]}
def ktau(x, y):
    s_ = 0; c = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)): s_ += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += 1
    return float(s_ / c)
y85 = [r85["rank"]["TC_e0.6615"][p] for p in PLP]; yj = [h(f"{p}_e{BJ}") if p != "mesh35" else h(f"mesh35_e{BJ}") for p in PLP]
y75 = [r75["ratio_to_V0"][p][M1]["heavy"] for p in PLP]
R["rank"] = {f"TC_e{BJ}": dict(zip(PLP, yj)), "tau_vs_085": ktau(y85, yj), "tau_vs_075": ktau(y75, yj), "best": PLP[int(np.argmax(yj))], "best_085": PLP[int(np.argmax(y85))]}
H = {}
H["H101a"] = {"class": "確認的（局面を 1940→1947 に置き換え）", "value": EPI["JP_all"]["ci95"][1], "rule": "JP beta 95% upper < 1", "supported": EPI["JP_all"]["ci95"][1] < 1}
H["H101b"] = {"class": "確認的（局面を 1940→1947 に置き換え）", "value": EPI["JP_all"]["beta"], "rule": "JP beta < 0.66", "supported": EPI["JP_all"]["beta"] < 0.66}
H["H101c"] = {"class": "確認的", "value": None, "rule": "beta(1940->1950) > beta(1940->1945)", "supported": None, "status": "検定できず（1945・1950 の市の表を開けなかった）"}
v = h(f"mesh35_e{BJ}") - r85["heavy_TC"]["mesh35_e0.6615"]
H["H101d"] = {"class": "予測", "value": v, "best": R["rank"]["best"], "rule": "|TC(mesh35,e=bJ) - 0.672| <= 0.010 and best plan = mesh35", "supported": abs(v) <= 0.010 and R["rank"]["best"] == "mesh35"}
R["hypotheses"] = H
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
import csv
JR = json.load(open(os.path.join(HERE, "episodes.json"), encoding="utf8"))
x = np.log10([r["P0"] for r in JR["japan_rows"]]); y = np.log10([r["P1"] / r["P0"] for r in JR["japan_rows"]])
axs[0].scatter(x, y, s=12, label="Japan 1940→1947 (cities)"); xd = np.log10([d[1] for d in JR["germany_rows"]]); yd = np.log10([d[2] / d[1] for d in JR["germany_rows"]])
axs[0].scatter(xd, yd, s=12, marker="x", label="Germany 1939→1950 (Großstädte)"); axs[0].axhline(0, color="k", lw=.5)
axs[0].set_xlabel("log10 P(before)"); axs[0].set_ylabel("log10 P(after)/P(before)"); axs[0].legend(fontsize=8)
ee = sorted(set(ES + [0.6615, 1.0])); tc = {0.6615: r85["heavy_TC"]["mesh35_e0.6615"], 1.0: r75["ratio_to_V0"]["mesh35"][M1]["heavy"]}
axs[1].plot(ee, [h(f"mesh35_e{e}") if f"mesh35_e{e}" in RT else tc[e] for e in ee], "o-", label="mesh 35")
for e, lab, c in ((0.6615, "085 ancient β", "tab:red"), (BJ, "Japan β", "tab:green"), (BD, "Germany β", "tab:purple")): axs[1].axvline(e, ls="--", color=c, lw=.8, label=lab)
axs[1].set_xlabel("e (survivor groups ~ pop^e)"); axs[1].set_ylabel("heavy-collapse TC/V0 (not survival probability)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_short_beta.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_085", "beta_used", "e_scan", "ref", "rank", "hypotheses")}, ensure_ascii=False, indent=1, default=float)); print("runtime", R["runtime_s"])
