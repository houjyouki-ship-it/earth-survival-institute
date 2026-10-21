"""研究085：崩壊の後に人は大きな町に残るか（open_problems 199）。
第1部 episodes.py（Reba et al. 2016 の Chandler / Modelski の表）で崩壊の前後の都市の人口の弾力性 β を出し、
第2部（本ファイル）で 075 の群の点の重みを国の中で pop^e（e=β）にして重い崩壊の TC/V0 を出す。
計算のみ（ネット接続なし）。事前登録：PREREGISTRATION.md。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
075 の sim.py を「PP = G74["PP"]」の前まで exec し（070・074 の差0の検査を含む）、075 の主を差0で再現してから e を変える。
075 の群の点の一様乱数 UPT はそのまま使う（共通の乱数）。新しい乱数は seed 20261085（α の寄せの版のみ）。"""
import json, os, time, subprocess, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
subprocess.run([sys.executable, os.path.join(HERE, "episodes.py")], check=True)
EPI = json.load(open(os.path.join(HERE, "episodes.json"), encoding="utf8"))
p75 = os.path.abspath(os.path.join(RS, "075-reach-to-tc-copies", "sim.py")); s = open(p75, encoding="utf8").read()
G = {"__file__": p75, "__name__": "sim075"}; exec(compile(s[: s.index('PP = G74["PP"]')], p75, "exec"), G)
r75 = json.load(open(os.path.join(RS, "075-reach-to-tc-copies", "results.json"), encoding="utf8"))
r76 = json.load(open(os.path.join(RS, "076-copy-attraction-marginal", "results.json"), encoding="utf8"))
R = {"study": "085-post-collapse-urban-elasticity", "seed": 20261085, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
RT, DV, M1, N, GMAX, run = G["RT"], G["DV"], G["M1"], G["N"], G["GMAX"], G["run"]
G74 = G["G74"]; PP = G74["PP"]; MESH = list(G74["mesh"]); SITES = dict(PP); PLP = list(PP) + ["mesh35"]; SITES["mesh35"] = MESH[:35]
hav, plat, plon = G["hav"], G["plat"], G["plon"]; wpts = G74["wpts"]; U_geo = G["U_geo"]; GEO3 = G["GEO3"]; NPT = G["NPT"]; UPT = G["UPT"]; PT0 = G["PT"].copy()
h = lambda nm: RT[nm][M1]["heavy"]
# ---------- 検査 ----------
chk = {}
for nm, S in (("mesh35", MESH[:35]), ("mesh200", MESH[:200])):
    run("chk_" + nm, S); chk[nm] = max(abs(RT["chk_" + nm][k][p] - r75["ratio_to_V0"][nm][k][p]) for k in RT["chk_" + nm] for p in RT["chk_" + nm][k])
R["check_075"] = {"max_abs_diff": chk, "reproduced": all(v < 1e-12 for v in chk.values())}; assert R["check_075"]["reproduced"], chk
def PT_e(e):
    PT = np.zeros((N, GMAX), int)
    for j, g in enumerate(GEO3):
        cdf = np.cumsum(wpts(g, e)); cdf /= cdf[-1]; m = U_geo == j
        PT[m] = np.minimum(np.searchsorted(cdf, UPT[m], side="right"), NPT - 1)
    return PT
assert (PT_e(1.0) == PT0).all(); R["check_PT_e1"] = True
# ---------- β の値 ----------
S_ = EPI["summary_median"]; ep = {r["episode"]: r for r in EPI["episodes"]}
def epmean(keys): v = [ep[k]["beta_half"] for k in keys if "beta_half" in ep[k]]; return float(np.mean(v))
per_episode = {"W.Roman": epmean(["W.Roman (Ch 361->622)", "W.Roman (Ch 361->800)"]), "BronzeAge": epmean(["Bronze Age E.Med (Mo BC1300->1000)"]),
               "Akkad": epmean(["Akkad/Ur III Mesop. (Mo BC2300->2000)"]), "Mongol": epmean(["Mongol W.Asia (Ch 1200->1300)", "Mongol W.Asia (Ch 1200->1400)"])}
B_MAIN = S_["beta_half"]["collapse"][0]; B_EP = float(np.median(list(per_episode.values()))); B_CTRL = S_["beta_half"]["control"][0]
R["beta"] = {"main_median_versions": B_MAIN, "median_of_episodes": B_EP, "per_episode": per_episode, "control_median": B_CTRL,
             "listed_only_collapse": S_["beta_listed"]["collapse"][0], "min_episode": min(per_episode.values()), "max_episode": max(per_episode.values())}
ES = sorted(set([0.25, 0.5, round(B_MAIN, 4), round(B_EP, 4), round(min(per_episode.values()), 4), round(max(per_episode.values()), 4), 1.0, round(B_CTRL, 4), 1.25]))
R["e_grid"] = ES
def run_e(nm, S, e):
    G["PT"] = PT_e(e); run(nm, S); G["PT"] = PT0
for e in ES:
    for n in (35, 200): run_e(f"mesh{n}_e{e}", MESH[:n], e)
run_e(f"mesh100_e{round(B_MAIN,4)}", MESH[:100], round(B_MAIN, 4))
for p in PP: run_e(f"{p}_e{round(B_MAIN,4)}", SITES[p], round(B_MAIN, 4))
d1 = max(abs(h(f"mesh{n}_e1.0") - r75["ratio_to_V0"][f"mesh{n}"][M1]["heavy"]) for n in (35, 200)); R["check_e1"] = d1; assert d1 < 1e-12
# 版：e=β と 076 の α=0.5 の寄せを重ねる（寄せの重みは pop^β·exp(-d/L)）
L0 = float(np.median(G["KMOV"])); rg = np.random.default_rng(20261085); UA = rg.uniform(size=(N, GMAX)); UB = rg.uniform(size=(N, GMAX))
def run_e_att(nm, S, e, alpha):
    dmin = hav(plat[:, None], plon[:, None], plat[np.array(S)][None, :], plon[np.array(S)][None, :]).min(1)
    PT = PT_e(e); m_att = UA < alpha
    for j, g in enumerate(GEO3):
        w = wpts(g, e) * np.exp(-dmin / L0); cdf = np.cumsum(w); cdf /= cdf[-1]; m = (U_geo[:, None] == j) & m_att
        PT[m] = np.minimum(np.searchsorted(cdf, UB[m], side="right"), NPT - 1)
    G["PT"] = PT; run(nm, S); G["PT"] = PT0
for n in (35, 200): run_e_att(f"mesh{n}_e{round(B_MAIN,4)}_a0.5", MESH[:n], round(B_MAIN, 4), 0.5)
# ---------- まとめ ----------
R["heavy_TC"] = {k: h(k) for k in RT if k != "chk070"}; R["total_TC"] = {k: RT[k][M1]["total"] for k in RT if k != "chk070"}
R["reach_stats"] = G["R"]["reach_stats"]
R["e_scan"] = {str(e): {"mesh35": h(f"mesh35_e{e}"), "mesh200": h(f"mesh200_e{e}"), "gain_35_200": h(f"mesh200_e{e}") - h(f"mesh35_e{e}"),
                        "share_groups_n0_mesh35": R["reach_stats"][f"mesh35_e{e}"]["share_groups_n0"]} for e in ES}
# 076 の α 走査に対する等価な α（線形の内挿・外挿、推測）
al = np.array([0.0, 0.1, 0.25, 0.5, 1.0]); tc = np.array([r76["alpha_scan"][str(a)]["35"] for a in al]); slope0 = (tc[1] - tc[0]) / 0.1
bm = round(B_MAIN, 4); tcm = h(f"mesh35_e{bm}")
R["equiv_alpha"] = {"slope_dTC_dalpha_0_0.1": float(slope0), **{f"e{e}": float((h(f"mesh35_e{e}") - tc[0]) / slope0) for e in ES}}
def ktau(x, y):
    s_ = 0; c = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)): s_ += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += 1
    return float(s_ / c)
y0 = [r75["ratio_to_V0"][p][M1]["heavy"] for p in PLP]; yb = [h(f"{p}_e{bm}") if p != "mesh35" else h(f"mesh35_e{bm}") for p in PLP]
R["rank"] = {"TC_e1": dict(zip(PLP, y0)), f"TC_e{bm}": dict(zip(PLP, yb)), "kendall_tau": ktau(y0, yb), "best_e1": PLP[int(np.argmax(y0))], "best_eb": PLP[int(np.argmax(yb))],
             "spread_e1": max(y0) - min(y0), "spread_eb": max(yb) - min(yb)}
# ---------- 仮説 ----------
H = {}
H["H97a"] = {"class": "確認的", "value": B_MAIN, "rule": "median beta (collapse, half-threshold version) < 1", "supported": B_MAIN < 1}
H["H97b"] = {"class": "確認的", "value": [B_MAIN, B_CTRL], "rule": "median beta collapse < median beta control", "supported": B_MAIN < B_CTRL}
v = tcm - r75["ratio_to_V0"]["mesh35"][M1]["heavy"]; H["H97c"] = {"class": "予測", "value": v, "rule": "heavy TC mesh35 e=beta - 0.680 <= -0.005", "supported": v <= -0.005}
v = h(f"mesh200_e{bm}") - tcm; H["H97d"] = {"class": "予測", "value": v, "rule": "gain 35->200 at e=beta >= 075 gain", "supported": v >= r75["ratio_to_V0"]["mesh200"][M1]["heavy"] - r75["ratio_to_V0"]["mesh35"][M1]["heavy"]}
v = R["rank"]["kendall_tau"]; H["H97e"] = {"class": "予測", "value": v, "rule": "Kendall tau over 10 plans >= 0.6", "supported": v >= 0.6}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
eps = [r for r in EPI["episodes"] if "beta_half" in r]
for i, r in enumerate(eps):
    c = "tab:red" if r["kind"] == "collapse" else "tab:blue"; lo, hi = r["beta_half_ci"]
    axs[0].errorbar(r["beta_half"], i, xerr=[[r["beta_half"] - lo], [hi - r["beta_half"]]], fmt="o", color=c)
axs[0].set_yticks(range(len(eps))); axs[0].set_yticklabels([f'{r["episode"]} n={r["n_t0"]}' for r in eps], fontsize=7)
axs[0].axvline(1, color="k", ls=":"); axs[0].axvline(B_MAIN, color="tab:red", ls="--", lw=.8); axs[0].axvline(B_CTRL, color="tab:blue", ls="--", lw=.8)
axs[0].set_xlabel("beta: log P(t1) on log P(t0) (vanished = half listing threshold), 95% bootstrap"); axs[0].set_xlim(-2, 3)
for n in (35, 200): axs[1].plot(ES, [h(f"mesh{n}_e{e}") for e in ES], "o-", label=f"mesh {n}")
axs[1].axvline(B_MAIN, color="tab:red", ls="--", lw=.8, label="collapse median beta"); axs[1].axvline(1, color="k", ls=":", lw=.8, label="075 (e=1)")
axs[1].set_xlabel("e (survivor groups placed within country ~ pop^e)"); axs[1].set_ylabel("heavy-collapse TC / V0 (not survival probability)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_beta_tc.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_075", "beta", "e_scan", "equiv_alpha", "rank", "hypotheses")}, ensure_ascii=False, indent=1, default=float)); print("runtime", R["runtime_s"])
