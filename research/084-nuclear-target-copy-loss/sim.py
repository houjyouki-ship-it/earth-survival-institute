"""研究084：核の標的の都市にある写しが丸ごと失われると、重い崩壊の TC/V0 と置き方の順位はどう変わるか（open_problems 168）。
計算のみ（ネット接続なし）。事前登録：PREREGISTRATION.md。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
075 の sim.py を「PP = G74["PP"]」の前まで exec し（070・074 の差0の検査を含む）、075 の主を差0で再現してから、
xia150・xia47 の地理の標本では標的の点から R_d 以内の写しを 0 部と数える。標的は sources/targets.json（Toon 2008、Xia 2022、Toon 2019）。
軽い崩壊の部分（S_ind）は変えていない（限界）。新しい乱数は seed 20261084（α=0.5 の寄せのみ）。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p75 = os.path.abspath(os.path.join(RS, "075-reach-to-tc-copies", "sim.py")); s = open(p75, encoding="utf8").read()
G = {"__file__": p75, "__name__": "sim075"}; exec(compile(s[: s.index('PP = G74["PP"]')], p75, "exec"), G)
r75 = json.load(open(os.path.join(RS, "075-reach-to-tc-copies", "results.json"), encoding="utf8"))
TGT = json.load(open(os.path.join(HERE, "sources", "targets.json"), encoding="utf8"))
R = {"study": "084-nuclear-target-copy-loss", "seed": 20261084, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
RT, DV, M1, N, GMAX = G["RT"], G["DV"], G["M1"], G["N"], G["GMAX"]; ratio_all, ver, rho75, counts, REACH = G["ratio_all"], G["ver"], G["rho75"], G["counts"], G["REACH"]
G74 = G["G74"]; PP = G74["PP"]; MESH = list(G74["mesh"]); SITES = dict(PP); PLP = list(PP) + ["mesh35"]; SITES["mesh35"] = MESH[:35]
hav, plat, plon, ppop, pc, A2 = G["hav"], G["plat"], G["plon"], G74["ppop"], G74["pc"], G74["A2"]
U_geo = G["U_geo"]; GEO3 = G["GEO3"]; Wp = G["Wp"]; KMOV = G["KMOV"]; NPT = G["NPT"]; PT0 = G["PT"].copy()
assert GEO3[:2] == ("xia150", "xia47")
h = lambda nm: RT[nm][M1]["heavy"]
# ---------- 標的 ----------
cA2 = np.array([A2[i] for i in pc])
def targets(weapons):
    out = []
    for c, n in weapons.items():
        idx = np.where(cA2 == c)[0]; assert len(idx), c
        out += list(idx[np.argsort(-ppop[idx], kind="stable")][:n])
    return np.array(out)
TG = {"xia150": targets(TGT["xia150"]["weapons"]), "xia47": targets(TGT["xia47"]["weapons"]),
      "xia150_noINPK": targets({k: v for k, v in TGT["xia150"]["weapons"].items() if k not in ("IN", "PK")})}
R["targets"] = {k: {"n_points": int(len(v)), "by_country": {c: int((cA2[v] == c).sum()) for c in sorted(set(cA2[v]))}} for k, v in TG.items()}
def destroyed(S, tg, Rd):
    S = np.array(S); d = hav(plat[S][:, None], plon[S][:, None], plat[tg][None, :], plon[tg][None, :]).min(1); return d <= Rd
def surv(S, tg, Rd): return [x for x, k in zip(S, destroyed(S, tg, Rd)) if not k]
# ---------- 実行 ----------
def run_nuc(nm, S, Rd=10.0, t150="xia150", t47="xia47", alpha=0.0, L=None, rng=None):
    S150, S47 = surv(S, TG[t150], Rd), surv(S, TG[t47], Rd)
    if alpha > 0:
        PTn = PT0.copy(); m_att = UA < alpha
        for j, (gname, Sg) in enumerate(zip(GEO3, (S150, S47, list(S)))):
            dmin = hav(plat[:, None], plon[:, None], plat[np.array(Sg)][None, :], plon[np.array(Sg)][None, :]).min(1) if len(Sg) else np.full(NPT, np.inf)
            w = Wp[(gname, 1.0)] * np.exp(-dmin / L); cdf = np.cumsum(w); cdf /= cdf[-1]; m = (U_geo[:, None] == j) & m_att
            PTn[m] = np.minimum(np.searchsorted(cdf, UB[m], side="right"), NPT - 1)
        G["PT"] = PTn
    Ds = [counts(Sg, REACH["main"]) for Sg in (S150, S47, list(S))]; G["PT"] = PT0
    D = np.where((U_geo == 0)[:, None], Ds[0], np.where((U_geo == 1)[:, None], Ds[1], Ds[2]))
    RT[nm], DV[nm] = ratio_all(ver(rho=rho75(D, 1, len(S), len(S))))
    R.setdefault("loss", {})[nm] = {"n": len(S), "lost150": len(S) - len(S150), "lost47": len(S) - len(S47), "share150": 1 - len(S150) / len(S), "share47": 1 - len(S47) / len(S)}
    print(nm, round(h(nm), 4), R["loss"][nm], round(time.time() - T0), "s", flush=True)
rg = np.random.default_rng(20261084); UA = rg.uniform(size=(N, GMAX)); UB = rg.uniform(size=(N, GMAX)); L0 = float(np.median(KMOV))
# 検査：R_d=0 かつ標的なしで 075 を差0（Rd=-1 で何も壊さない）
for nm in ("mesh35", "per_subregion@cty"):
    run_nuc("chk_" + nm, SITES[nm], Rd=-1.0)
    d = max(abs(RT["chk_" + nm][k][p] - r75["ratio_to_V0"][nm][k][p]) for k in RT["chk_" + nm] for p in RT["chk_" + nm][k]); assert d < 1e-12, (nm, d)
R["check_075_reproduced"] = True; print("check ok", flush=True)
for p in PLP: run_nuc(p, SITES[p])
for n in (100, 200): run_nuc(f"mesh{n}", MESH[:n])
for Rd in (5.0, 25.0):
    for p in ("mesh35", "pop@pt", "per_subregion@cty"): run_nuc(f"{p}_Rd{int(Rd)}", SITES[p], Rd=Rd)
    run_nuc(f"mesh200_Rd{int(Rd)}", MESH[:200], Rd=Rd)
run_nuc("mesh35_noINPK", MESH[:35], t150="xia150_noINPK"); run_nuc("mesh200_noINPK", MESH[:200], t150="xia150_noINPK")
# 標的を避けた mesh（標的の 10 km 以内を飛ばして貪欲の順に 35・200）：記述の版
allT = np.r_[TG["xia150"], TG["xia47"]]; ok = ~destroyed(MESH, allT, 10.0); MAV = [x for x, k in zip(MESH, ok) if k]
R["mesh_avoid_available"] = len(MAV)
for n in (35, 200): run_nuc(f"meshavoid{n}", MAV[:n])
for n in (35, 200):
    run_nuc(f"mesh{n}_a0.5", MESH[:n], alpha=0.5, L=L0); run_nuc(f"meshavoid{n}_a0.5", MAV[:n], alpha=0.5, L=L0)
# ---------- まとめ ----------
def ktau(x, y):
    s_ = 0; c = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)): s_ += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += 1
    return float(s_ / c)
y75 = [r75["ratio_to_V0"][p][M1]["heavy"] for p in PLP]; y84 = [h(p) for p in PLP]
R["heavy_TC"] = {k: h(k) for k in RT if k != "chk070"}; R["heavy_TC_075"] = dict(zip(PLP, y75)) | {f"mesh{n}": r75["ratio_to_V0"][f"mesh{n}"][M1]["heavy"] for n in (100, 200)}
R["rank"] = {"kendall_tau": ktau(y75, y84), "best_075": PLP[int(np.argmax(y75))], "best_084": PLP[int(np.argmax(y84))], "spread_084": max(y84) - min(y84)}
r76 = json.load(open(os.path.join(RS, "076-copy-attraction-marginal", "results.json"), encoding="utf8"))
R["ref_076_a0.5"] = {n: r76["alpha_scan"]["0.5"][str(n)] for n in (35, 200)}
g75 = R["heavy_TC_075"]["mesh200"] - R["heavy_TC_075"]["mesh35"]; g84 = h("mesh200") - h("mesh35"); R["gain_35_200"] = {"075": g75, "084": g84}
H = {}
v = R["loss"]["mesh35"]["share150"]; H["H96a"] = {"class": "確認的", "value": v, "rule": "mesh35 の写しのうち xia150 の標的 10 km 以内 >= 0.3", "supported": v >= 0.3}
v = R["loss"]["mesh35"]["share47"]; H["H96b"] = {"class": "予測", "value": v, "rule": "xia47 の標的 10 km 以内の mesh35 <= 0.15", "supported": v <= 0.15}
v = R["heavy_TC_075"]["mesh35"] - h("mesh35"); H["H96c"] = {"class": "予測", "value": v, "rule": "075 mesh35 0.680 から 0.01 以上下がる", "supported": v >= 0.01}
v = [R["rank"]["kendall_tau"], R["loss"]["pop@pt"]["share150"], R["loss"]["mesh35"]["share150"]]
H["H96d"] = {"class": "予測", "value": v, "rule": "tau >= 0.6 かつ pop@pt の失う割合 > mesh35", "supported": v[0] >= 0.6 and v[1] > v[2]}
H["H96e"] = {"class": "予測", "value": [g84, g75], "rule": "35->200 の上げ幅 > 075 の上げ幅", "supported": g84 > g75}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.arange(len(PLP)); axs[0].bar(x - .2, y75, .4, label="075 (no target loss)"); axs[0].bar(x + .2, y84, .4, label="084 (targets, R_d=10 km)")
axs[0].set_ylim(min(y84 + y75) - .02, max(y84 + y75) + .01); axs[0].set_xticks(x); axs[0].set_xticklabels(PLP, rotation=30, fontsize=7); axs[0].set_ylabel("heavy TC/V0 (not survival probability)"); axs[0].legend(fontsize=7)
for i, p in enumerate(PLP): axs[0].annotate(f"{R['loss'][p]['share150']:.0%}", (i + .2, y84[i]), fontsize=6, ha="center", va="bottom")
axs[1].scatter(plat * 0 + plon, plat, s=.5, c="0.85"); axs[1].scatter(plon[TG["xia150"]], plat[TG["xia150"]], s=2, c="C3", label="xia150 targets")
axs[1].scatter(plon[TG["xia47"]], plat[TG["xia47"]], s=2, c="C1", label="xia47 targets"); m35 = np.array(MESH[:35])
axs[1].scatter(plon[m35], plat[m35], marker="^", facecolors="none", edgecolors="C0", s=30, label="mesh35"); axs[1].legend(fontsize=7, markerscale=3); axs[1].set_title("Target points (NE, pop order) and mesh35")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_target_loss.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("targets", "rank", "gain_35_200", "hypotheses", "mesh_avoid_available", "ref_076_a0.5")}, ensure_ascii=False, indent=1, default=float))
print(json.dumps(R["heavy_TC"], default=float)); print("runtime", R["runtime_s"])
