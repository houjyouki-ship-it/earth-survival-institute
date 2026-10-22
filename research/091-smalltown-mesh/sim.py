"""研究091：写しを小さな町にも置ける網目（open_problems 223）。計算のみ（ネット接続なし、新しい資料・乱数なし）。
事前登録：PREREGISTRATION.md。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
090 の sim.py を版の走査の前まで exec（087・075・084 の差0の assert を含む）し、090 の mesh35_M などを差0で、074 の網目の最初の200か所を完全に再現（assert）してから、
GeoNames の点を写しの候補にした貪欲な網目を作り、090 の各版の群の点で評価する。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p90 = os.path.abspath(os.path.join(RS, "090-rural-survivor-points", "sim.py")); s = open(p90, encoding="utf8").read()
G = {"__file__": p90, "__name__": "sim090"}; exec(compile(s[: s.index('R["rural_share_of_weight"] = {}')], p90, "exec"), G)
r90 = json.load(open(os.path.join(RS, "090-rural-survivor-points", "results.json"), encoding="utf8"))
r74 = json.load(open(os.path.join(RS, "074-within-country-reach", "results.json"), encoding="utf8"))
R = {"study": "091-smalltown-mesh", "seed": None, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
RT, DV, M1, N, GMAX = G["RT"], G["DV"], G["M1"], G["N"], G["GMAX"]; ratio_all, ver, rho75, REACH, valid = G["ratio_all"], G["ver"], G["rho75"], G["REACH"], G["valid"]
ps, mass, PT_s, VERS, GEO, GEO3, NK, hav = G["ps"], G["mass"], G["PT_s"], G["VERS"], G["GEO"], G["GEO3"], G["NK"], G["G"]["G74"]["hav"]
G74 = G["G"]["G74"]; plat, plon, MESH74 = G74["plat"], G74["plon"], list(G74["mesh"]); KMOV = G74["KMOV"]
h = lambda nm: RT[nm][M1]["heavy"]
def counts_c(cla, clo, PT):  # 088 と同じ式（写しの座標を配列で受ける）
    Ds = np.sort(hav(ps.lat[:, None], ps.lon[:, None], cla[None, :], clo[None, :]), axis=1).astype(np.float32)
    out = np.full((N, GMAX), -1, int); ii, jj = np.nonzero(valid); pp = PT[ii, jj]; rr = REACH["main"][ii, jj].astype(np.float32); n = np.empty(len(ii), int)
    for c in range(0, len(ii), 40000): n[c:c + 40000] = (Ds[pp[c:c + 40000]] <= rr[c:c + 40000, None]).sum(1)
    out[ii, jj] = n; return out
def run_c(nm, cla, clo, PT):
    D = counts_c(np.asarray(cla), np.asarray(clo), PT); k = len(cla); RT[nm], DV[nm] = ratio_all(ver(rho=rho75(D, 1, k, k)))
    R.setdefault("reach", {})[nm] = {"share_groups_n0": float((D[valid] == 0).mean()), "mean_n": float(D[valid].mean())}
    print(nm, round(h(nm), 4), round(R["reach"][nm]["share_groups_n0"], 4), round(time.time() - T0), "s", flush=True)
PTV = {v: PT_s(ps, mass(**VERS[v])) for v in ("B_GN", "R", "R_all", "M", "M2")}
# ---------- 検査1：090 を差0 ----------
chk = {}
for v, n in (("M", 35), ("M", 200), ("R", 35)):
    S = np.array(MESH74[:n]); run_c(f"chk90_mesh{n}_{v}", plat[S], plon[S], PTV[v]); chk[f"mesh{n}_{v}"] = abs(h(f"chk90_mesh{n}_{v}") - r90["heavy_TC"][f"mesh{n}_{v}"])
R["check_090"] = chk; assert all(x == 0 for x in chk.values()), chk
# ---------- 貪欲な網目（074 の式を関数に） ----------
ks = np.sort(KMOV); RMAX = float(ks.max())
def cov(d): return 1 - np.searchsorted(ks, d, side="left") / len(ks)
def greedy(slat, slon, sw, clat, clon, nmax):
    NC = len(clat); rows, cols, vals = [], [], []
    for c0 in range(0, NC, 500):
        Dc = hav(slat[:, None], slon[:, None], clat[None, c0:c0 + 500], clon[None, c0:c0 + 500]); ii, jj = np.nonzero(Dc <= RMAX)
        rows.append(jj + c0); cols.append(ii); vals.append(cov(Dc[ii, jj]))
    cand_of, surv_of, cv = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
    o = np.argsort(cand_of, kind="stable"); cand_of, surv_of, cv = cand_of[o], surv_of[o], cv[o]
    starts = np.searchsorted(cand_of, np.arange(NC)); assert (np.diff(np.r_[starts, len(cv)]) > 0).all()
    wsv = sw[surv_of]; cur = np.zeros(len(slat)); mesh = []
    for t in range(nmax):
        gain = np.add.reduceat(np.maximum(cv - cur[surv_of], 0) * wsv, starts)
        b = int(np.argmax(gain)); mesh.append(b); sl = slice(starts[b], starts[b + 1] if b + 1 < NC else len(cv)); cur[surv_of[sl]] = np.maximum(cur[surv_of[sl]], cv[sl])
    return mesh, float(sw @ cur), len(cv)
# 検査2：074 の入力で 074 の網目の最初の200か所
m74, _, nnz = greedy(plat, plon, G74["wbar"], plat, plon, 200)
R["check_074_mesh200_identical"] = m74 == [int(x) for x in MESH74[:200]]; R["check_074_nnz"] = [nnz, r74["mesh_nnz"]]
assert R["check_074_mesh200_identical"] and nnz == r74["mesh_nnz"]; print("checks ok", round(time.time() - T0), "s", flush=True)
# ---------- ます目と候補 ----------
def wbar_of(v):
    m = mass(**VERS[v]); tot = np.bincount(ps.c, weights=m, minlength=NK)
    return np.mean([GEO[g][ps.c] * m / tot[ps.c] for g in GEO3], axis=0)
CELL = 0.5
cid = (np.floor((ps.lat + 90) / CELL) * 10000 + np.floor((ps.lon + 180) / CELL)).astype(np.int64); uc, inv = np.unique(cid, return_inverse=True); NCELL = len(uc)
def cells(w):
    sw = np.bincount(inv, weights=w, minlength=NCELL); sw_ = np.where(sw > 0, sw, 1)
    # 重心は 3 次元の単位ベクトルの重み付き平均（経度 180 度をまたいでも安全）
    la, lo = np.radians(ps.lat), np.radians(ps.lon); xyz = [np.bincount(inv, weights=w * f, minlength=NCELL) / sw_ for f in (np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la))]
    clat = np.degrees(np.arctan2(xyz[2], np.hypot(xyz[0], xyz[1]))); clon = np.degrees(np.arctan2(xyz[1], xyz[0])); return sw, clat, clon
o = np.lexsort((-ps.pop, inv)); first = o[np.r_[True, inv[o][1:] != inv[o][:-1]]]  # ます目ごとの人口最大の点
CAND = first; R["grid"] = {"cell_deg": CELL, "n_cells": int(NCELL), "n_candidates_GN": int(len(CAND)), "n_candidates_NE": int(len(plat))}
print(R["grid"], flush=True)
BUILD = {"GN_M": ("M", "GN"), "GN_B": ("B_GN", "GN"), "GN_Rall": ("R_all", "GN"), "NE_M": ("M", "NE")}
SITES = {"NE074": (plat[np.array(MESH74[:200])], plon[np.array(MESH74[:200])], G74["ppop"][np.array(MESH74[:200])])}
R["build"] = {}
for nm, (v, cs) in BUILD.items():
    sw, sla, slo = cells(wbar_of(v)); k = sw > 0
    if cs == "GN": cla, clo, cpo = ps.lat[CAND], ps.lon[CAND], ps.pop[CAND]
    else: cla, clo, cpo = plat, plon, G74["ppop"]
    mesh, obj, nnz = greedy(sla[k], slo[k], sw[k], cla, clo, 200); mesh = np.array(mesh)
    SITES[nm] = (cla[mesh], clo[mesh], cpo[mesh]); R["build"][nm] = {"weights": v, "candidates": cs, "nnz": nnz, "A_q0_build_200": obj}
    print("built", nm, nnz, round(time.time() - T0), "s", flush=True)
# 参考：074 の網目を同じます目の作りの重み（M）で測った被覆
for nm in SITES:
    sw, sla, slo = cells(wbar_of("M")); k = sw > 0; cl, co, _ = SITES[nm]; out = {}
    for n in (35, 200):
        D = hav(sla[k][:, None], slo[k][:, None], cl[None, :n], co[None, :n]); out[n] = float(sw[k] @ cov(D.min(1)) if True else 0)
    R.setdefault("A_q0_cellsM", {})[nm] = out
R["site_stats"] = {nm: {str(n): {"pop_median": float(np.median(SITES[nm][2][:n])), "pop_min": float(SITES[nm][2][:n].min()), "share_lt_50k": float((SITES[nm][2][:n] < 5e4).mean()),
                                 "share_lt_10k": float((SITES[nm][2][:n] < 1e4).mean())} for n in (35, 200)} for nm in SITES}
# ---------- 評価 ----------
EV = ("B_GN", "R", "R_all", "M", "M2")
for nm in SITES:
    for n in (35, 200):
        for v in EV:
            if nm == "NE074" and v in ("M",) and False: continue
            run_c(f"{nm}_{n}_{v}", SITES[nm][0][:n], SITES[nm][1][:n], PTV[v])
R["heavy_TC"] = {k: h(k) for k in RT if k != "chk070" and not k.startswith("chk")}
T = {nm: {str(n): {v: h(f"{nm}_{n}_{v}") for v in EV} for n in (35, 200)} for nm in SITES}; R["table"] = T
for v in ("B_GN", "R", "M"): assert abs(T["NE074"]["35"][v] - r90["heavy_TC"][f"mesh35_{v}"]) == 0
d = lambda nm, v, n="35": T[nm][n][v] - T["NE074"][n][v]
H = {}
x = d("GN_M", "M"); H["H103a"] = {"class": "予測", "value": x, "rule": "GN_M35 - NE074_35 (M) >= 0.005", "supported": x >= 0.005}
x = [T["GN_M"]["35"]["B_GN"], T["NE074"]["35"]["B_GN"]]; H["H103b"] = {"class": "予測", "value": x, "rule": "GN_M35 under B_GN >= NE074 under B_GN", "supported": x[0] >= x[1]}
x = R["reach"]["NE074_35_M"]["share_groups_n0"] - R["reach"]["GN_M_35_M"]["share_groups_n0"]; H["H103c"] = {"class": "予測", "value": x, "rule": "drop of share_groups_n0 (M, 35) >= 0.02", "supported": x >= 0.02}
x = [d("NE_M", "M"), d("GN_M", "M")]; H["H103d"] = {"class": "予測", "value": x, "rule": "NE_M gain < 0.5 * GN_M gain (M, 35)", "supported": x[0] < 0.5 * x[1]}
x = T["GN_M"]["200"]["M"] - T["GN_M"]["35"]["M"]; H["H103e"] = {"class": "予測", "value": x, "rule": "GN_M 35->200 gain (M) >= 0.0465", "supported": x >= 0.0465}
x = R["site_stats"]["GN_M"]["35"]["pop_median"]; H["H103f"] = {"class": "記述", "value": x, "rule": "median pop of GN_M 35 sites < 50,000 (not counted)", "supported": x < 5e4}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
R["gain_vs_NE074"] = {nm: {n: {v: T[nm][n][v] - T["NE074"][n][v] for v in EV} for n in ("35", "200")} for nm in SITES if nm != "NE074"}
R["share_groups_n0"] = {k: R["reach"][k]["share_groups_n0"] for k in R["reach"]}
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
axs[0].scatter(ps.lon[::5], ps.lat[::5], s=.2, c="0.8")
for nm, mk, c in (("NE074", "o", "C0"), ("GN_M", "^", "C3"), ("GN_B", "x", "C2")):
    axs[0].scatter(SITES[nm][1][:35], SITES[nm][0][:35], s=28, marker=mk, facecolors="none" if mk != "x" else c, edgecolors=c, label=f"{nm} first 35")
axs[0].legend(fontsize=8); axs[0].set_title("35 sites: NE large places vs GeoNames small-town candidates")
nms = list(SITES); xs = np.arange(len(EV)); wdt = 0.8 / len(nms)
for i, nm in enumerate(nms): axs[1].bar(xs + i * wdt, [T[nm]["35"][v] for v in EV], wdt, label=nm)
axs[1].set_xticks(xs + 0.4 - wdt / 2); axs[1].set_xticklabels([f"eval {v}" for v in EV], fontsize=8); axs[1].set_ylim(0.64, 0.71)
axs[1].set_ylabel("heavy-collapse TC/V0, 35 copies (not survival probability)"); axs[1].legend(fontsize=7); axs[1].set_title("build weights vs evaluation weights")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_smalltown_mesh.png"), dpi=115); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_090", "grid", "build", "A_q0_cellsM", "site_stats", "table", "hypotheses")}, ensure_ascii=False, indent=1, default=float)); print("runtime", R["runtime_s"])
