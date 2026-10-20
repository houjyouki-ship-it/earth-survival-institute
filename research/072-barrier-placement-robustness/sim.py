"""研究072：写しの置き方を k（移動能力の倍率）× q（写しの破損率）× 地理 × 距離の定義 で比べ、最悪の場合と最小の後悔で選ぶ。
計算のみ（ネット接続なし）。指標 A は「生存者（地理の重み）のうち届く範囲に生き残った写しが1つ以上ある割合」の厳密な期待値で、
生存確率でも復興確率でもない。乱数は random の置き方の抽出だけ（seed 20261072）。"""
import json, math, os, re, time, heapq, csv
import numpy as np
from matplotlib.path import Path
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); R = {}
S71 = os.path.join(HERE, "..", "071-text-seeking-reach", "sources"); S70 = os.path.join(HERE, "..", "070-group-deposit-overlap", "sources")
# ---------- 資料 ----------
src = open(os.path.join(S70, "make_cases.py"), encoding="utf8").read(); G = {"__file__": os.path.join(S70, "make_cases.py")}
cwd = os.getcwd(); os.chdir(S70); exec(src[: src.index("# 1) Xia")], G); os.chdir(cwd)
region, ISO, ALIAS = G["region"], G["ISO"], G["ALIAS"]
html = open(os.path.join(S71, "google_countries.html"), encoding="utf8").read()
XY = {a: (float(la), float(lo)) for a, la, lo in re.findall(r"<tr>\s*<td>([A-Z]{2})</td>\s*<td>([-\d.]+)</td>\s*<td>([-\d.]+)</td>", html)}
S5 = json.load(open(os.path.join(S71, "xia_tables.json"), encoding="utf8"))["S5_partial_livestock"]
COL = json.load(open(os.path.join(S71, "xia_tables.json"), encoding="utf8"))["columns"]
K, A2, LAT, LON, SUB, POP, SV150, SV47 = [], [], [], [], [], [], [], []
for name, v in sorted(S5.items()):
    n = ALIAS.get(name, name)
    if n is None or n not in ISO: continue
    a2 = ISO[n]["alpha-2"]; r = region(name)
    if a2 not in XY or r is None: continue
    K.append(name); A2.append(a2); LAT.append(XY[a2][0]); LON.append(XY[a2][1]); SUB.append(r)
    pop = v[COL.index("pop2010")]; POP.append(pop)
    SV150.append(max(pop - v[COL.index("150Tg")], 0.0)); SV47.append(max(pop - v[COL.index("47Tg")], 0.0))
# Vanuatu は Xia S5 にないので、refuge の重みだけを持つ国として足す（預け先の候補・被覆の数には入れない。報告の逸脱）
for name in ("Vanuatu",):
    n = ALIAS.get(name, name); a2 = ISO[n]["alpha-2"]
    K.append(name); A2.append(a2); LAT.append(XY[a2][0]); LON.append(XY[a2][1]); SUB.append(region(name)); POP.append(0.0); SV150.append(0.0); SV47.append(0.0)
NK = len(K); POP, SV150, SV47 = map(np.array, (POP, SV150, SV47)); LAT, LON = np.array(LAT), np.array(LON)
REFUGE = ["Australia", "New Zealand", "Iceland", "Solomon Islands", "Vanuatu"]
assert all(x in K for x in REFUGE), [x for x in REFUGE if x not in K]
ref = np.array([1.0 if k in REFUGE else 0.0 for k in K])
GEO = {"xia150": SV150 / SV150.sum(), "xia47": SV47 / SV47.sum(), "pop": POP / POP.sum(), "refuge": ref / ref.sum()}
CASES = np.array([c["dist_km"] for c in json.load(open(os.path.join(S71, "cases.json"), encoding="utf8"))["cases"]])
R["n_countries"] = NK; R["n_cases"] = len(CASES); R["case_median_km"] = float(np.median(CASES))
def hav(la1, lo1, la2, lo2):
    p1, p2 = np.radians(la1), np.radians(la2); dl = np.radians(lo2 - lo1)
    h = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
CAND = [i for i in range(NK) if K[i] != "Vanuatu"]
DGC = hav(LAT[:, None], LON[:, None], LAT[None, :], LON[None, :])
# ---------- 格子（1°）と国境 ----------
NE = json.load(open(os.path.join(HERE, "sources", "ne_110m_admin_0_countries.geojson"), encoding="utf8"))
NLA, NLO = 180, 360; glat = -89.5 + np.arange(NLA); glon = -179.5 + np.arange(NLO)
PLA, PLO = np.meshgrid(glat, glon, indexing="ij"); pts = np.c_[PLO.ravel(), PLA.ravel()]
CID = -np.ones(NLA * NLO, int)
for fi, f in enumerate(NE["features"]):
    g = f["geometry"]; polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
    for poly in polys:
        ring = np.array(poly[0]); bb = (pts[:, 0] >= ring[:, 0].min() - 1) & (pts[:, 0] <= ring[:, 0].max() + 1) & (pts[:, 1] >= ring[:, 1].min() - 1) & (pts[:, 1] <= ring[:, 1].max() + 1)
        idx = np.where(bb)[0]
        if len(idx) == 0: continue
        inside = Path(ring).contains_points(pts[idx])
        for hole in poly[1:]: inside &= ~Path(np.array(hole)).contains_points(pts[idx])
        CID[idx[inside]] = fi
LAND = CID >= 0; R["grid_land_share"] = float(LAND.mean())
def cell(la, lo): return int(min(max(math.floor(la + 90), 0), NLA - 1)) * NLO + int(math.floor(lo + 180)) % NLO
NODE = [cell(a, b) for a, b in zip(LAT, LON)]
R["country_cell_is_land"] = float(np.mean([LAND[c] for c in NODE]))
# 隣接（8近傍、経度は周回）
nbr = []
for i in range(NLA):
    for j in range(NLO):
        lst = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0: continue
                ii = i + di
                if not 0 <= ii < NLA: continue
                jj = (j + dj) % NLO; lst.append(ii * NLO + jj)
        nbr.append(lst)
LEN = [[float(hav(glat[a // NLO], glon[a % NLO], glat[b // NLO], glon[b % NLO])) for b in nbr[a]] for a in range(NLA * NLO)]
CUT = 2.0 * CASES.max() + 1.0  # これより遠いコストは届かない（k ≤ 2）
def dijkstra(src, c_sea, pb):
    fac = np.where(LAND, 1.0, c_sea); dist = {src: 0.0}; pq = [(0.0, src)]; done = set()
    while pq:
        d, a = heapq.heappop(pq)
        if a in done: continue
        done.add(a)
        if d > CUT: break
        fa = fac[a]; ca = CID[a]
        for b, L in zip(nbr[a], LEN[a]):
            w = L * (fa + fac[b]) / 2
            if pb and ca >= 0 and CID[b] >= 0 and ca != CID[b]: w += pb
            nd = d + w
            if nd < dist.get(b, 1e18): dist[b] = nd; heapq.heappush(pq, (nd, b))
    return dist
DIST_VERS = {"gc": None, "b3_0": (3, 0), "b3_300": (3, 300), "b10_0": (10, 0), "b10_300": (10, 300), "chk1_0": (1, 0)}
DM = {"gc": DGC}
for nm, par in DIST_VERS.items():
    if par is None: continue
    M = np.full((NK, NK), np.inf)
    for j in range(NK):
        dd = dijkstra(NODE[j], *par)
        for i in range(NK): M[i, j] = dd.get(NODE[i], np.inf)
    M = np.minimum(M, M.T); np.fill_diagonal(M, 0.0); DM[nm] = M
    print(nm, "done", round(time.time() - T0), "s", flush=True)
fin = np.isfinite(DM["chk1_0"]) & (DGC > 0) & (DGC < CUT)
ratio = DM["chk1_0"][fin] / DGC[fin]
R["check_grid_vs_gc"] = {"median_ratio": float(np.median(ratio)), "p05": float(np.percentile(ratio, 5)), "p95": float(np.percentile(ratio, 95)), "n_pairs": int(fin.sum())}
for nm in ("b3_0", "b3_300", "b10_0", "b10_300"):
    f2 = np.isfinite(DM[nm]) & (DGC > 0) & (DGC < CUT)
    R.setdefault("barrier_vs_gc_median_ratio", {})[nm] = float(np.median(DM[nm][f2] / DGC[f2]))
# ---------- 置き方 ----------
def greedy(R_km, N=None, D=DGC):
    un = set(CAND); S = []
    while un and (N is None or len(S) < N):
        b = max(CAND, key=lambda j: (sum(D[i, j] <= R_km for i in un), -j)); S.append(b)
        un -= {i for i in un if D[i, b] <= R_km}
    return S
def fill(P, N):
    P = list(P)[:N]
    while len(P) < N: P.append(max(CAND, key=lambda j: (min(DGC[j, p] for p in P), -j)))
    return P
P15 = greedy(1500); N = len(P15); R["N"] = N
subs = sorted(set(SUB[i] for i in CAND)); Psub = []
for s in subs:
    m = [i for i in CAND if SUB[i] == s]; Psub.append(min(m, key=lambda j: (sum(POP[i] * DGC[i, j] for i in m), j)))
BIG = json.load(open(os.path.join(S70, "cases.json"), encoding="utf8"))["largest_libraries"]
bigc = []
for l in BIG:
    if l["country"] in K and K.index(l["country"]) not in bigc: bigc.append(K.index(l["country"]))
P3 = greedy(3000)
PLANS = {"cover1500": P15, "cover750": fill(greedy(750, N), N), "cover3000x": [P3[i % len(P3)] for i in range(N)],
         "per_subregion": fill(Psub, N), "pop": list(np.argsort(-POP, kind="stable")[:N]),
         "surv150": list(np.argsort(-SV150, kind="stable")[:N]), "bigplus": fill(bigc, N)}
assert all(len(v) == N for v in PLANS.values())
R["plans"] = {k: [K[i] for i in v] for k, v in PLANS.items()}; R["n_subregions"] = len(subs); R["n_big_countries"] = len(bigc); R["n_cover3000_sites"] = len(P3)
rng = np.random.default_rng(20261072); RAND = [list(rng.choice(CAND, N, replace=False)) for _ in range(200)]
# ---------- 指標 ----------
NARA, XIAN = (34.68444444, 135.805), (34.26111111, 108.94222222)
d_ken = float(hav(*NARA, *XIAN)); H_LOSS = -math.log(2 / 3) / d_ken; R["kentoshi"] = {"d_km": d_ken, "h_per_km": H_LOSS}
def A_of(P, D, w, k, q, hazard=False):
    X = np.sort(D[:, P], axis=1)  # 各国から写しまでの距離（近い順）
    out = np.zeros(NK)
    for d in CASES:
        r = k * d; ok = X <= r
        if not hazard:
            n = ok.sum(1); out += 1 - q ** n
        else:
            j = np.arange(len(P)); pj = (1 - q) * q ** j
            out += (ok * pj[None, :] * np.exp(-H_LOSS * np.where(ok, X, 0))).sum(1)
    return float(w @ (out / len(CASES)))
KS, QS, GEOS, DISTS = (0.25, 0.5, 1.0, 2.0), (0.0, 0.5, 0.8), ("xia150", "xia47", "pop", "refuge"), ("gc", "b3_0", "b3_300", "b10_0", "b10_300")
SC = [(k, q, g, d) for k in KS for q in QS for g in GEOS for d in DISTS]; R["n_scenarios"] = len(SC)
def table(hazard=False):
    T = {p: [] for p in list(PLANS) + ["random"]}
    for (k, q, g, d) in SC:
        for p, P in PLANS.items(): T[p].append(A_of(P, DM[d], GEO[g], k, q, hazard))
        T["random"].append(float(np.mean([A_of(P, DM[d], GEO[g], k, q, hazard) for P in RAND[:(200 if not hazard else 50)]])))
    return {p: np.array(v) for p, v in T.items()}
def summarize(T):
    PL = list(PLANS); M = np.vstack([T[p] for p in PL]); best = M.max(0); reg = best - M
    S = {p: {"mean_A": float(T[p].mean()), "worst_A": float(T[p].min()), "max_regret": float(reg[i].max()), "mean_regret": float(reg[i].mean()),
             "share_best": float(np.mean(M[i] >= best - 1e-12))} for i, p in enumerate(PL)}
    S["random"] = {"mean_A": float(T["random"].mean()), "worst_A": float(T["random"].min()), "max_regret": float((best - T["random"]).max())}
    return S, min(PL, key=lambda p: S[p]["max_regret"]), max(PL, key=lambda p: S[p]["worst_A"])
TA = table(); SA, mmr, mxm = summarize(TA)
R["summary"] = SA; R["minimax_regret_plan"] = mmr; R["maximin_plan"] = mxm
R["table"] = {p: [round(x, 6) for x in v] for p, v in TA.items()}; R["scenarios"] = [list(s) for s in SC]
# 距離の定義ごとの選び方
R["by_dist"] = {}
for d in DISTS:
    idx = [i for i, s in enumerate(SC) if s[3] == d]; S, a, b = summarize({p: v[idx] for p, v in TA.items()})
    R["by_dist"][d] = {"minimax_regret_plan": a, "maximin_plan": b, "max_regret": {p: S[p]["max_regret"] for p in PLANS}, "mean_A": {p: S[p]["mean_A"] for p in PLANS}}
R["by_geo"] = {}
for g in GEOS:
    idx = [i for i, s in enumerate(SC) if s[2] == g]; S, a, b = summarize({p: v[idx] for p, v in TA.items()})
    R["by_geo"][g] = {"minimax_regret_plan": a, "maximin_plan": b, "mean_A": {p: S[p]["mean_A"] for p in PLANS}}
# 順位の入れ替わり（gc と各障壁、48 の (k,q,地理) の組）
def kendall(a, b):
    n = len(a); c = 0; t = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]); c += s; t += 1
    return c / t
PL = list(PLANS); flips = {}
for d in DISTS[1:]:
    diff = 0; taus = []
    for k in KS:
        for q in QS:
            for g in GEOS:
                i0 = SC.index((k, q, g, "gc")); i1 = SC.index((k, q, g, d))
                a = [TA[p][i0] for p in PL]; b = [TA[p][i1] for p in PL]
                diff += PL[int(np.argmax(a))] != PL[int(np.argmax(b))]; taus.append(kendall(a, b))
    flips[d] = {"top_changed_share": diff / 48, "mean_kendall_tau": float(np.mean(taus)), "min_kendall_tau": float(np.min(taus))}
R["rank_flips_vs_gc"] = flips
# 道中の損失の版
TH = table(hazard=True); SH, mmrH, mxmH = summarize(TH); R["hazard"] = {"summary": SH, "minimax_regret_plan": mmrH, "maximin_plan": mxmH}
# ---------- 仮説 ----------
H = {}
H["H84b"] = {"class": "予測", "value": mmr, "rule": "minimax regret の最良が cover1500 でない", "supported": mmr != "cover1500"}
c750 = float(np.mean(TA["cover750"] >= TA["cover1500"] - 1e-12))
H["H84c"] = {"class": "予測", "value": c750, "rule": "cover750 >= cover1500 の条件の割合 >= 0.75", "supported": c750 >= 0.75}
H["H84d"] = {"class": "予測", "value": flips["b10_300"]["top_changed_share"], "rule": "gc と b10_300 で最良の置き方が違う組の割合 >= 0.25", "supported": flips["b10_300"]["top_changed_share"] >= 0.25}
worst = max(PL, key=lambda p: SA[p]["max_regret"])
H["H84e"] = {"class": "予測", "value": [worst, SA["surv150"]["max_regret"]], "rule": "surv150 の最大の後悔が7つの中で最大", "supported": worst == "surv150"}
i0 = SC.index((1.0, 0.5, "refuge", "gc")); i1 = SC.index((1.0, 0.5, "refuge", "b10_300"))
drop = float(np.mean([TA[p][i0] for p in PL]) - np.mean([TA[p][i1] for p in PL]))
H["H84f"] = {"class": "予測", "value": drop, "rule": "refuge・k=1・q=0.5 で gc − b10_300 の7つの平均 >= 0.10", "supported": drop >= 0.10}
NS = json.load(open(os.path.join(HERE, "sources", "cases.json"), encoding="utf8"))
ind = [c["km"] for c in NS["cases"] if c["cls"] == "indiv"]; strict = [c["km"] for c in NS["cases"] if c["cls"] == "indiv" and c["flag"] == "routine"]
H["H84a"] = {"class": "確認的", "value": {"n": len(ind), "median_km": float(np.median(ind)), "strict_n": len(strict)},
             "rule": "個人の旅の距離の中央値 < 803 km（3件未満は評価不能）", "supported": (len(ind) >= 3 and float(np.median(ind)) < 803) if len(ind) >= 3 else None}
R["nonstate_network_km"] = [c["km"] for c in NS["cases"] if c["cls"] == "network"]
R["hypotheses"] = H
# 非国家の系列を届く距離に使った版（記述・探索）：個人の旅3件の距離
CASES_SAVE = CASES; CASES = np.array(ind, float); TN = {}
for p, P in PLANS.items(): TN[p] = [A_of(P, DM[d], GEO[g], 1.0, q) for q in QS for g in GEOS for d in DISTS]
CASES = CASES_SAVE
R["nonstate_reach_explore"] = {p: float(np.mean(v)) for p, v in TN.items()}
# ---------- 図 ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.ticker
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
M = np.vstack([TA[p] for p in PL]); reg = M.max(0) - M
fig, ax = plt.subplots(figsize=(13, 4.2)); im = ax.imshow(reg, aspect="auto", cmap="magma_r", vmin=0, vmax=max(0.05, reg.max()))
ax.set_yticks(range(len(PL))); ax.set_yticklabels(PL); ax.set_xlabel("scenario (k × q × geography × distance; 240)")
for x in range(60, 240, 60): ax.axvline(x - 0.5, color="w", lw=1)
ax.set_title("Regret of each placement (best plan in scenario − plan); blocks: k = 0.25, 0.5, 1, 2"); plt.colorbar(im, ax=ax, label="regret in A")
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_regret.png"), dpi=110); plt.close()
fig, axs = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, g in zip(axs, ("xia150", "pop", "refuge")):
    for p, c in zip(PL, plt.cm.tab10.colors):
        for d, ls in (("gc", "-"), ("b10_300", "--")):
            ax.plot(KS, [TA[p][SC.index((k, 0.5, g, d))] for k in KS], ls, color=c, label=f"{p} {d}" if g == "xia150" else None, marker="o", ms=3)
    ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.set_title(f"{g}, q = 0.5"); ax.set_xlabel("k (reach multiplier)")
axs[0].set_ylabel("A (share of weighted survivors reaching ≥1 surviving copy)"); axs[0].legend(fontsize=6, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_A_vs_k.png"), dpi=110); plt.close()
fig, ax = plt.subplots(figsize=(11, 5.5)); land = LAND.reshape(NLA, NLO)
ax.imshow(land, origin="lower", extent=(-180, 180, -90, 90), cmap="Greys", alpha=0.3)
for (p, mk, c) in (("cover1500", "o", "C0"), ("cover750", "s", "C1"), ("per_subregion", "^", "C2")):
    ax.scatter(LON[PLANS[p]], LAT[PLANS[p]], marker=mk, facecolors="none", edgecolors=c, s=40 if mk == "o" else 25, label=p)
ax.legend(); ax.set_title(f"Sites of three placements (N = {N})"); plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_sites.png"), dpi=110); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ("N", "n_countries", "check_grid_vs_gc", "barrier_vs_gc_median_ratio", "minimax_regret_plan", "maximin_plan", "rank_flips_vs_gc", "kentoshi")}, ensure_ascii=False, indent=1))
for p in list(PL) + ["random"]: print(p, {k: round(v, 4) for k, v in SA[p].items()})
print("hazard:", mmrH, mxmH, {p: round(SH[p]["max_regret"], 4) for p in PL})
print("by_dist:", {d: (v["minimax_regret_plan"], v["maximin_plan"]) for d, v in R["by_dist"].items()})
print("by_geo:", {d: (v["minimax_regret_plan"], v["maximin_plan"]) for d, v in R["by_geo"].items()})
for h, v in H.items(): print(h, v)
print("nonstate explore", R["nonstate_reach_explore"]); print("runtime", R["runtime_s"])
