"""研究090：全世界の GeoNames の群の点に農村の人口と崩壊の直後の市から郡部への移動を入れる（open_problems 218＋221）。
計算のみ（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
087 の sim.py を「新しい点の集合」の前まで exec（075・084 の差0の assert を含む）。新しい乱数なし（075 の UPT を共通に使う）。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p87 = os.path.abspath(os.path.join(RS, "087-geonames-target-loss", "sim.py")); s = open(p87, encoding="utf8").read()
G = {"__file__": p87, "__name__": "sim087"}; exec(compile(s[: s.index("# ---------- 新しい点の集合 ----------")], p87, "exec"), G)
r75 = json.load(open(os.path.join(RS, "075-reach-to-tc-copies", "results.json"), encoding="utf8"))
R = {"study": "090-rural-survivor-points", "seed": None, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。", "check_087": G["R"]["check"]}
RT, DV, M1, N, GMAX = G["RT"], G["DV"], G["M1"], G["N"], G["GMAX"]; ratio_all, ver, rho75, REACH, valid = G["ratio_all"], G["ver"], G["rho75"], G["REACH"], G["valid"]
PS, counts_q, U_geo, UPT, GEO3, GEO, A2, NK = G["PS"], G["counts_q"], G["U_geo"], G["UPT"], G["GEO3"], G["GEO"], G["A2"], G["NK"]
plat, plon, ppop, pc, PT0, SITES, PLP, MESH = G["plat"], G["plon"], G["ppop"], G["pc"], G["PT0"], G["SITES"], G["PLP"], G["MESH"]
a2i = {a: i for i, a in enumerate(A2)}; h = lambda nm: RT[nm][M1]["heavy"]
def PT_s(ps, mass):  # mass: 点ごとの国の中の相対の重み
    tot = np.bincount(ps.c, weights=mass, minlength=NK); PT = np.zeros((N, GMAX), int)
    for j, g in enumerate(GEO3):
        w = GEO[g][ps.c] * mass / np.where(tot[ps.c] > 0, tot[ps.c], 1); cdf = np.cumsum(w); cdf /= cdf[-1]; m = U_geo == j
        PT[m] = np.minimum(np.searchsorted(cdf, UPT[m], side="right"), ps.n - 1)
    return PT
assert (PT_s(G["NEPS"], ppop) == PT0).all(); print("check PT_s ok", round(time.time() - T0), "s", flush=True)
# ---------- 資料 ----------
def wb(ind):
    d = json.load(open(os.path.join(HERE, "sources", f"wb_{ind}.json"), encoding="utf8"))[1]; return {x["country"]["id"]: x["value"] for x in d if x["value"] is not None}
POP, RUR = wb("SP.POP.TOTL"), wb("SP.RUR.TOTL.ZS")
la, lo, po, cc = [], [], [], []
for l in open(os.path.join(RS, "087-geonames-target-loss", "sources", "cities1000.txt"), encoding="utf8"):
    f = l.rstrip("\n").split("\t")
    if f[8] in a2i and f[7] != "PPLX": la.append(float(f[4])); lo.append(float(f[5])); po.append(float(f[14] or 0)); cc.append(a2i[f[8]])
la, lo, po, cc = map(np.array, (la, lo, po, cc)); po = np.maximum(po, 1.0)
has = np.bincount(cc, minlength=NK) > 0; keep = ~has[pc]
ps = PS(np.r_[plat[keep], la], np.r_[plon[keep], lo], np.r_[ppop[keep], po], np.r_[pc[keep], cc], np.r_[np.zeros(keep.sum(), int), np.ones(len(la), int)])
gn_c = np.bincount(cc, weights=po, minlength=NK)
inwb = [a for a in A2 if a in POP and a in RUR]; popw = np.array([POP[a] for a in inwb]); rurw = np.array([RUR[a] / 100 for a in inwb])
world_rur = float((popw * rurw).sum() / popw.sum())
gn_sum = float(sum(gn_c[a2i[a]] for a in inwb)); wb_sum = float(popw.sum())
R["data"] = {"n_points": int(ps.n), "n_geonames": int(len(la)), "n_countries_A2": len(A2), "n_countries_with_geonames": int(has.sum()), "n_countries_in_wb": len(inwb),
             "geonames_pop_over_wb_pop": gn_sum / wb_sum, "world_rural_share_popweighted": world_rur,
             "geonames_over_wb_by_country_median": float(np.median([gn_c[a2i[a]] / POP[a] for a in inwb if has[a2i[a]]]))}
print(R["data"], flush=True)
u = np.array([1 - RUR[a] / 100 if a in RUR else 1 - world_rur for a in A2])  # 都市の割合
# ---------- 重み ----------
small = ps.pop < 10000
def mass(rural="eq_small", mu=(1.0, 1.0)):
    c = ps.c; pt = np.bincount(c, weights=ps.pop, minlength=NK)
    if rural is None: return ps.pop.copy()
    if rural == "eq_small": r = small.astype(float)
    elif rural == "sqrt_small": r = np.where(small, np.sqrt(ps.pop), 0.0)
    elif rural == "eq_all": r = np.ones(ps.n)
    rc = np.bincount(c, weights=r, minlength=NK); r = np.where(rc[c] > 0, r, 1.0); rc = np.bincount(c, weights=r, minlength=NK)  # 小さな点のない国は全点に
    return mu[0] * u[c] * ps.pop / pt[c] + mu[1] * (1 - u[c]) * r / rc[c]
MU1 = (0.856, 1.245); MU2 = (0.856 ** 2, 1.245 ** 2)
VERS = {"B_GN": dict(rural=None), "R": dict(), "R_sqrt": dict(rural="sqrt_small"), "R_all": dict(rural="eq_all"), "M": dict(mu=MU1), "M2": dict(mu=MU2), "M_sqrt": dict(rural="sqrt_small", mu=MU1)}
def run(nm, S, PT):
    S = list(S); D = counts_q(S, REACH["main"], ps, PT); RT[nm], DV[nm] = ratio_all(ver(rho=rho75(D, 1, len(S), len(S))))
    R.setdefault("reach", {})[nm] = {"share_groups_n0": float((D[valid] == 0).mean()), "mean_n": float(D[valid].mean())}
    print(nm, round(h(nm), 4), round(time.time() - T0), "s", flush=True)
R["rural_share_of_weight"] = {}
for v, kw in VERS.items():
    m = mass(**kw); PT = PT_s(ps, m)
    if kw.get("rural", "x") is not None:
        rpart = m - kw.get("mu", (1, 1))[0] * u[ps.c] * ps.pop / np.bincount(ps.c, weights=ps.pop, minlength=NK)[ps.c]
        R["rural_share_of_weight"][v] = float((GEO["pop"][ps.c] * rpart / np.bincount(ps.c, weights=m, minlength=NK)[ps.c]).sum() / GEO["pop"].sum())
    plans = PLP + ["mesh100", "mesh200"] if v in ("B_GN", "R", "M", "M2") else ["mesh35", "pop@pt", "per_subregion@pt", "mesh200"]
    for p in plans: run(f"{p}_{v}", SITES[p] if p in SITES else MESH[:int(p[4:])], PT)
# ---------- まとめ ----------
def ktau(x, y):
    s_ = 0; c = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)): s_ += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += 1
    return float(s_ / c)
t75 = {p: r75["ratio_to_V0"][p][M1]["heavy"] for p in PLP + ["mesh200"]}
R["heavy_TC"] = {k: h(k) for k in RT if k != "chk070"}
R["rank"] = {}
for v in ("B_GN", "R", "M", "M2"):
    y = [h(f"{p}_{v}") for p in PLP]
    R["rank"][v] = {"tau_075": ktau([t75[p] for p in PLP], y), "best": PLP[int(np.argmax(y))], "order": [PLP[i] for i in np.argsort(y)[::-1]], "gain_35_200": h(f"mesh200_{v}") - h(f"mesh35_{v}")}
R["rank"]["075"] = {"gain_35_200": t75["mesh200"] - t75["mesh35"]}
d = R["data"]; m35 = {v: h(f"mesh35_{v}") for v in VERS}; H = {}
H["H102a"] = {"class": "確認的", "value": d["geonames_pop_over_wb_pop"], "rule": "GeoNames pop sum / WB pop sum < 0.8", "supported": d["geonames_pop_over_wb_pop"] < 0.8}
H["H102b"] = {"class": "確認的", "value": world_rur, "rule": "pop-weighted WB rural share in [0.40, 0.48]", "supported": 0.40 <= world_rur <= 0.48}
x = m35["B_GN"] - t75["mesh35"]; H["H102c"] = {"class": "予測", "value": x, "rule": "|B_GN mesh35 - 0.6801| <= 0.010", "supported": abs(x) <= 0.010}
x = m35["B_GN"] - m35["R"]; H["H102d"] = {"class": "予測", "value": x, "rule": "B_GN - R mesh35 in [0.003, 0.030]", "supported": 0.003 <= x <= 0.030}
x = m35["M"] - m35["R"]; H["H102e"] = {"class": "予測", "value": x, "rule": "|M - R| mesh35 < 0.005", "supported": abs(x) < 0.005}
x = R["rank"]["M"]; H["H102f"] = {"class": "予測", "value": [x["tau_075"], x["best"]], "rule": "tau(075, M) >= 0.7 and best == mesh35", "supported": x["tau_075"] >= 0.7 and x["best"] == "mesh35"}
x = R["rank"]["M"]["gain_35_200"]; H["H102g"] = {"class": "予測", "value": x, "rule": "M gain 35->200 >= 075 gain (0.049)", "supported": x >= R["rank"]["075"]["gain_35_200"]}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
xs = np.arange(len(PLP))
for k, v in enumerate(("B_GN", "R", "M", "M2")): axs[0].plot(xs, [h(f"{p}_{v}") for p in PLP], "o-", label=v, ms=4)
axs[0].plot(xs, [t75[p] for p in PLP], "k--", label="075 (NE points)"); axs[0].set_xticks(xs); axs[0].set_xticklabels(PLP, rotation=60, ha="right", fontsize=7)
axs[0].set_ylabel("heavy-collapse TC/V0 (not survival probability)"); axs[0].legend(fontsize=7); axs[0].set_title("10 placements by survivor point set")
vv = list(VERS); axs[1].bar(np.arange(len(vv)) - .2, [m35[v] for v in vv], .4, label="mesh35"); axs[1].bar(np.arange(len(vv)) + .2, [h(f"mesh200_{v}") for v in vv], .4, label="mesh200")
axs[1].axhline(t75["mesh35"], color="k", ls="--", lw=.8, label="075 mesh35"); axs[1].set_xticks(range(len(vv))); axs[1].set_xticklabels(vv); axs[1].set_ylim(0.6, 0.76); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_rural_points.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("data", "rural_share_of_weight", "rank", "hypotheses")}, ensure_ascii=False, indent=1, default=float)); print("runtime", R["runtime_s"])
