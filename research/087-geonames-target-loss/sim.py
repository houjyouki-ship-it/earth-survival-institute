"""研究087：細かい地名の点の集合（GeoNames cities1000）での核の標的の都市の写しの喪失と生き残り（open_problems 217）。
計算のみ（ネット接続なし）。事前登録：PREREGISTRATION.md。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
075 の sim.py を「PP = G74["PP"]」の前まで exec（070・074 の差0の検査を含む）。群の点（生き残りの居場所）と写しの場所（074 の NE の点）を分けた
counts_q を書き、群の点＝NE の点で 075 の mesh35 と 084 の mesh35・pop@pt を差0で再現（assert）してから、標的の国の群の点を GeoNames に置き換える。
新しい乱数なし（075 の UPT を共通に使う）。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p75 = os.path.abspath(os.path.join(RS, "075-reach-to-tc-copies", "sim.py")); s = open(p75, encoding="utf8").read()
G = {"__file__": p75, "__name__": "sim075"}; exec(compile(s[: s.index('PP = G74["PP"]')], p75, "exec"), G)
r75 = json.load(open(os.path.join(RS, "075-reach-to-tc-copies", "results.json"), encoding="utf8"))
r84 = json.load(open(os.path.join(RS, "084-nuclear-target-copy-loss", "results.json"), encoding="utf8"))
TGT = json.load(open(os.path.join(RS, "084-nuclear-target-copy-loss", "sources", "targets.json"), encoding="utf8"))
R = {"study": "087-geonames-target-loss", "seed": None, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
D_DIRECT = {"xia150": 360_000_000, "xia47": 164_000_000}  # Xia 2022 Table 1（086 で再取得）
RT, DV, M1, N, GMAX = G["RT"], G["DV"], G["M1"], G["N"], G["GMAX"]; ratio_all, ver, rho75, counts, REACH, valid = G["ratio_all"], G["ver"], G["rho75"], G["counts"], G["REACH"], G["valid"]
G74 = G["G74"]; PP = G74["PP"]; MESH = list(G74["mesh"]); SITES = dict(PP); PLP = list(PP) + ["mesh35"]; SITES["mesh35"] = MESH[:35]
hav, plat, plon, ppop, pc, A2, GEO = G["hav"], G["plat"], G["plon"], G74["ppop"], G74["pc"], G74["A2"], G74["GEO"]
U_geo = G["U_geo"]; GEO3 = G["GEO3"]; NPT = G["NPT"]; UPT = G["UPT"]; PT0 = G["PT"].copy(); NK = int(pc.max()) + 1
h = lambda nm: RT[nm][M1]["heavy"]
cA2 = np.array([A2[i] for i in pc]); a2i = {a: i for i, a in enumerate(A2)}
TC_ALL = sorted(set(TGT["xia150"]["weapons"]) | set(TGT["xia47"]["weapons"]))
def mindist(la, lo, tla, tlo, chunk=4000):
    out = np.full(len(la), np.inf)
    for a in range(0, len(la), chunk): out[a:a + chunk] = hav(la[a:a + chunk][:, None], lo[a:a + chunk][:, None], tla[None, :], tlo[None, :]).min(1)
    return out
# ---------- 点の集合 ----------
class PS:  # 群の点の集合
    def __init__(s, lat, lon, pop, c, src): s.lat, s.lon, s.pop, s.c, s.src = map(np.asarray, (lat, lon, pop, c, src)); s.n = len(s.lat); s.cA2 = np.array([A2[i] for i in s.c])
NEPS = PS(plat, plon, ppop, pc, np.zeros(NPT, int))
def load_geonames(minpop=1000, drop_pplx=False):
    la, lo, po, cc, fc = [], [], [], [], []
    for l in open(os.path.join(HERE, "sources", "cities1000.txt"), encoding="utf8"):
        f = l.rstrip("\n").split("\t")
        if f[8] in TC_ALL and int(f[14] or 0) >= minpop and not (drop_pplx and f[7] == "PPLX"):
            la.append(float(f[4])); lo.append(float(f[5])); po.append(float(f[14])); cc.append(a2i[f[8]]); fc.append(f[7])
    return np.array(la), np.array(lo), np.array(po), np.array(cc), np.array(fc)
def mixed(minpop=1000, drop_pplx=False):
    la, lo, po, cc, fc = load_geonames(minpop, drop_pplx); keep = ~np.isin(cA2, TC_ALL)
    ps = PS(np.r_[plat[keep], la], np.r_[plon[keep], lo], np.r_[ppop[keep], po], np.r_[pc[keep], cc], np.r_[np.zeros(keep.sum(), int), np.ones(len(la), int)])
    ps.fc = np.r_[np.array([""] * keep.sum()), fc]; return ps
# ---------- 標的・重み・群の点の引き方 ----------
def targets(ps, weapons):
    out = []
    for c, n in weapons.items():
        idx = np.where(ps.cA2 == c)[0]; assert len(idx), c
        out += list(idx[np.argsort(-ps.pop[idx], kind="stable")][:n])
    return np.array(out)
def weights(ps, g, tg=None, Rd=10.0, info=None):
    sp = ps.pop; base = GEO[g][ps.c] * sp / np.bincount(ps.c, weights=sp, minlength=NK)[ps.c]
    if tg is None: return base
    dz = mindist(ps.lat, ps.lon, ps.lat[tg], ps.lon[tg]) <= Rd; f = np.zeros(ps.n); f[dz] = min(1.0, D_DIRECT[g] / ps.pop[dz].sum())
    sp2 = sp * (1 - f); tot2 = np.bincount(ps.c, weights=sp2, minlength=NK); full = tot2[ps.c] <= 0
    w = np.where(full, base, GEO[g][ps.c] * sp2 / np.where(full, 1, tot2[ps.c]))
    if info is not None:
        info.update({"n_zone_pts": int(dz.sum()), "pop_zone": float(ps.pop[dz].sum()), "f": float(f[dz].max()) if dz.any() else 0.0,
                     "geo_share_in_zone_before": float(base[dz].sum() / base.sum()), "geo_share_in_zone_after": float(w[dz].sum() / w.sum()),
                     "share_pts_outside_zone_by_country": {c: float(1 - dz[ps.cA2 == c].mean()) for c in TGT[g]["weapons"]}})
    return w
def PT_of(ps, tgs=None, Rd=10.0, info=None):
    PT = np.zeros((N, GMAX), int)
    for j, g in enumerate(GEO3):
        tg = None if tgs is None else tgs.get(g); ii = {} if info is not None else None
        cdf = np.cumsum(weights(ps, g, tg, Rd, ii)); cdf /= cdf[-1]; m = U_geo == j
        PT[m] = np.minimum(np.searchsorted(cdf, UPT[m], side="right"), ps.n - 1)
        if info is not None and tg is not None: info[g] = ii
    return PT
def counts_q(S, reach, ps, PT):  # 075 の counts と同じ式で、群の点の集合 ps と写しの場所（NE の点）を分ける
    S = np.array(S); Ds = np.sort(hav(ps.lat[:, None], ps.lon[:, None], plat[S][None, :], plon[S][None, :]), axis=1).astype(np.float32) if len(S) else np.zeros((ps.n, 0), np.float32)
    out = np.full((N, GMAX), -1, int); ii, jj = np.nonzero(valid); pp = PT[ii, jj]; rr = reach[ii, jj].astype(np.float32); n = np.empty(len(ii), int)
    for c in range(0, len(ii), 40000): n[c:c + 40000] = (Ds[pp[c:c + 40000]] <= rr[c:c + 40000, None]).sum(1)
    out[ii, jj] = n; return out
def run(nm, S, ps, PT, tgs=None, Rd=10.0):
    S = list(S)
    if tgs is None: Sg = [S, S, S]
    else: Sg = [[x for x, k in zip(S, mindist(plat[np.array(S)], plon[np.array(S)], ps.lat[tgs[g]], ps.lon[tgs[g]]) <= Rd) if not k] for g in ("xia150", "xia47")] + [S]
    Ds = [counts_q(x, REACH["main"], ps, PT) for x in Sg]
    D = np.where((U_geo == 0)[:, None], Ds[0], np.where((U_geo == 1)[:, None], Ds[1], Ds[2]))
    RT[nm], DV[nm] = ratio_all(ver(rho=rho75(D, 1, len(S), len(S))))
    R.setdefault("loss", {})[nm] = {"n": len(S), "share150": 1 - len(Sg[0]) / len(S), "share47": 1 - len(Sg[1]) / len(S)}
    print(nm, round(h(nm), 4), R["loss"][nm], round(time.time() - T0), "s", flush=True)
# ---------- 検査：NE の点で 075・084 を差0 ----------
assert (PT_of(NEPS) == PT0).all()
D1 = counts_q(MESH[:35], REACH["main"], NEPS, PT0); assert (D1 == counts(MESH[:35], REACH["main"])).all()
run("chk75_mesh35", MESH[:35], NEPS, PT0); d75 = max(abs(RT["chk75_mesh35"][k][p] - r75["ratio_to_V0"]["mesh35"][k][p]) for k in RT["chk75_mesh35"] for p in RT["chk75_mesh35"][k])
TGNE = {g: targets(NEPS, TGT[g]["weapons"]) for g in ("xia150", "xia47")}; chk = {}
for p in ("mesh35", "pop@pt"): run("chk84_" + p, SITES[p], NEPS, PT0, TGNE); chk[p] = abs(h("chk84_" + p) - r84["heavy_TC"][p])
R["check"] = {"075_mesh35_maxdiff": d75, "084_absdiff": chk}; assert d75 < 1e-12 and all(v < 1e-12 for v in chk.values()), R["check"]
print("checks ok", flush=True)
# ---------- 新しい点の集合 ----------
VARS = {"main": dict(minpop=1000), "pop15k": dict(minpop=15000), "noPPLX": dict(minpop=1000, drop_pplx=True)}
R["pointsets"] = {}; R["zone_info"] = {}
def setup(tag, **kw):
    ps = mixed(**kw); tgs = {g: targets(ps, TGT[g]["weapons"]) for g in ("xia150", "xia47")}; info = {}
    PTb = PT_of(ps); PTs = PT_of(ps, tgs, 10.0, info)
    R["pointsets"][tag] = {"n_points": int(ps.n), "n_geonames": int(ps.src.sum()), "by_country": {c: int((ps.cA2 == c).sum()) for c in TC_ALL}}
    R["zone_info"][tag] = info; return ps, tgs, PTb, PTs
ps, tgs, PTb, PTs = setup("main", **VARS["main"])
for p in PLP: run(f"{p}_B0", SITES[p], ps, PTb); run(f"{p}_CS", SITES[p], ps, PTs, tgs)
for n in (100, 200): run(f"mesh{n}_B0", MESH[:n], ps, PTb); run(f"mesh{n}_CS", MESH[:n], ps, PTs, tgs)
for n in (35, 200): run(f"mesh{n}_C", MESH[:n], ps, PTb, tgs)
for Rd in (5.0, 25.0):
    PTr = PT_of(ps, tgs, Rd)
    for n in (35, 200): run(f"mesh{n}_CS_Rd{int(Rd)}", MESH[:n], ps, PTr, tgs, Rd)
for tag in ("pop15k", "noPPLX"):
    ps2, tgs2, PTb2, PTs2 = setup(tag, **VARS[tag])
    for n in (35, 200): run(f"mesh{n}_B0_{tag}", MESH[:n], ps2, PTb2); run(f"mesh{n}_CS_{tag}", MESH[:n], ps2, PTs2, tgs2)
# ---------- まとめ ----------
def ktau(x, y):
    s_ = 0; c = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)): s_ += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += 1
    return float(s_ / c)
t75 = {p: r75["ratio_to_V0"][p][M1]["heavy"] for p in PLP} | {f"mesh{n}": r75["ratio_to_V0"][f"mesh{n}"][M1]["heavy"] for n in (100, 200)}
t84 = {k: r84["heavy_TC"][k] for k in PLP + ["mesh100", "mesh200"]}
R["heavy_TC"] = {k: h(k) for k in RT if k != "chk070"}
y84 = [t84[p] for p in PLP]; yCS = [h(f"{p}_CS") for p in PLP]; yB0 = [h(f"{p}_B0") for p in PLP]
R["rank"] = {"tau_084_CS": ktau(y84, yCS), "tau_B0_CS": ktau(yB0, yCS), "tau_075_B0": ktau([t75[p] for p in PLP], yB0), "best_CS": PLP[int(np.argmax(yCS))], "best_084": PLP[int(np.argmax(y84))],
             "spread_CS": max(yCS) - min(yCS), "order_CS": [PLP[i] for i in np.argsort(yCS)[::-1]]}
loss84 = t75["mesh35"] - t84["mesh35"]
L = {"084": loss84, "C": h("mesh35_B0") - h("mesh35_C"), "CS": h("mesh35_B0") - h("mesh35_CS"), "CS_Rd5": h("mesh35_B0") - h("mesh35_CS_Rd5"), "CS_Rd25": h("mesh35_B0") - h("mesh35_CS_Rd25"),
     "CS_pop15k": h("mesh35_B0_pop15k") - h("mesh35_CS_pop15k"), "CS_noPPLX": h("mesh35_B0_noPPLX") - h("mesh35_CS_noPPLX"), "B0_minus_075": h("mesh35_B0") - t75["mesh35"]}
R["loss_mesh35"] = L; R["loss_ratio_to_084"] = {k: v / loss84 for k, v in L.items() if k not in ("084", "B0_minus_075")}
R["gain_35_200"] = {"075": t75["mesh200"] - t75["mesh35"], "084": t84["mesh200"] - t84["mesh35"], "B0": h("mesh200_B0") - h("mesh35_B0"), "CS": h("mesh200_CS") - h("mesh35_CS"), "C": h("mesh200_C") - h("mesh35_C")}
H = {}
v = R["loss_ratio_to_084"]["CS"]; H["H99a"] = {"class": "確認的", "value": v, "rule": "(B0-CS)/0.0171 <= 0.5", "supported": v <= 0.5}
v = R["loss"]["mesh35_CS"]["share150"]; H["H99b"] = {"class": "確認的", "value": v, "rule": "xia150 で失う mesh35 の割合 <= 0.30", "supported": v <= 0.30}
v = R["rank"]["tau_084_CS"]; H["H99c"] = {"class": "予測", "value": v, "rule": "tau(084, CS) >= 0.7", "supported": v >= 0.7}
v = R["gain_35_200"]["CS"]; H["H99d"] = {"class": "予測", "value": v, "rule": "CS の 35->200 >= +0.044", "supported": v >= 0.044}
H["H99e"] = {"class": "予測", "value": [L["CS"], L["C"]], "rule": "B0-CS < B0-C", "supported": L["CS"] < L["C"]}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8)); x = np.arange(len(PLP))
for k, (lab, y) in enumerate((("075 NE, no loss", [t75[p] for p in PLP]), ("084 NE, copies lost", y84), ("087 B0 GeoNames, no loss", yB0), ("087 CS GeoNames, copies+survivors", yCS))):
    axs[0].bar(x + (k - 1.5) * .2, y, .2, label=lab)
allv = [t75[p] for p in PLP] + y84 + yB0 + yCS; axs[0].set_ylim(min(allv) - .01, max(allv) + .01); axs[0].set_xticks(x); axs[0].set_xticklabels(PLP, rotation=30, fontsize=7)
axs[0].set_ylabel("heavy TC/V0 (not survival probability)"); axs[0].legend(fontsize=7)
ks = ["084", "C", "CS", "CS_Rd5", "CS_Rd25", "CS_pop15k", "CS_noPPLX"]; axs[1].bar(ks, [L[k] for k in ks], color=["C1"] + ["C3"] * 6)
axs[1].set_ylabel("mesh35 heavy TC loss (vs own no-loss baseline)"); axs[1].tick_params(axis="x", labelsize=7); axs[1].set_title("NE points (084) vs GeoNames points (087)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_geonames_loss.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check", "pointsets", "zone_info", "rank", "loss_mesh35", "loss_ratio_to_084", "gain_35_200", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print("runtime", R["runtime_s"])
