"""研究086：核の標的の都市で生き残りを減らすと、084 の写しの喪失の損はどれだけ残るか（open_problems 213）。
計算のみ（ネット接続なし）。事前登録：PREREGISTRATION.md。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
075 の sim.py を「PP = G74["PP"]」の前まで exec し（070・074 の差0の検査を含む）、084 の run_nuc（α=0）を同じ式で書き、
084 の results.json の重い崩壊の TC を差0で再現してから、標的から R_d 以内の群の点の重みに (1-f) を掛ける。
直接の死者 D は Xia et al. 2022 Table 1（sources/xia_table1.html）：150 Tg 3.6億人、47 Tg 1.64億人。新しい乱数なし（075 の UPT を共通に使う）。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p75 = os.path.abspath(os.path.join(RS, "075-reach-to-tc-copies", "sim.py")); s = open(p75, encoding="utf8").read()
G = {"__file__": p75, "__name__": "sim075"}; exec(compile(s[: s.index('PP = G74["PP"]')], p75, "exec"), G)
r75 = json.load(open(os.path.join(RS, "075-reach-to-tc-copies", "results.json"), encoding="utf8"))
r84 = json.load(open(os.path.join(RS, "084-nuclear-target-copy-loss", "results.json"), encoding="utf8"))
TGT = json.load(open(os.path.join(RS, "084-nuclear-target-copy-loss", "sources", "targets.json"), encoding="utf8"))
R = {"study": "086-target-survivor-weight", "seed": None, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
D_DIRECT = {"xia150": 360_000_000, "xia47": 164_000_000}  # Xia 2022 Table 1
RT, DV, M1, N, GMAX = G["RT"], G["DV"], G["M1"], G["N"], G["GMAX"]; ratio_all, ver, rho75, counts, REACH = G["ratio_all"], G["ver"], G["rho75"], G["counts"], G["REACH"]
G74 = G["G74"]; PP = G74["PP"]; MESH = list(G74["mesh"]); SITES = dict(PP); PLP = list(PP) + ["mesh35"]; SITES["mesh35"] = MESH[:35]
hav, plat, plon, ppop, pc, A2, GEO = G["hav"], G["plat"], G["plon"], G74["ppop"], G74["pc"], G74["A2"], G74["GEO"]
U_geo = G["U_geo"]; GEO3 = G["GEO3"]; NPT = G["NPT"]; UPT = G["UPT"]; PT0 = G["PT"].copy(); NK = int(pc.max()) + 1
h = lambda nm: RT[nm][M1]["heavy"]
cA2 = np.array([A2[i] for i in pc])
def targets(weapons):
    out = []
    for c, n in weapons.items():
        idx = np.where(cA2 == c)[0]; assert len(idx), c
        out += list(idx[np.argsort(-ppop[idx], kind="stable")][:n])
    return np.array(out)
TG = {"xia150": targets(TGT["xia150"]["weapons"]), "xia47": targets(TGT["xia47"]["weapons"])}
assert len(TG["xia150"]) == r84["targets"]["xia150"]["n_points"] and len(TG["xia47"]) == r84["targets"]["xia47"]["n_points"]
def near(idx_pts, tg, Rd):
    idx_pts = np.asarray(idx_pts); out = np.zeros(len(idx_pts), bool)
    for a in range(0, len(idx_pts), 2000):
        b = idx_pts[a:a + 2000]; out[a:a + 2000] = hav(plat[b][:, None], plon[b][:, None], plat[tg][None, :], plon[tg][None, :]).min(1) <= Rd
    return out
def surv(S, tg, Rd): return [x for x, k in zip(S, near(S, tg, Rd)) if not k]
# ---------- 群の点の重み ----------
def weights(g, e=1.0, mode=None, Rd=10.0, info=None):
    sp = ppop ** e; base = GEO[g][pc] * sp / np.bincount(pc, weights=sp, minlength=NK)[pc]
    if mode is None or g not in TG: return base
    dz = near(np.arange(NPT), TG[g], Rd); f = np.zeros(NPT)
    if mode in ("V1", "V4"): f[dz] = min(1.0, D_DIRECT[g] / ppop[dz].sum())
    elif mode == "V2": k = D_DIRECT[g] / len(TG[g]); f[dz] = np.minimum(1.0, k / ppop[dz])
    elif mode == "V3": f[dz] = 1.0
    sp2 = sp * (1 - f)
    if mode == "V4": w = GEO[g][pc] * sp2 / np.bincount(pc, weights=sp, minlength=NK)[pc]
    else:
        tot2 = np.bincount(pc, weights=sp2, minlength=NK); full = tot2[pc] <= 0   # 国の点がすべて壊された国は元のまま（移る先がない）
        w = np.where(full, base, GEO[g][pc] * sp2 / np.where(full, 1, tot2[pc]))
    if info is not None:
        info.update({"n_zone_pts": int(dz.sum()), "pop_zone": float(ppop[dz].sum()), "f_mean_popw": float((f * ppop).sum() / ppop[dz].sum()),
                     "geo_share_in_zone_before": float(base[dz].sum() / base.sum()), "geo_share_in_zone_after": float(w[dz].sum() / w.sum()),
                     "countries_all_destroyed": sorted(set(cA2[(np.bincount(pc, weights=sp2, minlength=NK) <= 0)[pc] & (GEO[g][pc] > 0)].tolist()))})
    return w
def PT_of(**kw):
    PT = np.zeros((N, GMAX), int)
    for j, g in enumerate(GEO3):
        cdf = np.cumsum(weights(g, **kw)); cdf /= cdf[-1]; m = U_geo == j
        PT[m] = np.minimum(np.searchsorted(cdf, UPT[m], side="right"), NPT - 1)
    return PT
assert (PT_of() == PT0).all(); R["check_PT"] = True
def run_nuc(nm, S, PT=None, Rd=10.0):
    S150, S47 = surv(S, TG["xia150"], Rd), surv(S, TG["xia47"], Rd)
    G["PT"] = PT0 if PT is None else PT
    Ds = [counts(Sg, REACH["main"]) for Sg in (S150, S47, list(S))]; G["PT"] = PT0
    D = np.where((U_geo == 0)[:, None], Ds[0], np.where((U_geo == 1)[:, None], Ds[1], Ds[2]))
    RT[nm], DV[nm] = ratio_all(ver(rho=rho75(D, 1, len(S), len(S))))
    print(nm, round(h(nm), 4), round(time.time() - T0), "s", flush=True)
# ---------- 検査：084 を差0 ----------
chk = {}
for p in ("mesh35", "pop@pt"):
    run_nuc("chk84_" + p, SITES[p]); chk[p] = abs(h("chk84_" + p) - r84["heavy_TC"][p])
R["check_084"] = {"abs_diff": chk, "reproduced": all(v < 1e-12 for v in chk.values())}; assert R["check_084"]["reproduced"], chk
print("check ok", flush=True)
# ---------- 版 ----------
INFO = {}
for mode in ("V1", "V2", "V3", "V4"):
    INFO[mode] = {}
    for g in TG: INFO[mode][g] = {}; weights(g, mode=mode, info=INFO[mode][g])
R["zone_info"] = INFO
PTS = {m: PT_of(mode=m) for m in ("V1", "V2", "V3", "V4")}; PTS["V1_e0.66"] = PT_of(mode="V1", e=0.66); PTS["e0.66"] = PT_of(e=0.66)
for p in PLP: run_nuc(f"{p}_V1", SITES[p], PTS["V1"])
for n in (100, 200): run_nuc(f"mesh{n}_V1", MESH[:n], PTS["V1"])
for m in ("V2", "V3", "V4", "V1_e0.66", "e0.66"):
    for n in (35, 200): run_nuc(f"mesh{n}_{m}", MESH[:n], PTS[m])
for p in ("pop@pt", "per_subregion@cty"):
    for m in ("V3", "V1_e0.66"): run_nuc(f"{p}_{m}", SITES[p], PTS[m])
# 版（記述）：壊れの半径 R_d=5・25 km で写しの喪失と生き残りの減りを同じ半径にそろえる（V1）
for Rd in (5.0, 25.0):
    PTr = np.zeros((N, GMAX), int)
    for j, g in enumerate(GEO3):
        cdf = np.cumsum(weights(g, mode="V1", Rd=Rd)); cdf /= cdf[-1]; m = U_geo == j; PTr[m] = np.minimum(np.searchsorted(cdf, UPT[m], side="right"), NPT - 1)
    for n in (35, 200): run_nuc(f"mesh{n}_V1_Rd{int(Rd)}", MESH[:n], PTr, Rd=Rd)
# ---------- まとめ ----------
def ktau(x, y):
    s_ = 0; c = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)): s_ += np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]); c += 1
    return float(s_ / c)
t75 = {p: r75["ratio_to_V0"][p][M1]["heavy"] for p in PLP} | {f"mesh{n}": r75["ratio_to_V0"][f"mesh{n}"][M1]["heavy"] for n in (100, 200)}
t84 = {k: r84["heavy_TC"][k] for k in PLP + ["mesh100", "mesh200"]}
R["heavy_TC"] = {k: h(k) for k in RT if k != "chk070"}; R["heavy_TC_075"] = t75; R["heavy_TC_084"] = t84
y84 = [t84[p] for p in PLP]; y86 = [h(f"{p}_V1") for p in PLP]; y75 = [t75[p] for p in PLP]
R["rank"] = {"tau_084_V1": ktau(y84, y86), "tau_075_V1": ktau(y75, y86), "best_V1": PLP[int(np.argmax(y86))], "best_084": PLP[int(np.argmax(y84))], "spread_V1": max(y86) - min(y86),
             "order_V1": [PLP[i] for i in np.argsort(y86)[::-1]]}
loss84 = t75["mesh35"] - t84["mesh35"]
R["loss_mesh35"] = {"084": loss84, **{m: t75["mesh35"] - h(f"mesh35_{m}") for m in ("V1", "V2", "V3", "V4")}}
R["loss_remaining_ratio"] = {m: R["loss_mesh35"][m] / loss84 for m in ("V1", "V2", "V3", "V4")}
R["gain_35_200"] = {"075": t75["mesh200"] - t75["mesh35"], "084": t84["mesh200"] - t84["mesh35"], **{m: h(f"mesh200_{m}") - h(f"mesh35_{m}") for m in ("V1", "V2", "V3", "V4", "V1_e0.66", "e0.66")}}
H = {}
v = R["loss_remaining_ratio"]["V1"]; H["H98a"] = {"class": "予測", "value": v, "rule": "V1 の mesh35 の損 >= 084 の損の 0.5 倍", "supported": v >= 0.5}
v = R["rank"]["tau_084_V1"]; H["H98b"] = {"class": "予測", "value": v, "rule": "Kendall tau(084, V1) >= 0.7", "supported": v >= 0.7}
v = h("mesh35_V3"); H["H98c"] = {"class": "予測", "value": v, "rule": "V3 mesh35 >= 084 mesh35", "supported": v >= t84["mesh35"]}
v = R["gain_35_200"]["V1"]; H["H98d"] = {"class": "予測", "value": v, "rule": "V1 の 35->200 の上げ幅 <= 084 の上げ幅", "supported": v <= R["gain_35_200"]["084"]}
v = h("mesh35_V1_e0.66"); H["H98e"] = {"class": "予測", "value": v, "rule": "0.655 <= V1+e0.66 mesh35 <= 0.675", "supported": 0.655 <= v <= 0.675}
for x in H.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = H
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
x = np.arange(len(PLP)); axs[0].bar(x - .27, y75, .27, label="075 (no loss)"); axs[0].bar(x, y84, .27, label="084 (copies lost, survivors kept)"); axs[0].bar(x + .27, y86, .27, label="086 V1 (survivors reduced)")
lo = min(y75 + y84 + y86) - .01; axs[0].set_ylim(lo, max(y75 + y84 + y86) + .01); axs[0].set_xticks(x); axs[0].set_xticklabels(PLP, rotation=30, fontsize=7)
axs[0].set_ylabel("heavy TC/V0 (not survival probability)"); axs[0].legend(fontsize=7)
vs = ["084", "V1", "V2", "V3", "V4"]; axs[1].bar(vs, [R["loss_mesh35"][v_] for v_ in vs], color=["C1", "C2", "C2", "C2", "C2"])
axs[1].set_ylabel("mesh35 heavy TC loss vs 075"); axs[1].set_title("loss of TC from target-city copy loss, by survivor-weight version")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_survivor_weight.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_084", "zone_info", "rank", "loss_mesh35", "loss_remaining_ratio", "gain_35_200", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print(json.dumps(R["heavy_TC"], default=float)); print("runtime", R["runtime_s"])
