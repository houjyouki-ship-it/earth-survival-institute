"""研究088：標的の国で写しを標的の外の町へ移す案（open_problems 219、核の標的の系列を閉じる1回）。
計算のみ（ネット接続なし、新しい乱数なし）。事前登録：PREREGISTRATION.md。TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。
087 の sim.py を「# ---------- 新しい点の集合」の前まで exec（075 の exec と 075・084 の差0の assert を含む）。写しの座標を配列で受ける counts_c を書き、
移さない版で 087 の mesh35/mesh200 の B0・CS を差0で再現（assert）してから、標的の区域の写しを移す。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p87 = os.path.abspath(os.path.join(RS, "087-geonames-target-loss", "sim.py")); s = open(p87, encoding="utf8").read()
G = {"__file__": p87, "__name__": "sim087"}; exec(compile(s[: s.index("# ---------- 新しい点の集合")], p87, "exec"), G)
r87 = json.load(open(os.path.join(RS, "087-geonames-target-loss", "results.json"), encoding="utf8"))
RT, DV, M1, N, GMAX, ratio_all, ver, rho75, REACH, valid = (G[k] for k in ("RT", "DV", "M1", "N", "GMAX", "ratio_all", "ver", "rho75", "REACH", "valid"))
hav, plat, plon, pc, A2, U_geo, mindist, mixed, targets, PT_of, TGT, MESH = (G[k] for k in ("hav", "plat", "plon", "pc", "A2", "U_geo", "mindist", "mixed", "targets", "PT_of", "TGT", "MESH"))
h = lambda nm: RT[nm][M1]["heavy"]
R = {"study": "088-relocate-out-of-target", "seed": None, "note": "TC/V0 は生存確率・復興確率ではない。提言は模型の中の順位（外部の検証はまだ）。"}
GS = ("xia150", "xia47")
def counts_c(cla, clo, reach, ps, PT):
    if len(cla): Ds = np.sort(hav(ps.lat[:, None], ps.lon[:, None], cla[None, :], clo[None, :]), axis=1).astype(np.float32)
    else: Ds = np.zeros((ps.n, 0), np.float32)
    out = np.full((N, GMAX), -1, int); ii, jj = np.nonzero(valid); pp = PT[ii, jj]; rr = reach[ii, jj].astype(np.float32); n = np.empty(len(ii), int)
    for c in range(0, len(ii), 40000): n[c:c + 40000] = (Ds[pp[c:c + 40000]] <= rr[c:c + 40000, None]).sum(1)
    out[ii, jj] = n; return out
def run_c(nm, cla, clo, ps, PT, tgs=None, Rd=10.0):
    if tgs is None: keep = [np.ones(len(cla), bool)] * 3
    else: keep = [mindist(cla, clo, ps.lat[tgs[g]], ps.lon[tgs[g]]) > Rd for g in GS] + [np.ones(len(cla), bool)]
    Ds = [counts_c(cla[k], clo[k], REACH["main"], ps, PT) for k in keep]
    D = np.where((U_geo == 0)[:, None], Ds[0], np.where((U_geo == 1)[:, None], Ds[1], Ds[2]))
    RT[nm], DV[nm] = ratio_all(ver(rho=rho75(D, 1, len(cla), len(cla))))
    R.setdefault("lost_share", {})[nm] = {g: float(1 - keep[i].mean()) for i, g in enumerate(GS)}
    print(nm, round(h(nm), 4), R["lost_share"][nm], round(time.time() - T0), "s", flush=True)
# ---------- 087 の主の点の集合 ----------
ps = mixed(minpop=1000); tgs = {g: targets(ps, TGT[g]["weapons"]) for g in GS}
PTb = PT_of(ps); PTs = PT_of(ps, tgs, 10.0); PTs25 = PT_of(ps, tgs, 25.0)
TU = np.r_[tgs["xia150"], tgs["xia47"]]; tla, tlo = ps.lat[TU], ps.lon[TU]
gn = np.where(ps.src == 1)[0]; gn_dt = mindist(ps.lat[gn], ps.lon[gn], tla, tlo)
def relocate(S, Rmax=200.0, Rbuf=10.0, Rd=10.0, rule="largest"):
    S = np.array(S); cla, clo = plat[S].astype(float).copy(), plon[S].astype(float).copy()
    dz = mindist(cla, clo, tla, tlo) <= Rd; info = {"n_in_zone": int(dz.sum()), "moved": 0, "stuck": 0, "move_km": [], "dest_pop": []}
    for i in np.where(dz)[0]:
        c = A2[pc[S[i]]]; cand = gn[(ps.cA2[gn] == c) & (gn_dt > Rbuf)]
        d = hav(cla[i], clo[i], ps.lat[cand], ps.lon[cand]); ok = d <= Rmax
        if not ok.any(): info["stuck"] += 1; continue
        cand, d = cand[ok], d[ok]; j = int(np.argmax(ps.pop[cand])) if rule == "largest" else int(np.argmin(d))
        cla[i], clo[i] = ps.lat[cand[j]], ps.lon[cand[j]]; info["moved"] += 1; info["move_km"].append(float(d[j])); info["dest_pop"].append(float(ps.pop[cand[j]]))
    info["median_move_km"] = float(np.median(info["move_km"])) if info["move_km"] else None
    info["median_dest_pop"] = float(np.median(info["dest_pop"])) if info["dest_pop"] else None
    return cla, clo, info
# ---------- 検査：移さない版で 087 を差0 ----------
for n in (35, 200):
    S = np.array(MESH[:n]); run_c(f"mesh{n}_B0", plat[S], plon[S], ps, PTb); run_c(f"mesh{n}_CS", plat[S], plon[S], ps, PTs, tgs)
chk = {k: abs(h(k) - r87["heavy_TC"][k]) for k in ("mesh35_B0", "mesh35_CS", "mesh200_B0", "mesh200_CS")}
S = np.array(MESH[:35]); run_c("mesh35_CS_Rd25", plat[S], plon[S], ps, PTs25, tgs, 25.0); chk["mesh35_CS_Rd25"] = abs(h("mesh35_CS_Rd25") - r87["heavy_TC"]["mesh35_CS_Rd25"])
R["check_vs_087"] = chk; assert all(v < 1e-12 for v in chk.values()), chk; print("checks ok", chk, flush=True)
# ---------- 移す ----------
VARS = {"main": dict(), "Rmax50": dict(Rmax=50.0), "Rmax100": dict(Rmax=100.0), "nearest": dict(rule="nearest"), "Rbuf25": dict(Rbuf=25.0, Rd=25.0)}
R["relocation"] = {}
for n in (35, 200):
    for tag, kw in VARS.items():
        if n == 200 and tag not in ("main", "Rmax50"): continue
        cla, clo, info = relocate(MESH[:n], **kw); R["relocation"][f"mesh{n}_{tag}"] = {k: v for k, v in info.items() if k not in ("move_km", "dest_pop")}
        R["relocation"][f"mesh{n}_{tag}"]["move_km_all"] = info["move_km"]
        run_c(f"mesh{n}_MV_{tag}_B0", cla, clo, ps, PTb)
        if tag == "Rbuf25": run_c(f"mesh{n}_MV_{tag}_CS", cla, clo, ps, PTs25, tgs, 25.0)
        else: run_c(f"mesh{n}_MV_{tag}_CS", cla, clo, ps, PTs, tgs)
# 登録外（記述）：同じ移し先で R_d=25 km の損を受けた場合（R_buf 10 km は 25 km の区域では守り切れない）
cla, clo, _ = relocate(MESH[:35]); run_c("mesh35_MV_main_CS_Rd25", cla, clo, ps, PTs25, tgs, 25.0)
# ---------- まとめ ----------
NET = {}
for k in R["relocation"]:
    n, tag = k.split("_")
    base_cs = h(f"{n}_CS_Rd25") if tag == "Rbuf25" else h(f"{n}_CS")
    NET[k] = {"CS": base_cs, "MV_CS": h(f"{k.replace('_', '_MV_', 1)}_CS"), "net": h(f"{k.replace('_', '_MV_', 1)}_CS") - base_cs,
              "nonuke_cost": h(f"{k.replace('_', '_MV_', 1)}_B0") - h(f"{n}_B0"), "loss_087": h(f"{n}_B0") - base_cs}
    NET[k]["recovered_share"] = NET[k]["net"] / NET[k]["loss_087"]
NET["mesh35_main_underRd25"] = {"net_vs_CS_Rd25": h("mesh35_MV_main_CS_Rd25") - h("mesh35_CS_Rd25")}
R["net"] = NET; R["heavy_TC"] = {k: h(k) for k in RT if k.startswith("mesh")}
g200 = h("mesh200_MV_main_CS") - h("mesh35_MV_main_CS")
R["gain_35_200"] = {"CS_087": h("mesh200_CS") - h("mesh35_CS"), "MV_main": g200}
Hh = {}
v = NET["mesh35_main"]["net"]; Hh["H100a"] = {"class": "予測", "value": v, "rule": "main net >= +0.005", "supported": v >= 0.005}
v = NET["mesh35_main"]["nonuke_cost"]; Hh["H100b"] = {"class": "予測", "value": v, "rule": "|MV_B0 - B0| <= 0.002", "supported": abs(v) <= 0.002}
v = NET["mesh35_Rmax50"]["net"]; Hh["H100c"] = {"class": "予測", "value": v, "rule": "Rmax50 net >= +0.003", "supported": v >= 0.003}
Hh["H100d"] = {"class": "予測", "value": g200, "rule": "MV 35->200 >= +0.040 (087 CS +0.0404)", "supported": g200 >= R["gain_35_200"]["CS_087"]}
v = NET["mesh35_nearest"]["net"] - NET["mesh35_main"]["net"]; Hh["H100e"] = {"class": "予測", "value": v, "rule": "|nearest - main| <= 0.002", "supported": abs(v) <= 0.002}
for x in Hh.values(): x["supported"] = bool(x["supported"])
R["hypotheses"] = Hh
# ---------- 図 ----------
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
ks = [k for k in NET if k.startswith("mesh35_") and "under" not in k]; x = np.arange(len(ks))
axs[0].bar(x - .2, [NET[k]["loss_087"] for k in ks], .4, label="087 loss (B0 - CS)"); axs[0].bar(x + .2, [NET[k]["net"] for k in ks], .4, label="net gain by relocation")
axs[0].plot(x, [NET[k]["nonuke_cost"] for k in ks], "kx", label="no-nuke change (MV_B0 - B0)"); axs[0].axhline(0, color="k", lw=.6)
axs[0].set_xticks(x); axs[0].set_xticklabels([k.replace("mesh35_", "") for k in ks]); axs[0].set_ylabel("heavy TC/V0 difference (not survival prob.)"); axs[0].legend(fontsize=7); axs[0].set_title("mesh35")
mk = R["relocation"]["mesh35_main"]["move_km_all"]; axs[1].hist(mk, bins=15); axs[1].set_xlabel("move distance (km), mesh35 main"); axs[1].set_ylabel("copies")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_relocation.png"), dpi=120); plt.close()
R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("check_vs_087", "net", "gain_35_200", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print(json.dumps({k: {a: b for a, b in v.items() if a != "move_km_all"} for k, v in R["relocation"].items()}, ensure_ascii=False))
print("runtime", R["runtime_s"])
