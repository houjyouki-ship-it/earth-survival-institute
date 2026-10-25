"""研究130：段を保った社会（kept）の停滞を模型に入れる（open_problems 305）。計算のみ、ネット接続なし。
事前登録：PREREGISTRATION.md（2026-10-26 01:36 JST）。注：F_heavy・V は生存確率や復興確率ではない。
127 の sim.py を「SL = "slow_Tasmania"」の前まで exec（126 の assert を通る）し、p_kept=0 で 127 の Ccond・C1 と差 0 を assert。"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
P127 = os.path.join(RES, "127-stag-corr-horizon", "sim.py")
src = open(P127, encoding="utf8").read(); body = src[:src.index("\nSL = \"slow_Tasmania\"")]
G = {"__file__": P127, "__name__": "sim127"}
exec(compile(body, P127, "exec"), G)
R127 = json.load(open(os.path.join(RES, "127-stag-corr-horizon", "results.json"), encoding="utf8"))
stag_of, lost, share, COMPS, N = (G[k] for k in ("stag_of", "lost", "share", "COMPS", "N"))
base_val_k, run_k, KAPS, LK, r_pop, Vg = (G[k] for k in ("base_val_k", "run_k", "KAPS", "LK", "r_pop", "Vg"))
assert float(Vg["H"]) == 1000.0
SL = "slow_Tasmania"; MAIN3 = ("D2rz|none", "D1|main", "D4|main")
PL = (0.6, 0.9, 1.0); PK = (0.0, 0.1, 1/6, 0.25, 0.375, 0.5)
GRID = {}
for pl in PL:
    for pk in PK:
        S = stag_of(lambda k: np.where(lost[k], pl, pk))
        vb = base_val_k(S, r_pop)
        for vn in MAIN3: GRID[f"{pl:.4f}|{pk:.6f}|{vn}"] = run_k(KAPS[SL], LK[vn], S, r_pop, False, vb)
        print("pl", pl, "pk", round(pk, 4), "F", round(GRID[f'{pl:.4f}|{pk:.6f}|D2rz|none']['F_heavy'], 4), round(time.time() - t0), "s", flush=True)
chk = {}
for pl, v in ((0.6, "Ccond"), (1.0, "C1")):
    for vn in MAIN3:
        chk[f"{v}|{vn}"] = abs(GRID[f"{pl:.4f}|{0.0:.6f}|{vn}"]["F_heavy"] - R127["table"][f"H1000|{v}|W|{SL}|{vn}"]["F_heavy"])
mx = max(chk.values()); print("check vs 127 maxdiff", mx); assert mx < 1e-12, chk
F = lambda pl, pk, vn="D2rz|none", key="F_heavy": GRID[f"{pl:.4f}|{pk:.6f}|{vn}"][key]
# 事後（p_lost=0.9 固定、p_kept を Jeffreys の Beta で、格子を線形補間）
g = np.random.default_rng(20261026)
pkx = np.array(PK); fy = np.array([F(0.9, p) for p in PK])
POST = {}
for name, (a, b) in {"main_1of3": (1.5, 2.5), "ext_1of8": (1.5, 7.5)}.items():
    d = g.beta(a, b, 200000); inside = float((d <= 0.5).mean())
    fv = np.interp(np.minimum(d, 0.5), pkx, fy)  # 0.5 を超える分は 0.5 で打ち切り（外挿しない）
    POST[name] = {"pk_mean": float(d.mean()), "share_le_0.5": inside, "F_mean": float(fv.mean()),
                  "F_q10": float(np.quantile(fv, 0.1)), "F_q90": float(np.quantile(fv, 0.9))}
HYP = {}
f0, f375 = F(0.9, 0.0), F(0.9, 0.375)
HYP["H142a"] = {"class": "confirmatory", "value": {"F_pk0": f0, "F_pk0375": f375, "rel": f375 / f0 - 1}, "rule": "|rel| >= 0.03", "supported": bool(abs(f375 / f0 - 1) >= 0.03)}
rk = {f"{pl}|{pk:.4f}": [F(pl, pk, vn) for vn in MAIN3] for pl in PL for pk in PK}
HYP["H142b"] = {"class": "confirmatory", "value": rk, "rule": "D2rz>=D1>=D4 全格子", "supported": all(a >= b >= c for a, b, c in rk.values())}
dif = np.diff(fy)
HYP["H142c"] = {"class": "prediction", "value": fy.tolist(), "rule": "単調", "supported": bool((dif >= 0).all() or (dif <= 0).all())}
HYP["H142d"] = {"class": "prediction", "value": {"F_pk0": f0, "F_pk0375": f375}, "rule": "F(0.375) < F(0)", "supported": bool(f375 < f0)}
AMT = {f"{pl}|{pk:.4f}": {"F": F(pl, pk), "base": F(pl, pk, key="base_heavy"), "new": F(pl, pk, key="new_heavy")} for pl in PL for pk in PK}
OUT = {"study": "130-kept-stagnation", "note": "F_heavy・V は生存確率や復興確率ではない", "share_lost": share,
       "PL": PL, "PK": PK, "GRID": GRID, "check_127": chk, "posterior_pl0.9": POST, "hypotheses": HYP, "amounts_D2rz": AMT,
       "runtime_s": time.time() - t0}
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1, default=float)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for pl in PL: ax[0].plot(PK, [F(pl, p) for p in PK], "o-", label=f"p|lost={pl}")
for pl in PL: ax[1].plot(PK, [F(pl, p, key="base_heavy") / F(pl, 0.0, key="base_heavy") - 1 for p in PK], "o--", label=f"base p|lost={pl}")
for pl in PL: ax[1].plot(PK, [F(pl, p, key="new_heavy") / F(pl, 0.0, key="new_heavy") - 1 for p in PK], "s-", label=f"new p|lost={pl}")
ax[0].set_title("130: F_heavy vs p|kept (slow kappa, D2rz, W, H1000)", fontsize=9); ax[0].set_xlabel("p|kept")
ax[1].set_title("130: heavy value rel. to p|kept=0 (not a survival prob.)", fontsize=9); ax[1].set_xlabel("p|kept")
for a in ax: a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_pkept.png"), dpi=120)
for k, o in AMT.items(): print(k, {a: round(b, 5) for a, b in o.items()})
print(json.dumps(POST, default=float))
for h, v in HYP.items(): print(h, v["class"], v["supported"], json.dumps(v["value"], default=float)[:240])
print("runtime", round(time.time() - t0), "s")
