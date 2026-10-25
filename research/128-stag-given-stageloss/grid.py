"""研究128 の格子：p|lost ごとの F_heavy（127 の CVER と同じ乱数・仕組み）。計算のみ。
注：F_heavy・V は生存確率や復興確率ではない。127 の sim.py を「# 126 の関数は」の前まで exec し、127 の table と差 0 を assert。"""
import json, time, os
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
P127 = os.path.join(RES, "127-stag-corr-horizon", "sim.py")
src = open(P127, encoding="utf8").read(); body = src[:src.index("\nSL = \"slow_Tasmania\"")]
G = {"__file__": P127, "__name__": "sim127"}
exec(compile(body, P127, "exec"), G)
R127 = json.load(open(os.path.join(RES, "127-stag-corr-horizon", "results.json"), encoding="utf8"))
stag_of, lost, share, COMPS, N = (G[k] for k in ("stag_of", "lost", "share", "COMPS", "N"))
base_val_k, run_k, KAPS, LK, r_pop, Vg, P0 = (G[k] for k in ("base_val_k", "run_k", "KAPS", "LK", "r_pop", "Vg", "P0"))
assert float(Vg["H"]) == 1000.0
SL = "slow_Tasmania"; MAIN3 = ("D2rz|none", "D1|main", "D4|main"); OTH = ("trigger_mix", "fast_state")
pm = min(1.0, P0 / share["U3"])
PS = sorted(set([0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 1.0, pm]))
def S_of(p):  # 成分ごとの Cmass と同じ形（成分 k の share で上限 1）。U3 以外も 127 と揃えるため p は U3 の値として成分ごとに換算しない
    return stag_of(lambda k: np.where(lost[k], p, 0.0))
GRID = {}
for p in PS:
    S = S_of(p) if abs(p - pm) > 1e-12 else G["CVER"]["Cmass"]
    vb = base_val_k(S, r_pop)
    for vn in MAIN3: GRID[f"{p:.6f}|{SL}|{vn}"] = run_k(KAPS[SL], LK[vn], S, r_pop, False, vb)
    for kn in OTH: GRID[f"{p:.6f}|{kn}|D2rz|none"] = run_k(KAPS[kn], LK["D2rz|none"], S, r_pop, False, vb)
    print("p", round(p, 4), "F", round(GRID[f'{p:.6f}|{SL}|D2rz|none']['F_heavy'], 4), round(time.time() - t0), "s", flush=True)
V0 = {}
vb0 = base_val_k(G["CVER"]["V0"], r_pop)
V0["new_heavy"] = run_k(KAPS[SL], LK["D2rz|none"], G["CVER"]["V0"], r_pop, False, vb0)["new_heavy"]
chk = {}
for p, v in ((0.6, "Ccond"), (pm, "Cmass"), (1.0, "C1")):
    for vn in MAIN3:
        a = GRID[f"{p:.6f}|{SL}|{vn}"]["F_heavy"]; b = R127["table"][f"H1000|{v}|W|{SL}|{vn}"]["F_heavy"]
        chk[f"{v}|{vn}"] = abs(a - b)
mx = max(chk.values()); print("check vs 127 maxdiff", mx); assert mx < 1e-12, chk
json.dump({"PS": PS, "p_mass": pm, "share_lost": share, "GRID": GRID, "V0_new_heavy": V0["new_heavy"], "check_127": chk,
           "runtime_s": time.time() - t0}, open(os.path.join(HERE, "grid.json"), "w"), indent=1, default=float)
print("runtime", round(time.time() - t0))
