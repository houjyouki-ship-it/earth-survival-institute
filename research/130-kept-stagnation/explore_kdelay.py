"""研究130 の登録外の探索（事前登録なし、支持に数えない）。sim.py で kept の停滞の効果が構造的に 0 と分かったため、
kept（st=2,3）の T2 に phi×停滞年数を足す上限側の版を作る。注：F_heavy・V は生存確率や復興確率ではない。"""
import json, time, os
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
P127 = os.path.join(RES, "127-stag-corr-horizon", "sim.py")
src = open(P127, encoding="utf8").read(); body = src[:src.index("\nSL = \"slow_Tasmania\"")]
G = {"__file__": P127, "__name__": "sim127"}
exec(compile(body, P127, "exec"), G)
R130 = json.load(open(os.path.join(HERE, "results.json"), encoding="utf8"))
stag_of, lost, KAPS, LK, r_pop = (G[k] for k in ("stag_of", "lost", "KAPS", "LK", "r_pop"))
T2o = G["T2_stag"]; FIXED = G["FIXED"]; B = G["B"]
PHI = [0.0]
def T2x(k, st, stag, r):
    t = T2o(k, st, stag, r)
    if k in FIXED: return t
    return t + np.where((st == 2) | (st == 3), PHI[0] * stag, 0.0)
G["T2_stag"] = T2x
SL = "slow_Tasmania"; MAIN3 = ("D2rz|none", "D1|main", "D4|main"); OUT = {}
for phi in (0.0, 0.5, 1.0):
    PHI[0] = phi
    for pk in ((0.0,) if phi == 0 else (1/6, 0.375)):
        S = stag_of(lambda k: np.where(lost[k], 0.9, pk)); vb = G["base_val_k"](S, r_pop)
        for vn in MAIN3: OUT[f"{phi}|{pk:.4f}|{vn}"] = G["run_k"](KAPS[SL], LK[vn], S, r_pop, False, vb)
        print(phi, round(pk, 4), {vn[:4]: round(OUT[f'{phi}|{pk:.4f}|{vn}']['F_heavy'], 4) for vn in MAIN3},
              "base", round(OUT[f'{phi}|{pk:.4f}|D2rz|none']['base_heavy'], 5), "new", round(OUT[f'{phi}|{pk:.4f}|D2rz|none']['new_heavy'], 5), flush=True)
d = abs(OUT["0.0|0.0000|D2rz|none"]["F_heavy"] - R130["GRID"]["0.9000|0.000000|D2rz|none"]["F_heavy"]); assert d < 1e-12, d
json.dump({"note": "登録外の探索。F_heavy・V は生存確率や復興確率ではない", "OUT": OUT, "check_vs_sim": d, "runtime_s": time.time() - t0},
          open(os.path.join(HERE, "explore_kdelay.json"), "w"), indent=1, default=float)
print("runtime", round(time.time() - t0))
