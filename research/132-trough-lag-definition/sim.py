"""研究132：停滞の定義を「谷からの遅れ」に揃えて phi と stag を同じ物差しにしたときの F_heavy（open_problems 307）。
注：F_heavy・V は生存確率や復興確率ではない。計算のみ、ネット接続なし。事前登録 PREREGISTRATION.md（05:40 JST）。
再現：定義A・phi=0・0.5・1 で 131 の results.json と差 0（assert）。"""
import json, time, os, math
import numpy as np
t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
P127 = os.path.join(RES, "127-stag-corr-horizon", "sim.py")
src = open(P127, encoding="utf8").read(); body = src[:src.index("\nSL = \"slow_Tasmania\"")]
G = {"__file__": P127, "__name__": "sim127"}
exec(compile(body, P127, "exec"), G)
print("prefix", round(time.time() - t0), "s", flush=True)
R131 = json.load(open(os.path.join(RES, "131-phi-same-collapse", "results.json"), encoding="utf8"))["OUT"]
CJ = json.load(open(os.path.join(RES, "131-phi-same-collapse", "cases.json"), encoding="utf8"))
EX = json.load(open(os.path.join(RES, "131-phi-same-collapse", "explore_posttrough.json"), encoding="utf8"))
stag_of, lost, KAPS, LK, r_pop, U_pick = (G[k] for k in ("stag_of", "lost", "KAPS", "LK", "r_pop", "U_pick"))
T2o = G["T2_stag"]; FIXED = G["FIXED"]; COMPS = G["COMPS"]
r_med = CJ["R_MED"]
# 定義A（125）の値と定義B（谷からの遅れ）の値。順序は 125 の stag_years_no_immig と同じ（Greece, Britain, Greenland）
VA = np.asarray(G["V2"]["stag_years_no_immig"], float)
lagB = [(-700 - (-1000)) - math.log(6e5 / 3e5) / r_med, EX["lag_years"]["Britain"], 650.0]
VB = np.asarray(lagB, float)
assert abs(VA[0] - 200 - VB[0]) < 1e-6 and abs(VA[1] - 200 - VB[1]) < 1e-6   # 差は下り坂 200 年ずつ
idx = np.minimum((U_pick * len(VA)).astype(int), len(VA) - 1)
assert np.array_equal(G["pick"], VA[idx])
phiB = EX["phi"]; phi_vals = sorted(phiB.values()); PHI_B = {"median": phi_vals[1], "mean": float(np.mean(phi_vals)), "Italy": phiB["Italy"]}
CUR = {"phi": 0.0, "X": None}
def T2x(k, st, stag, r):
    t = T2o(k, st, stag, r)
    if k in FIXED: return t
    xs = stag if CUR["X"] is None else CUR["X"][k]
    return t + np.where((st == 2) | (st == 3), CUR["phi"] * xs, 0.0)
G["T2_stag"] = T2x
SL = "slow_Tasmania"; MAIN3 = ("D2rz|none", "D1|main", "D4|main"); PK = 1/6
pf = lambda k: np.where(lost[k], 0.9, PK)
def make_S(vals):
    G["pick"] = vals[idx]; return stag_of(pf)
SA = make_S(VA); SB = make_S(VB); G["pick"] = VA[idx]
def run(S, phi, X=None):
    CUR["phi"] = 0.0; CUR["X"] = None; vb = G["base_val_k"](S, r_pop)
    CUR["phi"] = phi; CUR["X"] = X
    o = {vn: G["run_k"](KAPS[SL], LK[vn], S, r_pop, False, vb) for vn in MAIN3}
    CUR["phi"] = 0.0; CUR["X"] = None; return o
OUT = {}; chk = {}
for phi in (0.0, 0.5, 1.0):
    o = run(SA, phi); OUT[f"A|{phi:.6f}"] = o
    for vn in MAIN3:
        d = abs(o[vn]["F_heavy"] - R131[f"{phi:.6f}|{vn}"]["F_heavy"]); chk[f"A|{phi}|{vn}"] = d; assert d < 1e-12, (phi, vn, d)
print("check vs 131 ok", flush=True)
gridB = sorted(set([0.0, 0.25, 0.5, 0.75, 1.0] + list(PHI_B.values())))
for phi in gridB:
    OUT[f"B|{phi:.6f}"] = run(SB, phi)
    OUT[f"MIX|{phi:.6f}"] = run(SA, phi, SB)
    print(round(phi, 4), {v: (round(OUT[f'B|{phi:.6f}'][vn]['F_heavy'], 4), round(OUT[f'MIX|{phi:.6f}'][vn]['F_heavy'], 4)) for v, vn in zip(("D2", "D1", "D4"), MAIN3)}, flush=True)
FH = lambda key: {vn.split("|")[0]: OUT[key][vn]["F_heavy"] for vn in MAIN3}
FT = lambda key: {vn.split("|")[0]: OUT[key][vn].get("F_total") for vn in MAIN3}
summ = {"A_phi0": FH("A|0.000000"), "B_phi0": FH("B|0.000000")}
for n, p in PHI_B.items(): summ[f"B_{n}"] = FH(f"B|{p:.6f}"); summ[f"MIX_{n}"] = FH(f"MIX|{p:.6f}")
order_ok = all(OUT[k]["D2rz|none"]["F_heavy"] >= OUT[k]["D1|main"]["F_heavy"] >= OUT[k]["D4|main"]["F_heavy"] for k in OUT)
a = summ["A_phi0"]["D2rz"]; b0 = summ["B_phi0"]["D2rz"]; bI = summ["B_Italy"]["D2rz"]
H = {"H144a": {"B_phi0": b0, "change": b0 / a - 1, "supported": b0 < 2.065},
     "H144b": {"B_Italy": bI, "supported": bI >= 1.6},
     "H144c": {"supported": order_ok},
     "H144d": {"width_B": b0 - bI, "width_131": 2.174 - 1.270, "supported": (b0 - bI) <= 0.5 * (2.174 - 1.270)}}
json.dump({"note": "F_heavy・V は生存確率や復興確率ではない", "VA": VA.tolist(), "VB": VB.tolist(), "PHI_B": PHI_B, "gridB": gridB,
           "OUT": OUT, "summary": summ, "hyp": H, "check_vs_131": chk, "order_ok_all": order_ok, "runtime_s": time.time() - t0},
          open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 4.2))
for vn, c in zip(MAIN3, ("C0", "C1", "C2")):
    ax.plot(gridB, [OUT[f"B|{p:.6f}"][vn]["F_heavy"] for p in gridB], "o-", color=c, label=vn.split("|")[0] + " def B (trough lag)")
    ax.plot(gridB, [OUT[f"MIX|{p:.6f}"][vn]["F_heavy"] for p in gridB], "x--", color=c, alpha=0.6, label=vn.split("|")[0] + " mix")
    ax.plot([0, 0.5, 1], [OUT[f"A|{p:.6f}"][vn]["F_heavy"] for p in (0.0, 0.5, 1.0)], "s:", color=c, alpha=0.4)
for n, p in PHI_B.items(): ax.axvline(p, color="gray", ls=":", lw=0.8)
ax.axhline(1, color="k", lw=0.6); ax.set_xlabel("phi"); ax.set_ylabel("F_heavy (slow kappa, p|lost 0.9)")
ax.set_title("132: F_heavy with trough-lag stagnation (not a survival probability)", fontsize=9); ax.legend(fontsize=6); fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_trough_lag.png"), dpi=120)
print(json.dumps({"summary": summ, "hyp": H}, ensure_ascii=False, default=float, indent=1)); print("runtime", round(time.time() - t0))
