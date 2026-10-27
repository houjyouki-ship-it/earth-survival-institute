"""研究148：媒体の読める確率の地平依存（open_problems 340）。
147 の割り当て直した AI・GEN の段の時刻のまま、141 の run() の地平 H を 1,000〜10,000 年に替える。
計算のみ・ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-27 13:34 JST、計算の前）。
注：P_read は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
src = open(f"{RES}/147-aigen-reassign/sim.py", encoding="utf8").read()
src = src[:src.index("MED = {}")].replace("/tmp/rep147", "/tmp/rep148")
NS = {"__file__": f"{RES}/147-aigen-reassign/sim.py", "__name__": "m148"}
exec(compile(src, "sim147_head", "exec"), NS)
R147 = json.load(open(f"{RES}/147-aigen-reassign/results.json", encoding="utf8"))
NS141, run141, TT, NEW, tt_new, V141, KEYS = NS["NS"], NS["run141"], NS["TT"], NS["NEW"], NS["tt_new"], NS["V141"], NS["KEYS"]
TT0 = dict(TT)
HS = (1000.0, 2000.0, 5000.0, 10000.0); VN = ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW")
MED = {}; CHECK = {}
for vn in VN:
    for s in ("AI", "GEN"):
        sn, sv = NEW[(vn, s)]; TT[s] = tt_new(s, sn, sv)
    for Hh in HS:
        NS141["H"] = Hh
        for v141, var in V141.items():
            o, _ = run141(var)
            amt = {k: o[k]["amount_x_P"] for k in o}
            MED[f"{int(Hh)}|{vn}|{v141}"] = {"P_read_H": {k: o[k]["P_read_H"] for k in KEYS},
                                              "amount_x_P": {k: amt[k] for k in KEYS}, "argmax_amount": max(amt, key=amt.get)}
            if Hh == 1000.0:
                ref = R147["media"][f"{vn}|{v141}"]
                for k in KEYS:
                    d = abs(o[k]["P_read_H"] - ref["P_read_H"][k]); CHECK[f"147_{vn}_{v141}_{k}"] = d; assert d == 0.0, (vn, v141, k, d)
                assert MED[f"1000|{vn}|{v141}"]["argmax_amount"] == ref["argmax_amount"]
    TT.update(TT0)
NS141["H"] = 1000.0
# 段の到達時刻の分布（P_MIX、A_XPT_S の重み付きでなく標本の単純な分位：AI・GEN）
for s in ("AI", "GEN"):
    sn, sv = NEW[("P_MIX", s)]; TT[s] = tt_new(s, sn, sv)
TIERQ = {s: {"t1_q": [float(np.quantile(TT[s][0], q)) for q in (0.25, 0.5, 0.75, 0.9)],
             "t2_q": [float(np.quantile(TT[s][1], q)) for q in (0.25, 0.5, 0.75, 0.9)],
             "share_t2_gt_1000": float(np.mean(TT[s][1] > 1000)), "share_t2_gt_5000": float(np.mean(TT[s][1] > 5000))} for s in ("GEN", "AI", "BIO", "CASC")}
TT.update(TT0)
# ---- 仮説 ----
m = lambda H, vn="P_MIX", v="main": MED[f"{H}|{vn}|{v}"]["P_read_H"]
ratio = {H: m(H)["rosetta"] / m(H)["fiche24"] for H in (1000, 2000, 5000, 10000)}
allk = [(int(H), vn, v) for H in HS for vn in VN for v in V141]
b_ok = all(MED[f"{H}|{vn}|{v}"]["argmax_amount"] == "silica" for H, vn, v in allk)
e_fail = [f"{H}|{vn}|{v}" for H, vn, v in allk if not (m(H, vn, v)["rosetta"] >= max(m(H, vn, v)["fiche48"], m(H, vn, v)["piql_hand"]))]
HY = {"H160a": {"class": "prediction", "value": ratio[5000], "ratio_by_H": ratio, "supported": ratio[5000] >= 3.0},
      "H160b": {"class": "confirmatory", "argmax": {f"{H}|{vn}|{v}": MED[f"{H}|{vn}|{v}"]["argmax_amount"] for H, vn, v in allk}, "supported": b_ok},
      "H160c": {"class": "prediction", "value": m(5000)["silica"], "supported": m(5000)["silica"] >= 0.80},
      "H160d": {"class": "prediction", "value": m(5000)["fiche24"] - m(1000)["fiche24"], "supported": m(5000)["fiche24"] - m(1000)["fiche24"] < 0.05},
      "H160e": {"class": "confirmatory", "failures": e_fail, "supported": len(e_fail) == 0}}
LONG = ("mom_mf", "rosetta", "silica"); SHORT = ("fiche24", "fiche48", "piql")
gap = {f"{int(H)}|{v}": {"long_mean": float(np.mean([m(int(H), "P_MIX", v)[k] for k in LONG])), "short_mean": float(np.mean([m(int(H), "P_MIX", v)[k] for k in SHORT]))} for H in HS for v in V141}
RESULT = {"study": "148-horizon-media", "note": "P_read は生存確率や復興確率ではない（模型の中の量）", "check_max_diff": max(CHECK.values()), "n_checks": len(CHECK),
          "horizons": HS, "media": MED, "tier_quantiles_P_MIX": TIERQ, "long_vs_short_P_MIX": gap, "hypotheses": HY, "runtime_s": time.time() - T0}
json.dump(RESULT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for k in KEYS:
    ax[0].plot(HS, [m(int(H))[k] for H in HS], marker="o", label=k)
ax[0].set_xscale("log"); ax[0].set_xlabel("horizon H (years)"); ax[0].set_ylabel("P_read within H (heavy, A_XPT_S, P_MIX)"); ax[0].legend(fontsize=7, ncol=2)
for v, ls in zip(V141, ("-", "--", ":")):
    ax[1].plot(HS, [m(int(H), "P_MIX", v)["rosetta"] / max(m(int(H), "P_MIX", v)["fiche24"], 1e-9) for H in HS], ls, marker="o", label=f"rosetta/fiche24 {v}")
ax[1].set_xscale("log"); ax[1].set_xlabel("horizon H (years)"); ax[1].set_ylabel("ratio"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_horizon.png", dpi=110)
for v in V141:
    for H in HS:
        print(int(H), v, {k: round(x, 3) for k, x in m(int(H), "P_MIX", v).items()}, MED[f"{int(H)}|P_MIX|{v}"]["argmax_amount"])
print("tier q", json.dumps(TIERQ)); print({k: (v["supported"], v.get("value")) for k, v in HY.items()}); print("ratio", ratio, "e_fail", e_fail[:10])
print("checks", len(CHECK), "max diff", max(CHECK.values()), "runtime", round(time.time() - T0, 1))
