"""研究142：停滞の模型（125・132・133）の値で 141 の媒体の P_read を出し直す（open_problems 326）。
注：P_read・量×読める確率は模型の中の量で、生存確率でも復興確率でもない。事前登録 PREREGISTRATION.md（01:45 JST）。
仕組み：141 の sim.py を写し（/tmp/rep142）で実行して段の時刻 TT・媒体表を得る。141 の run() と同じ乱数の消費順で、
停滞の遅れだけを差し替える（共通乱数）。遅れの値は 132 の VA/VB と 125 の decline_years（133 と同じ）。"""
import json, os, shutil, runpy, time, platform
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
W = "/tmp/rep142/research"
for d in ("141-medium-tier-readability", "038-restart-state-evidence"):
    os.makedirs(f"{W}/{d}", exist_ok=True)
shutil.copy(f"{RES}/141-medium-tier-readability/sim.py", f"{W}/141-medium-tier-readability/sim.py")
shutil.copy(f"{RES}/038-restart-state-evidence/results.json", f"{W}/038-restart-state-evidence/results.json")
assert os.path.exists("/tmp/rep141/038-restart-state-evidence/sim.py")
G = runpy.run_path(f"{W}/141-medium-tier-readability/sim.py", run_name="m141")
R141 = json.load(open(f"{RES}/141-medium-tier-readability/results.json"))
Rcopy = json.load(open(f"{W}/141-medium-tier-readability/results.json"))
chk = {}
for vn in R141["variants"]:
    for k, o in R141["variants"][vn]["media"].items():
        chk[f"{vn}|{k}"] = abs(o["P_read_H"] - Rcopy["variants"][vn]["media"][k]["P_read_H"])
assert max(chk.values()) == 0.0
ns, TT, MEDIA, HSC, H, t_build = G["ns"], G["TT"], G["MEDIA"], G["HSC"], G["H"], G["t_build"]
# 停滞の入力（既存の results.json から）
R132 = json.load(open(f"{RES}/132-trough-lag-definition/results.json")); R125 = json.load(open(f"{RES}/125-pop-recovery-history/results.json"))
VA = np.asarray(R132["VA"], float); VB = np.asarray(R132["VB"], float)
dec = {c["key"]: c["decline_years"] for c in R125["cases"] + R125["cases_supplementary"]}
DC = [dec["Greece_LBA"], dec["Britain_postRoman"], dec["Easter_Island"]]; DS = [dec["England_BlackDeath"], dec["Hawaii_contact"], dec["LatinAmerica_conquest"]]
DCc = DC + [dec["Copan_valley"]]
assert DC == [200, 200, 155] and DS == [102, 94, 150] and DCc[-1] == 425
pick = lambda vals, u: np.asarray(vals, float)[np.minimum((u * len(vals)).astype(int), len(vals) - 1)]

def delay(spec, n, s):
    """141 の stag を差し替える遅れ。spec=None なら 141 の logU 版（再現用）。"""
    rd = np.random.default_rng(20261142 + HSC.index(s)); u_s, u1, u2, u3 = (rd.uniform(size=n) for _ in range(4))
    p = spec.get("p", 0.9); st = u_s < p
    if spec["kind"] == "A": return np.where(st, pick(spec.get("VA", VA), u1), 0.0)
    if spec["kind"] in ("E1", "E2", "E3"):
        d = np.where(st, pick(VB, u1) + pick(DCc if spec["kind"] == "E3" else DC, u2), 0.0)
        if spec["kind"] == "E2": d = d + np.where(~st, pick(DS, u3), 0.0)
        return d
    raise ValueError(spec)

def run(spec, anchor="A_XPT_S"):
    out = {k[0]: {"P_read_H": 0.0, "E_frac_H": 0.0, "P_K01": 0.0, "P_K12": 0.0, "P_K2e": 0.0} for k in MEDIA}
    wsum = 0.0; dl = []
    for s in HSC:
        t1, t2, hs = TT[s]; n = len(t1)
        if n == 0: continue
        w = ns["WT"][anchor]["mean_weights"][s] * hs; wsum += w
        rs = np.random.default_rng(20261141 + HSC.index(s))
        L = lambda lo, hi: np.exp(rs.uniform(np.log(lo), np.log(hi), n))
        crude = np.where(rs.uniform(size=n) < (0.9 if spec and spec["kind"] == "crude" else 0.0), L(300, 900), 0.0)  # 141 と同じ消費
        stag = crude if (spec is None or spec["kind"] == "crude") else delay(spec, n, s)
        dl.append((w, stag))
        t1s, t2s = t1 + stag, t2 + stag; tb = t_build[:n]
        m23 = 300.0 * np.exp(0.8 * rs.standard_normal(n))
        salv = rs.uniform(size=n) < 0.0
        t_salv = L(1, 20); life_salv = np.where(rs.uniform(size=n) < 0.5, L(500, 5000), L(30, 300))
        R1 = np.where(salv, np.minimum(t_salv, t1s), t1s)
        T = {"R0": np.zeros(n), "R1": R1, "R2": t2s, "R2e": t2s + tb, "R3": t2s + m23}; T["R3d"] = T["R3"]
        qM = L(0.03, 0.43)
        for (k, nm, tier, lo, hi, unit, cap, src) in MEDIA:
            life = L(lo, hi) * L(0.3, 1.0) / np.log(2) ** 0.5 * (-np.log(rs.uniform(size=n))) ** 0.5
            ta = T[tier]; ok = (ta <= H) & (life > ta); frac = np.ones(n)
            if k == "piql":
                okR3 = (T["R3"] <= H) & (life > T["R3"]); frac = np.where(okR3, 1.0, np.where(ok, qM, 0.0)); ok = ok | okR3
            o = out[k]
            o["P_read_H"] += w * np.mean(ok); o["E_frac_H"] += w * np.mean(ok * frac)
            o["P_K01"] += w * np.mean((ta < t1s) & (life > ta)); o["P_K12"] += w * np.mean((ta < t2s) & (life > np.maximum(ta, 0)))
            o["P_K2e"] += w * np.mean((ta < t2s + tb) & (life > ta))
    for k in out:
        for kk in list(out[k]): out[k][kk] /= wsum
        out[k]["amount_x_P"] = [m[6] for m in MEDIA if m[0] == k][0] * out[k]["E_frac_H"]
    ws = np.array([w for w, _ in dl]); allv = np.concatenate([d for _, d in dl])
    wts = np.concatenate([np.full(len(d), w / len(d)) for w, d in dl])
    mean_delay = float(np.sum(wts * allv) / wts.sum()); p_any = float(np.sum(wts * (allv > 0)) / wts.sum())
    return out, {"mean_delay": mean_delay, "p_delay": p_any}

SPECS = {"none": None, "crude141": {"kind": "crude"}, "A": {"kind": "A"}, "E1": {"kind": "E1"}, "E2": {"kind": "E2"},
         "E3": {"kind": "E3"}, "A_pl05": {"kind": "A", "p": 0.5}, "A_noGL": {"kind": "A", "VA": VA[:2]}}
RES2 = {}
for nm, sp in SPECS.items():
    o, d = run(sp); RES2[nm] = {"media": o, "delay": d}
# 再現：none と crude141 は 141 の main・stag_0.9 と差 0
for nm, vn in (("none", "main"), ("crude141", "stag_0.9")):
    for k in RES2[nm]["media"]:
        dd = abs(RES2[nm]["media"][k]["P_read_H"] - R141["variants"][vn]["media"][k]["P_read_H"]); chk[f"rep_{nm}|{k}"] = dd
        assert dd < 1e-12, (nm, k, dd)
mA = RES2["A"]["media"]; m0 = RES2["none"]["media"]; mc = RES2["crude141"]["media"]
best_amt = {nm: max(r["media"], key=lambda k: r["media"][k]["amount_x_P"]) for nm, r in RES2.items()}
mid = ("fiche24", "fiche48", "piql", "silica")
K2e_best500 = max(mA[k]["P_K2e"] for k in ("fiche24", "fiche48", "piql_hand"))
HYP = {
 "H154a": {"mom_mf": mA["mom_mf"]["P_read_H"], "fiche24": mA["fiche24"]["P_read_H"],
           "supported": bool(mA["mom_mf"]["P_read_H"] >= 0.9 and mA["fiche24"]["P_read_H"] < 0.5)},
 "H154b": {"argmax": best_amt, "supported": all(v == "silica" for nm, v in best_amt.items() if nm not in ("none", "crude141"))},
 "H154c": {k: [mc[k]["P_read_H"], mA[k]["P_read_H"], m0[k]["P_read_H"]] for k in mid},
 "H154d": {"mom_mf_K2e": mA["mom_mf"]["P_K2e"], "best500_K2e": K2e_best500, "ratio": mA["mom_mf"]["P_K2e"] / max(K2e_best500, 1e-12)},
 "H154e": {"piql": mA["piql"]["P_read_H"], "supported": bool(mA["piql"]["P_read_H"] >= 0.5)},
}
HYP["H154c"]["supported"] = all(mc[k]["P_read_H"] <= mA[k]["P_read_H"] <= m0[k]["P_read_H"] for k in mid)
HYP["H154d"]["supported"] = bool(HYP["H154d"]["ratio"] >= 1.5)
# 登録外（記述）：寿命の下端が何年なら停滞版 A で P_read が 0.9 に届くか（fiche24 の寿命を倍にした走査）
json.dump({"study": "142-stag-media-read", "note": "P_read は模型の中の量で生存確率・復興確率ではない", "python": platform.python_version(),
           "inputs": {"VA": VA.tolist(), "VB": VB.tolist(), "DC": DC, "DS": DS, "DC_copan": DCc, "p_lost": 0.9},
           "checks_max": max(chk.values()), "variants": RES2, "argmax_amount": best_amt, "hyp": HYP, "runtime_s": round(time.time() - T0, 1)},
          open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
keys = [m[0] for m in MEDIA]
print("checks ok, max diff", max(chk.values()))
print("delay", {nm: {k: round(v, 1) if k == "mean_delay" else round(v, 3) for k, v in r["delay"].items()} for nm, r in RES2.items()})
print(f"{'media':10s} " + " ".join(f"{nm:>8s}" for nm in SPECS))
for k in keys: print(f"{k:10s} " + " ".join(f"{RES2[nm]['media'][k]['P_read_H']:8.3f}" for nm in SPECS))
print("P_K2e (A):", {k: round(mA[k]["P_K2e"], 3) for k in keys})
print("P_K01 (A):", {k: round(mA[k]["P_K01"], 3) for k in keys})
print("piql E_frac:", {nm: round(r["media"]["piql"]["E_frac_H"], 3) for nm, r in RES2.items()})
print("silica/mom_mf amount ratio:", {nm: round(r["media"]["silica"]["amount_x_P"] / r["media"]["mom_mf"]["amount_x_P"]) for nm, r in RES2.items()})
print("argmax", best_amt); print(json.dumps(HYP, ensure_ascii=False, default=float))
fig, ax = plt.subplots(figsize=(9, 5)); x = np.arange(len(keys)); sel = ["none", "crude141", "A", "E2", "E3", "A_pl05"]
for j, nm in enumerate(sel): ax.bar(x + (j - 2.5) * 0.14, [RES2[nm]["media"][k]["P_read_H"] for k in keys], 0.14, label=nm)
ax.set_xticks(x); ax.set_xticklabels(keys, rotation=60, fontsize=8); ax.set_ylabel("P_read within 1000 y (model quantity)")
ax.set_title("142: media readability under historical stagnation (not a survival probability)", fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{HERE}/fig_stag_media.png", dpi=110)
print("runtime", round(time.time() - T0, 1))
