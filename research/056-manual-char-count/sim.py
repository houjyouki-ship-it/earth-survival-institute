"""研究056：T（板の手引き）の字数を実在の実務書の字数から積み上げる（open_problems 151）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md。
055 の sim.py を「# ---------- 仮説」の前まで exec し（055 の費用と CE を差0で再現）、cost_T の字数だけを差し替える。新しい乱数は seed 20261056。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P55 = os.path.abspath(os.path.join(HERE, "..", "055-authoring-productivity", "sim.py"))
src = open(P55, encoding="utf8").read(); cut = src.index("# ---------- 仮説")
G = {"__file__": P55, "__name__": "sim055"}
exec(compile(src[:cut], P55, "exec"), G)
r55 = json.load(open(os.path.join(HERE, "..", "055-authoring-productivity", "results.json")))
R = {"study": "056-manual-char-count", "seed": 20261056}

def leaves(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items(): yield from leaves(v, p + "/" + k)
    elif isinstance(x, list):
        for i, v in enumerate(x): yield from leaves(v, p + f"[{i}]")
    else: yield p, x
a_ = dict(leaves(r55["main_chars"])); b_ = dict(leaves(json.loads(json.dumps(G["R"]["main_chars"], default=float))))
R["check_055_main_chars"] = {"n_leaves": len(a_), "max_abs_rel_diff": max(abs(a_[k] - b_[k]) / max(1e-300, abs(a_[k])) for k in a_ if a_[k] != 0)}

gc = G["gc"]; NC = G["NC"]; q = G["q"]; usd_per_char = G["usd_per_char"]; eur = G["eur"]
C_C = G["C_C"]; cost_STC = G["cost_STC"]; cost_SLD = G["cost_SLD"]; vals = G["vals"]; grow_ratio = G["grow_ratio"]; KEYS = G["KEYS"]
gn = np.random.default_rng(20261056)
def lu(lo, hi, n=NC): return np.exp(gn.uniform(np.log(lo), np.log(hi), n))

# ---------- 段ごとの字数（sources/gutenberg_counts.json、ure_stage_counts.json） ----------
gcnt = json.load(open(os.path.join(HERE, "sources", "gutenberg_counts.json")))
ure = json.load(open(os.path.join(HERE, "sources", "ure_stage_counts.json")))["stages"]
c = lambda i: gcnt[str(i)]["chars"]
STAGES = {  # 段: (簡潔, 全部, 読取装置より前か)
    "metal":   (ure["Fe_metal"]["chars"], c(38015), True),
    "glass":   (ure["glass"]["chars"], ure["glass"]["chars"] + c(30066) + c(32962) + c(55266), True),
    "lens":    (c(53542), c(53542) + c(53494) + c(53740), True),
    "chem":    (ure["chem"]["chars"], ure["chem"]["chars"] + c(30775) + c(21724), True),
    "elec":    (c(22766), c(22766) + c(27257) + c(39053) + c(14986), True),
    "mach":    (c(48179), c(60819) + c(56776), True),
    "ceramic": (ure["ceramic_build"]["chars"], ure["ceramic_build"]["chars"] + c(38248), True),
    "reader":  (c(6934), c(6934) + c(38526) + c(15617) + c(33437), False),
    "med":     (c(48929), c(26718) + c(17921), False),
    "agri":    (c(48753), c(48753) + c(35696) + c(20168), False),
}
R["stage_anchors"] = {k: {"concise": v[0], "full": v[1], "pre_reader": v[2]} for k, v in STAGES.items()}
R["sum_concise"] = sum(v[0] for v in STAGES.values()); R["sum_full"] = sum(v[1] for v in STAGES.values())
cs = {k: lu(v[0], v[1]) for k, v in STAGES.items()}
m_tacit = lu(1, 3); m_cover = lu(1, 2)
base = sum(cs.values())
CH = {
    "CH_heavy": base * m_tacit * m_cover,
    "CH_reader": cs["reader"] * m_tacit,
    "CH_comp": lu(gcnt["53143"]["chars"], gcnt["53143"]["chars"] + gcnt["44562"]["chars"]) * m_tacit,
    "CH_053": lu(1e7, 1e8),
    "CH_greq": lu(0.725e6, 8.9e6),
}
pre_share = sum(cs[k] for k, v in STAGES.items() if v[2]) / base
R["chars"] = {k: q(v) for k, v in CH.items()}
R["pre_reader_share"] = q(pre_share)
R["stage_share_median"] = {k: float(np.median(cs[k] / base)) for k in STAGES}

# ---------- 費用と CE（055 の run を字数の配列で） ----------
def run_ch(version, chars):
    per_tab = lu(5e4, 5e6); n_tab = np.ceil(chars / per_tab)
    auth = chars * usd_per_char(version); C_T = auth + n_tab * 195 * eur * 3
    C_G = chars * usd_per_char(version) + np.ceil(chars / lu(5e4, 5e6)) * 195 * eur * 3
    C_TC = C_T + C_C; C_STC = cost_STC(n_tab); C_LD = cost_SLD(chars, 2)
    out = {"costs": {"C_T": q(C_T), "C_TC": q(C_TC), "C_STC": q(C_STC), "C_SLD_AU": q(C_LD)}, "CE": {}, "_C_TC": C_TC, "_C_STC": C_STC}
    cost = {"TC": C_TC, "S_TC_given_TC": C_STC, "S_LD_given_TC": C_LD}
    for key in KEYS:
        v = vals(key); a, u = key.split("/"); out["CE"][key] = {}
        for p in ("total", "heavy", "light"):
            vt = v["TAB"][p]; d = {}
            if vt <= 0: continue
            for x, cc in cost.items():
                ce = v[x][p] / vt * C_T / cc
                d[x] = {"value_ratio": v[x][p] / vt, "CE_q": q(ce)}
            d["P_TC_gt_STC"] = float(((v["TC"][p] / C_TC) > (v["S_TC_given_TC"][p] / C_STC)).mean())
            if p == "total":
                ce = grow_ratio(a, u) * C_T / C_G; d["GROW"] = {"CE_q": q(ce)}
            out["CE"][key][p] = d
    return out
VERS = ["V_mix", "V_orig", "V_comp"]
RUN = {ch: {ver: run_ch(ver, CH[ch]) for ver in VERS} for ch in CH}
# 釣り合い：CH_heavy の T＋C が CH_reader と同じ1ドルあたりの価値になるのに要る価値の倍率
R["breakeven_value_factor_heavy_vs_reader"] = {ver: float(np.median(RUN["CH_heavy"][ver]["_C_TC"]) / np.median(RUN["CH_reader"][ver]["_C_TC"])) for ver in VERS}
# S_TC と T＋C の順位が入れ替わる字数（V_mix、A_Ord・U3、中央値）：字数を固定して走査
scan = []
for L in np.logspace(5.5, 8, 26):
    o = run_ch("V_mix", np.full(NC, L))
    scan.append({"chars": float(L), **{key: {"TC": o["CE"][key]["total"]["TC"]["CE_q"][1], "S_TC": o["CE"][key]["total"]["S_TC_given_TC"]["CE_q"][1],
                 "P_TC_gt_STC": o["CE"][key]["total"]["P_TC_gt_STC"]} for key in KEYS}})
R["scan_fixed_chars_V_mix"] = scan
def crossing(key):
    for a, b in zip(scan, scan[1:]):
        fa = a[key]["TC"] - a[key]["S_TC"]; fb = b[key]["TC"] - b[key]["S_TC"]
        if fa > 0 >= fb: return float(np.exp(np.log(a["chars"]) + (np.log(b["chars"]) - np.log(a["chars"])) * fa / (fa - fb)))
    return None
R["crossing_chars_TC_eq_STC"] = {key: crossing(key) for key in KEYS}
for ch in RUN:
    for ver in RUN[ch]: RUN[ch][ver].pop("_C_TC"); RUN[ch][ver].pop("_C_STC")
R["runs"] = RUN

# ---------- 仮説 ----------
med = lambda ch, key, x, p="total", ver="V_mix": RUN[ch][ver]["CE"][key][p][x]["CE_q"][1]
Hh = {}
mh = R["chars"]["CH_heavy"][1]
Hh["H68a"] = {"value": mh, "rule": "median CH_heavy in [1e7,1e8]", "supported": 1e7 <= mh <= 1e8}
lo, hi = R["chars"]["CH_reader"][0], R["chars"]["CH_reader"][2]
Hh["H68b"] = {"value": [lo, hi], "rule": "CH_reader 5-95% inside [0.725e6, 8.9e6]", "supported": lo >= 0.725e6 and hi <= 8.9e6}
Hh["H68c"] = {"value": [mh, R["chars"]["CH_053"][1]], "rule": "median CH_heavy < median CH_053", "supported": mh < R["chars"]["CH_053"][1]}
v = {key: [med("CH_heavy", key, "S_TC_given_TC"), med("CH_heavy", key, "TC")] for key in KEYS}
Hh["H68d"] = {"value": v, "rule": "CH_heavy V_mix: S_TC > TC > 1 both", "supported": all(s > t > 1 for s, t in v.values())}
v = [med("CH_reader", "A_Ord/U3", "TC"), med("CH_reader", "A_Ord/U3", "S_TC_given_TC")]
Hh["H68e"] = {"value": v, "rule": "CH_reader V_mix A_Ord: TC > S_TC", "supported": v[0] > v[1]}
Hh["H68f"] = {"value": R["pre_reader_share"][1], "rule": "median pre-reader share > 0.5", "supported": R["pre_reader_share"][1] > 0.5}
rr = R["chars"]["CH_comp"][1] / mh
Hh["H68g"] = {"value": rr, "rule": "CH_comp/CH_heavy median in [1/3,3]", "supported": 1 / 3 <= rr <= 3}
v = {key: med("CH_heavy", key, "S_TC_given_TC", "heavy") for key in KEYS}
Hh["H68h"] = {"value": v, "rule": "heavy-only CE(S_TC) > 1 both (CH_heavy V_mix)", "supported": all(x > 1 for x in v.values())}
R["hypotheses"] = Hh
R["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, ax = plt.subplots(figsize=(8, 4.8))
names = list(CH); data = [np.log10(CH[k]) for k in names]
ax.violinplot(data, showmedians=True); ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names)
ax.axhspan(7, 8, color="C1", alpha=0.12, label="053: 1e7-1e8"); ax.axhspan(np.log10(0.725e6), np.log10(8.9e6), color="C2", alpha=0.12, label="024 g_req")
ax.set_ylabel("log10 characters of manual T"); ax.legend(loc="upper left"); ax.set_title("056: T length from counted practitioner texts")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_char_distributions.png"), dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.8))
xs = [s["chars"] for s in scan]
for key, col in zip(KEYS, ("C0", "C3")):
    ax.plot(xs, [s[key]["TC"] for s in scan], "-", color=col, label=f"T+C {key}")
    ax.plot(xs, [s[key]["S_TC"] for s in scan], "--", color=col, label=f"S_TC|T+C {key}")
for ch, ls in (("CH_reader", ":"), ("CH_heavy", "-.")):
    ax.axvline(R["chars"][ch][1], color="k", ls=ls, lw=1, label=f"median {ch}")
ax.axhline(1, color="gray", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("characters of T (fixed)"); ax.set_ylabel("median CE (TAB=1), V_mix"); ax.legend(fontsize=7); ax.set_title("Rank flip between T+C and S_TC vs T length")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ce_vs_chars.png"), dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.2))
ks = list(STAGES); lo_ = [STAGES[k][0] / 1e6 for k in ks]; hi_ = [STAGES[k][1] / 1e6 for k in ks]
ax.barh(ks, hi_, color="C0", alpha=0.35, label="full (sum of treatises)"); ax.barh(ks, lo_, color="C0", label="concise")
ax.set_xlabel("million characters (counted, Project Gutenberg)"); ax.legend(); ax.set_title("Stage anchors")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_stage_anchors.png"), dpi=120); plt.close(fig)

json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({"check055": R["check_055_main_chars"], "sum_concise": R["sum_concise"], "sum_full": R["sum_full"], "chars": R["chars"],
                  "pre_share": R["pre_reader_share"], "stage_share": R["stage_share_median"], "breakeven": R["breakeven_value_factor_heavy_vs_reader"],
                  "crossing": R["crossing_chars_TC_eq_STC"]}, indent=1, default=float))
for ch in RUN:
    for ver in VERS:
        o = RUN[ch][ver]
        print(ch, ver, "C_T", [round(x / 1e6, 3) for x in o["costs"]["C_T"]], "C_STC", round(o["costs"]["C_STC"][1] / 1e6, 3))
        for key in KEYS:
            for p in ("total", "heavy", "light"):
                d = o["CE"][key][p]
                print("   ", key, p, "TC", round(d["TC"]["CE_q"][1], 3), "S_TC", round(d["S_TC_given_TC"]["CE_q"][1], 3), "S_LD", round(d["S_LD_given_TC"]["CE_q"][1], 3),
                      "P(TC>STC)", round(d["P_TC_gt_STC"], 3), ("GROW " + str(round(d["GROW"]["CE_q"][1], 3))) if "GROW" in d else "")
for k, h in Hh.items(): print(k, h["supported"], h["value"])
print("runtime", round(R["runtime_s"], 1))
