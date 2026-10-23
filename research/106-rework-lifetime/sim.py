"""研究106：削り直し・擦り合わせ（Whitworth の三面）で廃墟の道具の精度の面を作り直すと、使える年数 L がどこまで延び、
橋渡しの版（105：L までに元の閾値に届かなければ効かない）で F_heavy がどう変わるか（open_problems 253＋251）。
あわせて 104 の主な版を橋渡しの版で出し直す（254）。計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-24 01:36 JST）。
104 の sim.py を「# ---------- 仮説」の前まで exec（105 と同じ）。新しい乱数は seed 20261106。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P104 = os.path.abspath(os.path.join(HERE, "..", "104-ruin-glass-thresholds", "sim.py"))
src = open(P104, encoding="utf8").read(); cut = src.index("# ---------- 仮説")
G = {"__file__": P104, "__name__": "sim104"}
exec(compile(src[:cut], P104, "exec"), G)
R104 = json.load(open(os.path.join(os.path.dirname(P104), "results.json"), encoding="utf8"))
R105 = json.load(open(os.path.join(HERE, "..", "105-imported-tools-m", "results.json"), encoding="utf8"))
K = "A_Ord/U3"
d104 = max(abs(G["R"]["runs"][t][K]["F_heavy"] - R104["runs"][t][K]["F_heavy"]) for t in R104["runs"])
assert d104 < 1e-12
evaluate, N_Fe, N_lens, N_T2e, L_bulk, L_prec, L_lens, ok_n, Nn, parts, FIXED, B, Wc, d12 = (G[k] for k in (
    "evaluate", "N_Fe", "N_lens", "N_T2e", "L_tool_bulk", "L_tool_prec", "L_lens", "ok_n", "N", "parts", "FIXED", "B", "W", "d12"))
R = {"study": "106-rework-lifetime", "seed": 20261106, "check_104_runs_reproduced_maxdiff": d104}
g = np.random.default_rng(20261106)
def lu(lo, hi, n=Nn): return np.exp(g.uniform(np.log(lo), np.log(hi), n))

# ---------- 橋渡しの版（105 と同じ式） ----------
_orig = G["T2s_mod"]
def T2s_bridge(k, m_l=None, L_l=np.inf, m_t=None, L_t=np.inf, ok=True):
    if k in FIXED: return B[k]["T2s"]
    st, n, wait, d01e, tFe = parts(k)
    wl = wait(N_lens); wt = wait(N_T2e)
    if m_l is not None:
        wl2 = wait(np.maximum(m_l * N_lens, N_Fe)); wl = np.where(ok & (wl <= L_l), wl2, wl)
    if m_t is not None:
        wt2 = wait(m_t * N_T2e); wt = np.where(ok & (wt <= L_t), wt2, wt)
    t1 = np.where(st >= 5, np.maximum(tFe + d01e, wl), 0.0)
    t2 = np.where(st >= 4, np.maximum(t1 + d12, wt), 0.0)
    return np.select([st == 2, st == 3, st >= 4], [Wc, np.zeros(Nn), t2], default=np.zeros(Nn))
G["T2s_mod"] = T2s_bridge
FB = lambda tag, **kw: evaluate(tag, **kw)[K]["F_heavy"]
# 再現：105 の B_m0.3_bulk
rep = FB("B_m0.3_bulk_repro", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_bulk, ok=ok_n)
R["check_105_B_m0.3_bulk_diff"] = abs(rep - R105["bridge_F_heavy"]["B_m0.3_bulk"]); assert R["check_105_B_m0.3_bulk_diff"] < 1e-12

# ---------- 作り直しの L ----------
# 【事実】Surface plate（Wikipedia）：三面を対で擦り合わせ高い所を手で削ると、手本の基準なしに平面ができる（"automatic generation of gages"）。
# 【事実】Hand scraper（Wikipedia）：要る道具は未仕上げの鋳物の板 3 枚、平スクレーパー、光明丹（または鉛丹）だけ。人数・時間の数値は資料にない。
# 【推測】1 回の作り直しで削る厚さ δ、知識が残る確率 p_know。削れる厚さは部材の th/4 = 5 mm（103/104 の定義）。
delta = lu(0.02, 0.2)
n_max = 5.0 / delta
L_rw_full = np.minimum(L_bulk, n_max * L_prec)          # 知識が必ず残る
know = g.uniform(size=Nn) < lu(0.1, 1.0)                 # p_know logU(0.1,1) を標本ごとに
L_rw_mix = np.where(know, L_rw_full, L_prec)
R["inputs"] = {k: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)] for k, v in
               {"L_prec": L_prec, "L_bulk": L_bulk, "n_max": n_max, "L_rw_full": L_rw_full, "L_rw_mix": L_rw_mix}.items()}
R["share_rw_limited_by_material"] = float(np.mean(n_max * L_prec < L_bulk))
# 重い崩壊（st>=4）で wait(N_T2e) <= L となる割合（橋渡しで効きうる標本）
def share_bridge(L):
    out = []
    for k in G["COMPS"]:
        if k in FIXED: continue
        st, n, wait, *_ = parts(k); m = st >= 4
        if m.any(): out.append(((wait(N_T2e) <= L) & ok_n)[m] if np.ndim(L) == 0 else ((wait(N_T2e) <= L) & ok_n)[m])
    return float(np.concatenate(out).mean())
R["share_heavy_wait_T2e_le_L"] = {"prec": share_bridge(L_prec), "rework_full": share_bridge(L_rw_full),
                                  "rework_mix": share_bridge(L_rw_mix), "bulk": share_bridge(L_bulk)}

m_prod = lu(0.7, 1.0); m_main_l = lu(0.1, 1.0); m_main_t = lu(0.1, 1.0)
runs = {
 "noRW_prec_m0.3": FB("noRW_prec_m0.3", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_prec, ok=ok_n),
 "RW_full_m0.3": FB("RW_full_m0.3", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_rw_full, ok=ok_n),
 "RW_mix_m0.3": FB("RW_mix_m0.3", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_rw_mix, ok=ok_n),
 "RW_full_mprod": FB("RW_full_mprod", m_l=m_prod, L_l=L_lens, m_t=m_prod, L_t=L_rw_full, ok=ok_n),
 "RW_mix_mprod": FB("RW_mix_mprod", m_l=m_prod, L_l=L_lens, m_t=m_prod, L_t=L_rw_mix, ok=ok_n),
 "RW_full_mmain": FB("RW_full_mmain", m_l=m_main_l, L_l=L_lens, m_t=m_main_t, L_t=L_rw_full, ok=ok_n),
 "RW_full_m0.1": FB("RW_full_m0.1", m_l=0.1, L_l=L_lens, m_t=0.1, L_t=L_rw_full, ok=ok_n),
 "RW_full_m0.3_delta0.02": FB("RW_full_m0.3_d002", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=np.minimum(L_bulk, 250 * L_prec), ok=ok_n),
 "RW_full_m0.3_delta0.2": FB("RW_full_m0.3_d02", m_l=0.3, L_l=L_lens, m_t=0.3, L_t=np.minimum(L_bulk, 25 * L_prec), ok=ok_n),
}
R["runs_F_heavy"] = runs

# ---------- 254：104 の主な版を橋渡しの版で ----------
v104 = {t: R104["runs"][t][K]["F_heavy"] for t in R104["runs"]}
redo = {
 "both_m0.3_life_bulk": dict(m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_bulk, ok=ok_n),
 "both_m0.3_life_prec": dict(m_l=0.3, L_l=L_lens, m_t=0.3, L_t=L_prec, ok=ok_n),
 "both_main_life_bulk": dict(m_l=G["m_l_main"], L_l=L_lens, m_t=G["m_t_main"], L_t=L_bulk, ok=ok_n),
 "both_main_life_prec": dict(m_l=G["m_l_main"], L_l=L_lens, m_t=G["m_t_main"], L_t=L_prec, ok=ok_n),
 "both_m0.1_life_bulk": dict(m_l=0.1, L_l=L_lens, m_t=0.1, L_t=L_bulk, ok=ok_n),
 "t2_m0.3_life_bulk": dict(m_t=0.3, L_t=L_bulk, ok=ok_n),
 "lens_main": dict(m_l=G["m_l_main"], L_l=L_lens, ok=ok_n),
}
R254 = {}
for t, kw in redo.items():
    fb = FB("bridge_" + t, **kw); fo = v104[t]
    R254[t] = {"F104": fo, "F_bridge": fb, "gain_reduction": (1 - (fb - 1) / (fo - 1)) if fo - 1 > 1e-9 else None}
R["redo_254"] = R254
G["T2s_mod"] = _orig

# ---------- 仮説 ----------
H = {}
H["H118a"] = {"class": "confirmatory", "rule": "public source states three-plate method yields a flat reference without a pre-existing master",
              "supported": True, "note": "Surface plate: 'do not derive their precision from more-precise standards ... three approximately flat surfaces ... only stable, mutually conjugate surface shape is a plane'"}
H["H118b"] = {"class": "confirmatory", "rule": "none of the opened sources gives a number of people or time for hand-making a reference plane",
              "supported": True, "note": "Surface plate, Hand scraper, Lapping, Joseph Whitworth, Scraper (metalworking), Flatness: no headcount or duration"}
H["H118c"] = {"class": "prediction", "rule": "bridge m=0.3: RW_full >= 1.30 and noRW_prec < 1.05",
              "value": [runs["RW_full_m0.3"], runs["noRW_prec_m0.3"]],
              "supported": bool(runs["RW_full_m0.3"] >= 1.30 and runs["noRW_prec_m0.3"] < 1.05)}
H["H118d"] = {"class": "prediction", "rule": "m_prod logU(0.7,1) with full rework: F_heavy-1 < 0.06", "value": runs["RW_full_mprod"],
              "supported": bool(runs["RW_full_mprod"] - 1 < 0.06)}
sel = ["both_m0.3_life_prec", "both_main_life_bulk", "both_main_life_prec", "both_m0.1_life_bulk", "t2_m0.3_life_bulk"]
H["H118e"] = {"class": "prediction", "rule": "each of 5 listed 104 runs: bridge reduces F_heavy-1 by >= 20%",
              "value": {t: R254[t]["gain_reduction"] for t in sel},
              "supported": bool(all((R254[t]["gain_reduction"] or 0) >= 0.20 for t in sel))}
R["hypotheses"] = H

# ---------- 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for lab, L in (("precision face (no rework)", L_prec), ("rework (p_know logU 0.1-1)", L_rw_mix), ("rework always", L_rw_full), ("bulk 20 mm", L_bulk)):
    x = np.sort(L); ax[0].plot(x, np.arange(1, len(x) + 1) / len(x), label=lab)
ax[0].set_xscale("log"); ax[0].set_xlabel("usable life L of ruin machine tools (years)"); ax[0].set_ylabel("CDF")
ax[0].legend(fontsize=7); ax[0].set_title("106: re-scraping (three-plate) extends L toward bulk life", fontsize=8)
names = list(runs); ax[1].barh(names, [runs[n] - 1 for n in names], color="C0")
ax[1].set_xlabel("F_heavy - 1 (A_Ord/U3, bridge version)"); ax[1].tick_params(axis="y", labelsize=7)
ax[1].set_title("bridge: effect needs both rework (L) and low m", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(HERE, "fig_rework.png"), dpi=120)
R["runtime_s"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print(json.dumps({k: R[k] for k in R if k != "runtime_s"}, indent=1, ensure_ascii=False, default=float))
