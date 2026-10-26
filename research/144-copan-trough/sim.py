"""研究144：Copán 谷の谷の人口の下限（open_problems 331）を原典の要約で確かめ、その場の集合の p|lost を決める。
計算のみ・ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-27 05:34 JST）。
注：F_heavy・V・P_read は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, math, runpy, shutil, time, contextlib, io
import numpy as np
from scipy import stats, integrate
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R125 = json.load(open(f"{RES}/125-pop-recovery-history/results.json", encoding="utf8"))
R143 = json.load(open(f"{RES}/143-plost-new-cases/results.json", encoding="utf8"))
GR = json.load(open(f"{RES}/128-stag-given-stageloss/grid.json"))
r = R125["inputs"]["r_median_038"]
PS = np.array(GR["PS"]); SL = "slow_Tasmania"
FC = np.array([GR["GRID"][f"{p:.6f}|{SL}|D2rz|none"]["F_heavy"] for p in PS])
def Fx(x):
    x = np.atleast_1d(x); y = np.interp(x, PS, FC); lo = x < PS[0]
    y[lo] = FC[0] + (x[lo] - PS[0]) * (FC[1] - FC[0]) / (PS[1] - PS[0]); return y
def post_mix(w, k=2, n=3, a0=0.5, b0=0.5):
    """Copán が停滞である確率 w の混合：w·Beta(a0+k+1, b0+n-k-1) + (1-w)·Beta(a0+k, b0+n-k)。k,n は Copán を除く 2/2 と Copán の 1 件。"""
    dS = stats.beta(a0 + k + 1, b0 + n - k - 1); dN = stats.beta(a0 + k, b0 + n - k)
    pdf = lambda x: w * dS.pdf(x) + (1 - w) * dN.pdf(x)
    pm = w * dS.mean() + (1 - w) * dN.mean()
    Fm = integrate.quad(lambda x: Fx(x)[0] * pdf(x), 0, 1, limit=200)[0]
    return {"w": w, "p_mean": pm, "F_mean": Fm}
# ---- 再現：143 のその場の 3/3・2/3 と損益分岐 ----
CHECK = {}
for w, key in ((1.0, "insitu(Greece,Britain,Copan_stag)|jeffreys"), (0.0, "insitu_Copan_no|jeffreys")):
    q = post_mix(w); ref = R143["posterior"][key]
    CHECK[key] = {"p_diff": abs(q["p_mean"] - ref["p_mean"]), "F_diff": abs(q["F_mean"] - ref["F_mean"])}
    assert CHECK[key]["p_diff"] < 1e-12 and CHECK[key]["F_diff"] < 1e-6, CHECK
brk143 = 28000 / math.exp(r * (2023 - 800)); CHECK["breakeven_143"] = abs(brk143 - R143["copan"]["Pmin_breakeven_upper"]); assert CHECK["breakeven_143"] < 1e-9
# ---- 本回の資料（sources/ の頁の文だけ） ----
S = "sources/"
EV = {
 "P0_timing": S + "webster1992_abstract_es.txt（Webster・Sanders・van Rossum 1992, Ancient Mesoamerica 3:185-198 の西語抄録）：'Un pico máximo de 26,000 a 27,000 personas se alcanzó entre 750 y 900 d.C'",
 "decline": S + "同：'No hubo un colapso demográfico abrupto, y cerca de un tercio de la población estaba presente todavía hacia 1000–1050 d.C; 200 años después el valle estaba esencialmente despoblado'。英語抄録：simulation 'between A.D. 400–1250'、2,048 の水和年代、既知の遺跡の 15%",
 "manahan": S + "manahan.txt（FAMSI 98038、Manahan）：Ejar 期（1000〜1200）の村は 'very small yet discrete ... within 200 meters of the site center'；'left the valley virtually devoid of inhabitants within a century following the end of the Copán dynasty'（Webster らより早く深い減少を主張）",
 "wiki_2026": S + "wiki_Copan_2026.wikitext：Late Classic の谷 18,000〜25,000（中心 6,000〜9,000、農村 476 km²）；'only a few farming hamlets at the time of the arrival of the Spanish'",
 "census": S + "citypop_copan_ruinas.html：Copán Ruinas 自治体（361 km²）2001 年 30,703、2013 年 38,109、2023 推計 45,312（INE、過少計数を補正）。" + S + "unah.txt：1950〜2013 の年平均増加 2.26%、2013 年 農村 29,974＋都市 8,027。1950〜1988 の各回の値は図のみで文字なし（取れず）",
}
NUM = {"Pmin_number_found": None, "P0_range": [26000, 27000], "P0_window": [750, 900], "third_at_1000_1050": True,
       "essentially_depopulated_by": [1200, 1250], "census": {"2001": 30703, "2013": 38109, "2023proj": 45312}, "growth_1950_2013": 0.0226}
# T_rec：谷が P0 に戻った年。2001 年に 30,703 ≥ 27,000 なので上限 2001。2.26% の平均で逆算（推測、図の値なし）
P2013 = 29974 + 8027
def year_reach(P):  # 1950〜2013 の平均増加率で一定と置いた逆算
    return 2013 - math.log(P2013 / P) / math.log(1 + NUM["growth_1950_2013"])
TREC = {"upper_2001": 2001, "interp_2.26pct_27000": year_reach(27000), "interp_2.26pct_26000": year_reach(26000), "pop1950_backcalc": P2013 / (1 + NUM["growth_1950_2013"]) ** 63}
def excess(P0, Pmin, ts, trec): return (trec - ts) - math.log(P0 / Pmin) / r
def breakeven(P0, ts, trec): return P0 / math.exp(r * (trec - ts))
BRK = {f"P0={p0}|ts={ts}|Trec={tr}": breakeven(p0, ts, tr) for p0 in (26000, 27000, 28000) for ts in (800, 900) for tr in (1985, 1994, 2001, 2023)}
# ---- 登録の規則：数がない → Pmin を [200,1000] の対数一様、P0・ts・T_rec は格子で等重み ----
rng = np.random.default_rng(144)
def w_stag(lo, hi, P0s=(26000, 27000), tss=(800, 900), trecs=(round(year_reach(26500)), 2001), n=200000):
    pm = np.exp(rng.uniform(math.log(lo), math.log(hi), n)); out = {}
    for p0 in P0s:
        for ts in tss:
            for tr in trecs:
                out[f"P0={p0}|ts={ts}|Trec={tr}"] = float(np.mean((tr - ts) - np.log(p0 / pm) / r > 0))
    return float(np.mean(list(out.values()))), out
VARS = {}
for lab, lo, hi, trecs, tss in (("main_reg[200,1000]", 200, 1000, (round(year_reach(26500)), 2001), (800, 900)),
                                ("reg_ts800_only", 200, 1000, (round(year_reach(26500)), 2001), (800,)),
                                ("143_inputs_Trec2023", 200, 1000, (2023,), (800, 900)),
                                ("unreg_[50,1000](esencialmente despoblado)", 50, 1000, (round(year_reach(26500)), 2001), (800, 900)),
                                ("unreg_[50,500]", 50, 500, (round(year_reach(26500)), 2001), (800, 900))):
    w, cells = w_stag(lo, hi, trecs=trecs, tss=tss); VARS[lab] = {"w": w, "cells": cells, **post_mix(w)}
# 登録外：Copán を『放棄』と見なすと規則で停滞かつその場の集合から外れ、その場は Greece・Britain の 2/2
_d = stats.beta(0.5 + 2, 0.5); VARS["unreg_Copan_as_abandon(insitu 2/2)"] = {"w": float("nan"), "cells": {}, "p_mean": _d.mean(),
    "F_mean": integrate.quad(lambda x: Fx(x)[0] * _d.pdf(x), 0, 1, limit=200)[0]}
# ---- 142 の媒体の表（143 と同じく 142 の sim を写しで実行） ----
W = "/tmp/rep144/research"; os.makedirs(f"{W}/142-stag-media-read", exist_ok=True)
for d in ("141-medium-tier-readability", "038-restart-state-evidence", "132-trough-lag-definition", "125-pop-recovery-history"):
    if not os.path.exists(f"{W}/{d}"): os.symlink(f"{RES}/{d}", f"{W}/{d}")
shutil.copy(f"{RES}/142-stag-media-read/sim.py", f"{W}/142-stag-media-read/sim.py")
_cwd = os.getcwd(); os.chdir(f"{W}/142-stag-media-read")
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{W}/142-stag-media-read/sim.py", run_name="m144")
os.chdir(_cwd)
KEYS = ("mom_mf", "fiche24", "rosetta", "piql", "silica")
MED = {}
for lab, p in (("insitu_3/3(143)", R143["media_P_read_1000y"]["insitu"]["p"]), *[(k, v["p_mean"]) for k, v in VARS.items()]):
    o, dl = G["run"]({"kind": "A", "p": p}); MED[lab] = {"p": p, **{k: o[k]["P_read_H"] for k in KEYS}}
d = abs(MED["insitu_3/3(143)"]["fiche24"] - R143["media_P_read_1000y"]["insitu"]["fiche24"]); CHECK["143_media_insitu_fiche24"] = d; assert d < 1e-12
# ---- 仮説 ----
mv = VARS["main_reg[200,1000]"]
H = {
 "H156a": {"class": "confirmatory", "supported": None, "value": "登録文の向きの誤り：143 の規則では Pmin が小さいほど停滞なし（789 未満→2/3）。原典の要約に数なし（'esencialmente despoblado' 1200〜1250、Manahan 'virtually devoid'）。規則により判定保留、混合で閉じる"},
 "H156b": {"class": "confirmatory", "supported": True, "value": f"自治体（361 km²）は 2001 年に 30,703 ≥ 27,000、1950 年の逆算 {P2013 / 1.0226 ** 63:.0f}。2.26% 一定の逆算で 27,000 到達は {year_reach(27000):.0f} 年（推測）"},
 "H156c": {"class": "confirmatory", "supported": True, "value": "頂点 26,000〜27,000 は 750〜900 年（Webster ら 1992）。Wikipedia の 400〜450 年 28,000 とは食い違う"},
 "H156d": {"class": "prediction", "supported": mv["w"] >= 2 / 3, "value": mv["w"]},
 "H156e": {"class": "prediction", "supported": 2.0 <= mv["F_mean"] <= 2.17, "value": mv["F_mean"]},
}
OUT = {"study": "144-copan-trough", "note": "F_heavy・P_read は生存確率や復興確率ではない", "r_median_038": r, "check": CHECK, "evidence": EV, "numbers": NUM,
       "T_rec": TREC, "breakeven_Pmin": BRK, "variants": VARS, "media_P_read_1000y": MED, "hypotheses": H, "runtime_s": None}
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
pm = np.logspace(math.log10(30), math.log10(3000), 300)
for p0, ts, tr, ls in ((27000, 800, 2001, "-"), (27000, 900, 2001, "--"), (27000, 800, round(year_reach(26500)), "-."), (28000, 800, 2023, ":")):
    ax[0].plot(pm, [excess(p0, x, ts, tr) for x in pm], ls, label=f"P0={p0}, ts={ts}, Trec={tr}")
ax[0].axhline(0, c="k", lw=.6); ax[0].axvspan(200, 1000, alpha=.12, label="registered Pmin range")
ax[0].set_xscale("log"); ax[0].set_xlabel("Copán valley trough population Pmin"); ax[0].set_ylabel("excess vs 038 median (years)"); ax[0].legend(fontsize=7)
ax[0].set_title("Stagnation if excess > 0", fontsize=9)
labs = list(VARS); ax[1].barh(range(len(labs)), [VARS[l]["F_mean"] for l in labs]); ax[1].set_yticks(range(len(labs))); ax[1].set_yticklabels(labs, fontsize=7)
ax[1].axvline(R143["posterior"]["insitu(Greece,Britain,Copan_stag)|jeffreys"]["F_mean"], c="r", ls=":", label="3/3 (143)")
ax[1].axvline(R143["posterior"]["insitu_Copan_no|jeffreys"]["F_mean"], c="g", ls=":", label="2/3 (143)")
ax[1].set_xlim(1.85, 2.2); ax[1].set_xlabel("F_heavy (in-situ set, slow κ·D2rz·W·H1000)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_copan_trough.png", dpi=120)
OUT["runtime_s"] = time.time() - T0
json.dump(OUT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1)
print("check", CHECK); print("Trec", TREC); print("breakeven", {k: round(v) for k, v in BRK.items()})
for k, v in VARS.items(): print(k, "w=%.3f p=%.3f F=%.3f" % (v["w"], v["p_mean"], v["F_mean"]), {c: round(x, 2) for c, x in v["cells"].items()})
for k, v in MED.items(): print(k, {a: round(b, 3) for a, b in v.items()})
print({k: (v["supported"], v["value"] if not isinstance(v["value"], str) else "") for k, v in H.items()}); print("runtime", round(OUT["runtime_s"], 1))
