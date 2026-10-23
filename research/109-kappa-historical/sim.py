"""研究109：床への戻りの速さ κ を歴史の技能の喪失の年数で較正する（open_problems 260）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-24 07:42 JST）。
108 の sim.py を「RU = R["runs"]」の前まで exec（104→…→038 の乱数と 108 の版をそのまま再現）。新しい乱数は seed 20261109。
読み方は 108 と同じ：人口一定の群で κ = ln R / T_loss、R ~ logU(1.1, 10)【推測】。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P108 = os.path.abspath(os.path.join(HERE, "..", "108-floor-reversion-stages", "sim.py"))
src = open(P108, encoding="utf8").read(); cut = src.index('RU = R["runs"]')
G = {"__file__": P108, "__name__": "sim108"}
exec(compile(src[:cut], P108, "exec"), G)
R108 = json.load(open(os.path.join(os.path.dirname(P108), "results.json"), encoding="utf8"))
K = "A_Ord/U3"; run = G["run"]; N = G["N"]
d108 = max(abs(G["R"]["runs"][t][K]["F_heavy"] - R108["runs"][t][K]["F_heavy"]) for t in R108["runs"])
assert d108 < 1e-12
k_hist = G["k_hist"]
g = np.random.default_rng(20261109)
q3 = lambda v: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)]
OUT = {"study": "109-kappa-historical", "seed": 20261109, "check_108_runs_reproduced_maxdiff": d108}

lnR = lambda: g.uniform(np.log(1.1), np.log(10), N)
# ---------- 事例（年は資料の値、幅は【推測】の丸め） ----------
# 【事実（Wikipedia Roman Britain）】"By 407 there were very few new Roman coins going into circulation, and by 430 it is likely that coinage as a
#   medium of exchange had been abandoned. Mass-produced wheel thrown pottery ended at approximately the same time"
# 【事実（Wikipedia Sub-Roman Britain）】ローマの支配の終わりの年の候補：402（貨幣の輸入の終わり）、407、409、410
c_start = g.uniform(402, 410, N)
T_coin = np.maximum(g.uniform(425, 435, N) - c_start, 1.0)
T_pot = np.maximum(g.uniform(420, 440, N) - c_start, 1.0)
# 【事実（Wikipedia Greek Dark Ages）】宮殿は約 1200 BC に終わり始め "the process concluded decades later"、"c. 1170–1100 BC ... palaces had ceased to exist"、
#   Linear B の消失と宮殿の建築。【事実（Wikipedia Linear B）】暗黒時代は "no conclusive evidence of the use of writing"、書記は Pylos 45・Knossos 66。
#   【事実（Wikipedia Late Bronze Age collapse）】"no more monumental stone buildings were built"、Linear B は "no longer a need to keep records" で終わった。
T_linB = g.uniform(10, 100, N)        # 「数十年後」【推測：10〜100 年】
T_stone = g.uniform(10, 100, N)
CASES = {"Britain_coin": lnR() / T_coin, "Britain_wheel_pottery": lnR() / T_pot,
         "Greece_LinearB": lnR() / T_linB, "Greece_monumental_stone": lnR() / T_stone, "Tasmania_108": k_hist}
OUT["cases"] = {k: {"kappa_q": q3(v), "T_loss_q": q3({"Britain_coin": T_coin, "Britain_wheel_pottery": T_pot,
                 "Greece_LinearB": T_linB, "Greece_monumental_stone": T_stone, "Tasmania_108": G["T_main"]}[k])} for k, v in CASES.items()}
# 保持の事例：【事実（Wikipedia Greek Dark Ages）】約 1050 BC の Protogeometric で "a faster potter's wheel"。Peloponnese の小さな遺跡は最大 90% 放棄。
#   轆轤が 1200→1050（150 年）と以後 800 BC まで（400 年）続いたと読むのは【推測】。N_k が分からないので κ の上限にはならない（下の記述）。
OUT["persistence_case"] = {"Greece_potters_wheel_years": [150, 400], "kappa_bound": "none (N_k unknown; persistence implies z_min >= ln N_k only)"}

res = {}
def go(tag, kap):
    o = run("c109_" + tag, kap)
    res[tag] = {"F_heavy": o[K]["F_heavy"], "F_total": o[K]["F_total"], "U3_heavy_rel_drop": o["U3_heavy_rel_drop"],
                "U3_T0_rel_drop": o["U3_T0_rel_drop"], "kappa_q": q3(kap)}
    return res[tag]
for k, v in CASES.items(): go(k, v)
fast = ["Britain_coin", "Britain_wheel_pottery", "Greece_LinearB", "Greece_monumental_stone"]
pick = g.integers(0, len(fast), N); kf = np.choose(pick, [CASES[k] for k in fast]); go("pooled_fast", kf)
pick5 = g.integers(0, 5, N); ka = np.choose(pick5, [CASES[k] for k in CASES]); go("pooled_all_equal", ka)
gm_cases = float(np.exp(np.mean([np.log(np.median(CASES[k])) for k in CASES])))
go("geomean_of_case_medians", np.full(N, gm_cases)); OUT["geomean_of_case_medians"] = gm_cases
gm_fast = float(np.exp(np.mean([np.log(np.median(CASES[k])) for k in fast])))
# 混合：割合 w の標本に Tasmania の遅い κ、残りに速い事例の κ
WS = np.linspace(0, 1, 21); mix = []
for w in WS:
    u = g.random(N) < w; o = go(f"mix_w{w:.2f}", np.where(u, k_hist, kf))
    mix.append({"w_slow": float(w), "F_heavy": o["F_heavy"], "U3_heavy_rel_drop": o["U3_heavy_rel_drop"]})
OUT["mix"] = mix
fh = np.array([m["F_heavy"] for m in mix]); dr = np.array([m["U3_heavy_rel_drop"] for m in mix])
OUT["w_crit_Fheavy_1.1"] = float(np.interp(1.1, fh, WS)); OUT["w_crit_heavy_drop_0.3"] = float(np.interp(0.3, dr, WS))
OUT["runs"] = res

# ---------- 仮説 ----------
H = {}
v = q3(T_pot); m = float(np.median(CASES["Britain_wheel_pottery"]))
H["H121a"] = {"class": "confirmatory", "value": {"T_loss_q": v, "kappa_median": m}, "rule": "T_loss<=60 (all) and median kappa > 0.0063", "supported": bool(v[2] <= 60 and m > 0.0063)}
v = q3(T_linB); m = float(np.median(CASES["Greece_LinearB"]))
H["H121b"] = {"class": "confirmatory", "value": {"T_loss_q": v, "kappa_median": m}, "rule": "T_loss<=100 and median kappa > 0.0063", "supported": bool(v[2] <= 100 and m > 0.0063)}
v = res["geomean_of_case_medians"]["F_heavy"]
H["H121c"] = {"class": "prediction", "value": {"kappa": gm_cases, "F_heavy": v}, "rule": "F_heavy < 1.1 at log-mean kappa of loss cases incl. Tasmania", "supported": bool(v < 1.1)}
H["H121d"] = {"class": "prediction", "value": "Greek potter's wheel, 1200->1050/800 BC (faster wheel c.1050)", "rule": ">=1 skill persisting >=300 y through depopulation", "supported": True, "note": "continuation is an inference from 'faster potter's wheel'"}
H["H121e"] = {"class": "prediction", "value": {"LinearB_scribes_kappa_median": float(np.median(CASES["Greece_LinearB"])), "broad_case": "wheel persisted; kappa not estimable"},
              "rule": "narrow-practitioner kappa > broad-practitioner kappa", "supported": None, "note": "untestable quantitatively; direction consistent"}
OUT["hypotheses"] = H
OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
names = list(CASES) + ["pooled_fast", "pooled_all_equal", "geomean_of_case_medians"]
for i, k in enumerate(names):
    lo, md, hi = res[k]["kappa_q"]; axs[0].plot([lo, hi], [i, i], "-", color="C0"); axs[0].plot(md, i, "o", color="C0")
axs[0].set_xscale("log"); axs[0].set_yticks(range(len(names))); axs[0].set_yticklabels(names, fontsize=7)
axs[0].axvspan(0.0016, 0.0063, color="C3", alpha=0.2, label="108 threshold band (heavy -30% .. F_heavy 1.1)")
axs[0].set_xlabel("kappa per year (5-50-95%)"); axs[0].legend(fontsize=7); axs[0].set_title("109: kappa from historical skill losses", fontsize=9)
axs[1].plot(WS, fh, "o-", label="F_heavy (A_Ord/U3)"); axs[1].plot(WS, 1 + dr, "s--", label="1 + U3 heavy-share drop")
axs[1].axhline(1.1, color="gray", lw=0.6); axs[1].set_xlabel("weight on slow (Tasmania) kappa; rest = pooled fast cases"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_kappa_hist.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_108", d108, "geomean_cases", round(gm_cases, 5), "geomean_fast", round(gm_fast, 5))
for k in names: print(k, {a: (round(b, 4) if isinstance(b, float) else [round(x, 5) for x in b]) for a, b in res[k].items()})
for m in mix: print("mix", m["w_slow"], round(m["F_heavy"], 4), round(m["U3_heavy_rel_drop"], 3))
print("w_crit_Fheavy_1.1", round(OUT["w_crit_Fheavy_1.1"], 3), "w_crit_drop_0.3", round(OUT["w_crit_heavy_drop_0.3"], 3))
for k, h in H.items(): print(k, h["class"], h["supported"], json.dumps(h["value"], default=float)[:140])
print("runtime", round(OUT["runtime_s"], 1))
