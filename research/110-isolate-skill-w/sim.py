"""研究110：孤立した小さな集団の技能の喪失で遅い κ の重み w を置く（open_problems 262）。
計算のみ、ネット接続なし。事前登録：PREREGISTRATION.md（2026-10-24 09:42 JST）。
109 の sim.py を「# ---------- 仮説」の前まで exec（108→104→…の乱数と 109 の全版を再現、assert）。新しい乱数は seed 20261110。
κ = ln R / T_loss、R ~ logU(1.1, 10)【推測】（108・109 と同じ）。遅い型＝κ 中央値 ≤ 0.0063／年。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P109 = os.path.abspath(os.path.join(HERE, "..", "109-kappa-historical", "sim.py"))
src = open(P109, encoding="utf8").read(); cut = src.index("# ---------- 仮説")
G = {"__file__": P109, "__name__": "sim109"}
exec(compile(src[:cut], P109, "exec"), G)
R109 = json.load(open(os.path.join(os.path.dirname(P109), "results.json"), encoding="utf8"))
d109 = max(abs(G["res"][t]["F_heavy"] - R109["runs"][t]["F_heavy"]) for t in R109["runs"])
assert d109 < 1e-12, d109
run, N, K, k_hist, kf = G["run"], G["N"], G["K"], G["k_hist"], G["kf"]
g = np.random.default_rng(20261110)
q3 = lambda v: [float(np.quantile(v, p)) for p in (0.05, 0.5, 0.95)]
lnR = lambda: g.uniform(np.log(1.1), np.log(10), N)
SLOW = 0.0063
OUT = {"study": "110-isolate-skill-w", "seed": 20261110, "check_109_runs_reproduced_maxdiff": d109}

# ---------- 事例の T_loss（資料の年、幅は【推測】の一様） ----------
s_inu = g.uniform(1600, 1700, N)                       # 「17 世紀ごろ」孤立
T_inu_boat = g.uniform(0, 1, N) * (1818 - s_inu)       # 1818 年の接触までに喪失、時点は一様【推測】
T_inu_boat = np.maximum(T_inu_boat, 1.0)
T_inu_bow = np.maximum(g.uniform(0, 1, N) * (g.uniform(1840, 1870, N) - s_inu), 1.0)
T_tonga = g.uniform(2900, 3280, N) - 1550              # 定住 3000〜3180±100 BP → 1550 BP
T_easter = g.uniform(1600, 1800, N) - g.uniform(1150, 1250, N)
T_andaman = g.uniform(1000, 26000, N)                  # 感度のみ：喪失が起きたかも年も不明【推測】
TL = {"Inughuit_boats": T_inu_boat, "Inughuit_bows_leisters": T_inu_bow, "Tonga_pottery": T_tonga,
      "EasterIsland_seaworthy_vessels": T_easter, "Andaman_fire_making": T_andaman}
KAP = {k: lnR() / v for k, v in TL.items()}
KAP["Tasmania_108"] = k_hist
for k in ["Britain_coin", "Britain_wheel_pottery", "Greece_LinearB", "Greece_monumental_stone"]:
    KAP[k] = G["CASES"][k]
cls = {k: ("slow" if np.median(v) <= SLOW else "fast") for k, v in KAP.items()}
OUT["cases"] = {k: {"kappa_q": q3(v), "T_loss_q": (q3(TL[k]) if k in TL else None), "type": cls[k],
                    "P_kappa_le_slow": float(np.mean(v <= SLOW))} for k, v in KAP.items()}
CAUSE = {"Inughuit_boats": "isolation", "Inughuit_bows_leisters": "isolation", "Tonga_pottery": "demand/material",
         "EasterIsland_seaworthy_vessels": "resource", "Andaman_fire_making": "unknown", "Tasmania_108": "isolation(contested)",
         "Britain_coin": "state/demand", "Britain_wheel_pottery": "state/demand", "Greece_LinearB": "state/demand", "Greece_monumental_stone": "state/demand"}

# ---------- 版：どの事例を数えるか ----------
STATE4 = ["Britain_coin", "Britain_wheel_pottery", "Greece_LinearB", "Greece_monumental_stone"]
VERS = {
 "V_main_isolates_plus_state": ["Inughuit_boats", "Tonga_pottery", "EasterIsland_seaworthy_vessels", "Tasmania_108"] + STATE4,
 "V_isolates_only": ["Inughuit_boats", "Tonga_pottery", "EasterIsland_seaworthy_vessels", "Tasmania_108"],
 "V_isolation_cause_only": ["Inughuit_boats", "Tasmania_108"],
 "V_isolation_cause_plus_state": ["Inughuit_boats", "Tasmania_108"] + STATE4,
 "V_episode_split_inughuit": ["Inughuit_boats", "Inughuit_bows_leisters", "Tonga_pottery", "EasterIsland_seaworthy_vessels", "Tasmania_108"] + STATE4,
 "V_plus_andaman": ["Inughuit_boats", "Tonga_pottery", "EasterIsland_seaworthy_vessels", "Tasmania_108", "Andaman_fire_making"] + STATE4,
}
res = {}
def go(tag, kap):
    o = run("c110_" + tag, kap)
    res[tag] = {"F_heavy": o[K]["F_heavy"], "F_total": o[K]["F_total"], "U3_heavy_rel_drop": o["U3_heavy_rel_drop"], "kappa_q": q3(kap)}
    return res[tag]
W = {}
for v, cs in VERS.items():
    s = sum(cls[c] == "slow" for c in cs); f = len(cs) - s
    wpost = g.beta(0.5 + s, 0.5 + f, N)
    W[v] = {"n": len(cs), "slow": int(s), "w_post_q": q3(wpost)}
    # 登録の混合：遅い側は Tasmania の κ、速い側は 109 の速い 4 件の混合
    u = g.random(N) < wpost
    W[v]["registered_mix"] = go(v + "_reg", np.where(u, k_hist, kf))
    # 別版（登録外）：事例そのものの κ を等しい重みで混ぜる
    pick = g.integers(0, len(cs), N)
    W[v]["empirical_pool"] = go(v + "_emp", np.choose(pick, [KAP[c] for c in cs]))
# 点の w（事後の中央値）で
for v in ["V_main_isolates_plus_state", "V_isolation_cause_only"]:
    wm = W[v]["w_post_q"][1]; u = g.random(N) < wm
    W[v]["point_w"] = go(v + "_pt", np.where(u, k_hist, kf))
OUT["versions"] = W; OUT["runs"] = res; OUT["cause"] = CAUSE

# ---------- 仮説 ----------
H = {}
m = float(np.median(KAP["Inughuit_boats"])); tq = q3(T_inu_boat)
H["H122a"] = {"class": "confirmatory", "value": {"T_loss_q": tq, "P_T_le_100": float(np.mean(T_inu_boat <= 100)), "kappa_median": m},
              "rule": "median T_loss<=100 and median kappa>0.0063", "supported": bool(tq[1] <= 100 and m > SLOW),
              "note": "T only bounded above (loss by 1818); uniform timing is an assumption"}
iso = ["Inughuit_boats", "Tonga_pottery", "EasterIsland_seaworthy_vessels"]
nf = sum(cls[c] == "fast" for c in iso)
H["H122b"] = {"class": "confirmatory", "value": {c: cls[c] for c in iso}, "rule": ">= half of dated isolate losses (excl. Tasmania) fast", "supported": bool(nf >= len(iso) / 2)}
wq = W["V_main_isolates_plus_state"]["w_post_q"]
H["H122c"] = {"class": "prediction", "value": wq, "rule": "0.022<=median w<0.336", "supported": bool(0.022 <= wq[1] < 0.336)}
fh = W["V_main_isolates_plus_state"]["registered_mix"]["F_heavy"]
H["H122d"] = {"class": "prediction", "value": fh, "rule": "1.1<=F_heavy<=2.0 (registered mix, main)", "supported": bool(1.1 <= fh <= 2.0)}
H["H122e"] = {"class": "descriptive", "value": CAUSE, "rule": "losses concentrated in narrow-practitioner or demand-lost skills", "supported": None}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
names = list(KAP)
for i, k in enumerate(names):
    lo, md, hi = q3(KAP[k]); c = "C3" if cls[k] == "slow" else "C0"
    axs[0].plot([lo, hi], [i, i], "-", color=c); axs[0].plot(md, i, "o", color=c)
axs[0].axvline(SLOW, color="gray", ls="--", lw=0.8); axs[0].set_xscale("log"); axs[0].set_yticks(range(len(names))); axs[0].set_yticklabels(names, fontsize=7)
axs[0].set_xlabel("kappa per year (5-50-95%); red = slow type"); axs[0].set_title("110: kappa from isolate + state cases", fontsize=9)
vs = list(VERS); x = np.arange(len(vs))
axs[1].bar(x - 0.2, [W[v]["registered_mix"]["F_heavy"] for v in vs], 0.4, label="registered mix (Tasmania slow / 109 fast)")
axs[1].bar(x + 0.2, [W[v]["empirical_pool"]["F_heavy"] for v in vs], 0.4, label="empirical pool of case kappas")
axs[1].axhline(1.1, color="gray", lw=0.6); axs[1].set_xticks(x); axs[1].set_xticklabels([v.replace("V_", "") for v in vs], rotation=30, ha="right", fontsize=7)
axs[1].set_ylabel("F_heavy (A_Ord/U3)"); axs[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_isolate_w.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_109", d109)
for k in names: print(k, cls[k], [round(x, 5) for x in q3(KAP[k])], "P_slow", round(float(np.mean(KAP[k] <= SLOW)), 3))
for v in vs:
    print(v, "n", W[v]["n"], "slow", W[v]["slow"], "w", [round(x, 3) for x in W[v]["w_post_q"]],
          "F_reg", round(W[v]["registered_mix"]["F_heavy"], 4), "drop_reg", round(W[v]["registered_mix"]["U3_heavy_rel_drop"], 3),
          "F_emp", round(W[v]["empirical_pool"]["F_heavy"], 4), "drop_emp", round(W[v]["empirical_pool"]["U3_heavy_rel_drop"], 3))
for v in ["V_main_isolates_plus_state", "V_isolation_cause_only"]: print("point_w", v, round(W[v]["point_w"]["F_heavy"], 4))
for k, h in H.items(): print(k, h["class"], h["supported"], json.dumps(h["value"], default=float)[:160])
print("runtime", round(OUT["runtime_s"], 1))
