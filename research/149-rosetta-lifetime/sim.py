"""研究149：Rosetta 型（ニッケル）の寿命の幅を出典で置き直す（open_problems 341）。
148 と同じく 147 の割り当て直した AI・GEN の段の時刻を使い、141 の MEDIA の rosetta の (lo, hi) だけを替えて
H=1,000・2,000・5,000・10,000 年の P_read を出す。計算のみ・ネット接続なし。
事前登録：PREREGISTRATION.md（2026-10-27 15:35 JST、資料・計算の前）。出典は sources.md。
注：P_read は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
src = open(f"{RES}/147-aigen-reassign/sim.py", encoding="utf8").read()
src = src[:src.index("MED = {}")].replace("/tmp/rep147", "/tmp/rep149")
NS = {"__file__": f"{RES}/147-aigen-reassign/sim.py", "__name__": "m149"}
exec(compile(src, "sim147_head", "exec"), NS)
R148 = json.load(open(f"{RES}/148-horizon-media/results.json", encoding="utf8"))
NS141, run141, TT, NEW, tt_new, V141, KEYS = NS["NS"], NS["run141"], NS["TT"], NS["NEW"], NS["tt_new"], NS["V141"], NS["KEYS"]
TT0 = dict(TT)
MEDIA = NS141["MEDIA"]; IR = [i for i, m in enumerate(MEDIA) if m[0] == "rosetta"][0]; ROW0 = MEDIA[IR]
assert (ROW0[3], ROW0[4]) == (2000, 1e4)
# 寿命の版（lo, hi 年）。根拠は sources.md。
LIFE = {
 "L141_press2000": (2000, 1e4),   # 141 の幅。BBC 2000・Mercury News 2000 の「2,000〜10,000 年」と一致【事実】
 "N98_1k_5k":      (1000, 5000),  # Norsam の COO（Rose 1998 の伝聞）「1-5000 year medium」を 1,000〜5,000 と読む【推測：読み】
 "N98_literal":    (1, 5000),     # 同じ文を字義どおり 1〜5,000 年と読む【推測：読み】
 "B08_2000":       (2000, 2000),  # Rosetta Project ブログ 2008・Wikipedia の「2,000 年」を 1 点で
 "UNION_1k_10k":   (1000, 1e4),   # 開けた主張の和（字義の 1 年を除く）＝主の出典の幅【推測：和の取り方】
 "HI_10k_100k":    (1e4, 1e5),    # H161d 用の仮の上振れ（出典なし）【推測】
}
HS = (1000.0, 2000.0, 5000.0, 10000.0); VN = ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW")
OUT = {}; CHECK = {}
for vn in VN:
    for s in ("AI", "GEN"):
        sn, sv = NEW[(vn, s)]; TT[s] = tt_new(s, sn, sv)
    for Hh in HS:
        NS141["H"] = Hh
        for ln, (lo, hi) in LIFE.items():
            MEDIA[IR] = ROW0[:3] + (lo, hi) + ROW0[5:]
            for v141, var in V141.items():
                o, _ = run141(var)
                pr = {k: o[k]["P_read_H"] for k in KEYS}
                amt = {k: o[k]["amount_x_P"] for k in o}
                OUT[f"{int(Hh)}|{vn}|{ln}|{v141}"] = {"P_read_H": pr, "argmax_amount": max(amt, key=amt.get),
                    "rank_rosetta_vs_silica": "rosetta" if pr["rosetta"] > pr["silica"] else "silica"}
                if ln == "L141_press2000":
                    ref = R148["media"][f"{int(Hh)}|{vn}|{v141}"]["P_read_H"]
                    for k in KEYS:
                        d = abs(pr[k] - ref[k]); CHECK[f"{int(Hh)}_{vn}_{v141}_{k}"] = d; assert d == 0.0, (Hh, vn, v141, k, d)
                else:  # 他の媒体は動かない
                    base = OUT[f"{int(Hh)}|{vn}|L141_press2000|{v141}"]["P_read_H"]
                    for k in KEYS:
                        if k != "rosetta": assert pr[k] == base[k], (k, ln)
        MEDIA[IR] = ROW0
    TT.update(TT0)
NS141["H"] = 1000.0
g = lambda H, ln, vn="P_MIX", v="main": OUT[f"{H}|{vn}|{ln}|{v}"]["P_read_H"]
allc = [(int(H), vn, v) for H in HS for vn in VN for v in V141]
srcL = ("L141_press2000", "N98_1k_5k", "B08_2000", "UNION_1k_10k")
b_fail = [f"{H}|{vn}|{ln}|{v}" for (H, vn, v) in allc if H >= 5000 for ln in srcL if g(H, ln, vn, v)["rosetta"] > g(H, ln, vn, v)["silica"]]
c_diff = {ln: abs(g(1000, ln)["rosetta"] - g(1000, "L141_press2000")["rosetta"]) for ln in LIFE}
HY = {
 "H161a": {"class": "prediction", "value_UNION": g(5000, "UNION_1k_10k")["rosetta"], "values": {ln: g(5000, ln)["rosetta"] for ln in LIFE},
           "supported": g(5000, "UNION_1k_10k")["rosetta"] >= 0.90},
 "H161b": {"class": "confirmatory", "failures": b_fail, "n": len([1 for (H, _, _) in allc if H >= 5000]) * len(srcL), "supported": len(b_fail) == 0},
 "H161c": {"class": "confirmatory", "diff_at_1000_P_MIX_main": c_diff,
           "supported": max(c_diff[ln] for ln in srcL) < 0.05},
 "H161d": {"class": "prediction", "rosetta": g(2000, "HI_10k_100k")["rosetta"], "silica": g(2000, "HI_10k_100k")["silica"],
           "supported": g(2000, "HI_10k_100k")["rosetta"] > g(2000, "HI_10k_100k")["silica"]},
}
# R2 の中の最大（148 の H160e と同じ問い）が版で保たれるか
r2_fail = [f"{H}|{vn}|{ln}|{v}" for (H, vn, v) in allc for ln in LIFE
           if g(H, ln, vn, v)["rosetta"] < max(g(H, ln, vn, v)["fiche48"], g(H, ln, vn, v)["piql_hand"])]
crossover = {ln: next((int(H) for H in HS if g(int(H), ln)["silica"] > g(int(H), ln)["rosetta"]), None) for ln in LIFE}
RESULT = {"study": "149-rosetta-lifetime", "note": "P_read は生存確率や復興確率ではない（模型の中の量）",
          "life_variants": LIFE, "check_max_diff_vs_148": max(CHECK.values()), "n_checks": len(CHECK),
          "media": OUT, "hypotheses": HY, "R2_rosetta_not_max": r2_fail, "silica_overtakes_rosetta_first_H_P_MIX_main": crossover,
          "argmax_amount_all_silica": all(v["argmax_amount"] == "silica" for v in OUT.values()), "runtime_s": time.time() - T0}
json.dump(RESULT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for ln in LIFE:
    ax.plot(HS, [g(int(H), ln)["rosetta"] for H in HS], marker="o", label=f"rosetta {ln}")
ax.plot(HS, [g(int(H), "L141_press2000")["silica"] for H in HS], "k--", marker="s", label="silica")
ax.set_xscale("log"); ax.set_xlabel("horizon H (years)"); ax.set_ylabel("P_read within H (heavy, A_XPT_S, P_MIX, main)")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{HERE}/fig_rosetta_life.png", dpi=110)
for ln in LIFE:
    print(ln, [round(g(int(H), ln)["rosetta"], 3) for H in HS], "| stag_0.9", [round(g(int(H), ln, "P_MIX", "stag_0.9")["rosetta"], 3) for H in HS])
print("silica", [round(g(int(H), "L141_press2000")["silica"], 3) for H in HS])
print({k: (v["supported"]) for k, v in HY.items()}, "H161a", HY["H161a"]["value_UNION"], "c", c_diff)
print("crossover", crossover, "R2 fail", len(r2_fail), r2_fail[:6], "b_fail", b_fail[:6])
print("checks", len(CHECK), "max diff", max(CHECK.values()), "argmax silica", RESULT["argmax_amount_all_silica"], "runtime", round(time.time() - T0, 1))
