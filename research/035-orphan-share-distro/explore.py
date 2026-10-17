"""研究035 explore.py（事前登録外、初回実行の後に追加。判定には使わない）
(1) 名前規則による一致（M2 のみ・M3 のみ）の精度を、Debian の Homepage 欄に repo 名が含まれるかで粗く確かめる。
(2) 写しが生き残り見つかる割合 a の分岐点：他の係数を中央値に固定し、f_orph(半々・地域平均) が 034 の交差点に等しくなる a。"""
import json, lzma, re, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
from collect import stanzas, fields
hp = {}
for st in stanzas(f"{HERE}/data/Sources_trixie.xz"):
    d = fields(st)
    if d.get("Package"): hp[d["Package"]] = (d.get("Homepage", "") + " " + d.get("Vcs-Git", "") + " " + d.get("Vcs-Browser", "")).lower()
bin2src = {}
for st in stanzas(f"{HERE}/data/Packages_trixie.xz"):
    d = fields(st)
    if d.get("Package"): bin2src[d["Package"]] = d.get("Source", d["Package"]).split(" ")[0]
m = json.load(open(f"{HERE}/data/match_trixie.json"))
def prec(items, key, wkey):
    ok = tot = wok = wtot = 0; unk = 0; ex = []
    for it in items:
        if it["m1"] or not it[key]: continue
        repo = it["name"].split("/")[1].lower(); owner = it["name"].split("/")[0].lower()
        hs = " ".join(hp.get(bin2src.get(x, x), hp.get(x, "")) for x in it[key])
        if not hs.strip(): unk += 1; continue
        good = (repo in hs) or (owner in hs)
        tot += 1; ok += good; wtot += it[wkey]; wok += it[wkey] * good
        if not good and len(ex) < 8: ex.append((it["name"], it[key][:2], hs[:80]))
    return {"n_checked": tot, "n_no_homepage": unk, "precision_count": ok / max(1, tot), "precision_weight": wok / max(1e-12, wtot), "mismatch_examples": ex}
E1 = {"M2_only_dep": prec(m["U"], "m2", "v"), "M3_only_star": prec(m["star"], "m3", "stars")}
# 重みを補正した収録割合（名前規則の一致を精度で割り引く）
U = m["U"]; v = np.array([u["v"] for u in U]); S = m["star"]; st = np.array([float(s["stars"]) for s in S])
m1 = np.array([bool(u["m1"]) for u in U]); m2o = np.array([bool(u["m2"]) and not u["m1"] for u in U])
s1 = np.array([bool(s["m1"]) for s in S]); s3o = np.array([bool(s["m3"]) and not s["m1"] for s in S])
E1["dep_share_precision_adjusted"] = float((v[m1].sum() + E1["M2_only_dep"]["precision_weight"] * v[m2o].sum()) / v.sum())
E1["star_share_precision_adjusted"] = float((st[s1].sum() + E1["M3_only_star"]["precision_weight"] * st[s3o].sum()) / st.sum())
# (2) a の分岐点
import runpy, sys, io, contextlib
R = json.load(open(f"{HERE}/results.json")); BRK = R["breakeven_034"]
Ufull = {u["name"]: u for u in json.load(open(f"{HERE}/data/U_main.json")) if u["arch"]}
D1 = np.array([Ufull[u["name"]]["D1"] for u in U]); dU = m1 | np.array([bool(u["m2"]) for u in U]); dS = s1 | np.array([bool(s["m3"]) for s in S])
nreg = {r: R["mirrors"]["by_region"][r] * R["mirrors"]["f_src"] for r in ("NA", "EU", "ASIA", "SOUTH")}
med = {"v": 0.85, "p_m": 0.55, "g": 0.25, "kappa": np.sqrt(0.1), "s": {"SOUTH": 0.075, "NA": 0.25, "EU": 0.25, "ASIA": 0.25}}
def f_at(a, r, wmode):
    pdeb = 1 - (1 - med["p_m"]) ** nreg[r]
    def one(w, d, N, kap):
        inst = (1 - med["g"]) * (1 - np.exp(-N * kap * a * med["s"][r])); deb = med["v"] * pdeb * d
        return float(((1 - deb) * (1 - inst)) @ (w / w.sum()))
    fd = one(v, dU, D1, 1.0); fs = one(st, dS, st, med["kappa"])
    return {"dep": fd, "star": fs, "mix": 0.5 * fd + 0.5 * fs}[wmode]
grid = np.exp(np.linspace(np.log(1e-7), np.log(1.0), 400))
E2 = {}
for wmode in ("dep", "star", "mix"):
    for r in ("SOUTH", "EU"):
        f = np.array([f_at(a, r, wmode) for a in grid]); o = {"f_at_a_min": float(f[0]), "f_at_a_1": float(f[-1])}
        for an in ("A_XPT_S", "A_XPT_E"):
            idx = np.where(f <= BRK[an])[0]; o[f"a_star_{an}"] = float(grid[idx[0]]) if len(idx) else None
        E2[f"{wmode}_{r}"] = o
OUT = {"precision": E1, "a_breakeven_median_coefs": E2, "note": "事前登録外。a の分岐点は他の係数を中央値（v 0.85、p_m 0.55、g 0.25、κ 0.32、s_SOUTH 0.075、他 0.25）に固定した値"}
json.dump(OUT, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
