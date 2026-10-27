"""研究150：読める確率に「見つかる確率」の時間減衰（060 の D(T)）と部数を入れ、中間層（Rosetta）と R1（フィッシュ 24 倍・MOM 10 倍）を同じ費用で比べる（open_problems 342＋344）。
148・149 と同じく 147 の割り当て直した AI・GEN の段の時刻を使い、Rosetta の寿命は 149 の UNION_1k_10k（1,000〜1 万年）。
141 の run() を写し、乱数の順を変えずに、写しの追加と発見の判定を別の乱数で足す。計算のみ・ネット接続なし。
事前登録：PREREGISTRATION.md（2026-10-27 17:36 JST、資料・計算の前）。費用の出典は sources.md。
注：P_read は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
src = open(f"{RES}/147-aigen-reassign/sim.py", encoding="utf8").read()
src = src[:src.index("MED = {}")].replace("/tmp/rep147", "/tmp/rep150")
NS = {"__file__": f"{RES}/147-aigen-reassign/sim.py", "__name__": "m150"}
exec(compile(src, "sim147_head", "exec"), NS)
R149 = json.load(open(f"{RES}/149-rosetta-lifetime/results.json", encoding="utf8"))
NS141, TT, NEW, tt_new, V141, KEYS = NS["NS"], NS["TT"], NS["NEW"], NS["tt_new"], NS["V141"], NS["KEYS"]
TT0 = dict(TT)
MEDIA = NS141["MEDIA"]; IR = [i for i, m in enumerate(MEDIA) if m[0] == "rosetta"][0]
MEDIA[IR] = MEDIA[IR][:3] + (1000, 1e4) + MEDIA[IR][5:]     # 149 の UNION_1k_10k
ns38, HSC, t_build = NS141["ns"], NS141["HSC"], NS141["t_build"]

# 発見の版（060 の D(T)=m+(1-m)ρ, m=exp(-T/τ)）。τ・ρ は標本ごと、当否は写しごと【推測】
DISC = {"none": None,
        "vault": {"tau": (100, 1000), "rho": (0.02, 0.8)},
        "half1000": {"tau": (1000 / np.log(2),) * 2, "rho": (0.0, 0.0)},
        "mem_long": {"tau": (1000, 5000), "rho": (0.02, 0.8)},
        "marker": {"tau": (100, 1000), "rho": (0.9, 0.9)}}
COPYK = ("fiche24", "mom_mf", "rosetta", "silica")
NCOP = (1, 2, 3, 5, 10, 20, 40, 80, 120, 160)

def run(var, H, disc, copies=False):
    """141 の run() と同じ乱数の順。disc＝発見の版、copies＝COPYK の媒体で部数 NCOP の P_read も出す。"""
    out = {k[0]: {"P_read_H": 0.0, "E_frac_H": 0.0} for k in MEDIA}
    cop = {k: np.zeros(len(NCOP)) for k in COPYK}
    wsum = 0.0
    for s in HSC:
        t1, t2, hs = TT[s]; n = len(t1)
        if n == 0: continue
        w = ns38["WT"][var.get("anchor", "A_XPT_S")]["mean_weights"][s] * hs; wsum += w
        rs = np.random.default_rng(20261141 + HSC.index(s))
        L = lambda lo, hi: np.exp(rs.uniform(np.log(lo), np.log(hi), n))
        stag = np.where(rs.uniform(size=n) < var.get("p_stag", 0.0), L(*var.get("stag", (300, 900))), 0.0)
        t1s, t2s = t1 + stag, t2 + stag
        tb = t_build[:n]
        m23 = var.get("m23", 300.0) * np.exp(0.8 * rs.standard_normal(n))
        salv = rs.uniform(size=n) < var.get("p_salv", 0.0)
        t_salv = L(1, 20); life_salv = np.where(rs.uniform(size=n) < 0.5, L(500, 5000), L(30, 300))
        R1 = np.where(salv, np.minimum(t_salv, t1s), t1s)
        T = {"R0": np.zeros(n), "R1": R1, "R2": t2s, "R2e": t2s + tb, "R3": t2s + m23}
        T["R3d"] = T["R3"] + (L(*var["drive"]) if var.get("drive") else 0.0)
        qM = L(0.03, 0.43)
        # 発見の τ・ρ（標本ごと、媒体で共有）：別の乱数
        rd = np.random.default_rng(20261150 + HSC.index(s))
        if disc is not None:
            lt = np.log(disc["tau"]); tau = np.exp(rd.uniform(lt[0], lt[1], n)) if lt[0] < lt[1] else np.full(n, disc["tau"][0])
            lr = disc["rho"]; rho = np.exp(rd.uniform(np.log(lr[0]), np.log(lr[1]), n)) if 0 < lr[0] < lr[1] else np.full(n, lr[0])
            Dfun = lambda ta: np.exp(-ta / tau) + (1 - np.exp(-ta / tau)) * rho
        for j, (k, nm, tier, lo, hi, unit, cap, srcm) in enumerate(MEDIA):
            life = L(lo, hi) * L(0.3, 1.0) / np.log(2) ** 0.5 * (-np.log(rs.uniform(size=n))) ** 0.5
            ta = T[tier]
            ok = (ta <= H) & (life > ta)
            frac = np.ones(n)
            if k == "piql":
                okR3 = (T["R3"] <= H) & (life > T["R3"])
                frac = np.where(okR3, 1.0, np.where(ok, qM, 0.0)); ok = ok | okR3
            rx = np.random.default_rng([20261150, HSC.index(s), j])
            if disc is not None:
                Dk = Dfun(np.where(k == "piql", np.where(okR3 & ~((ta <= H) & (life > ta)), T["R3"], ta), ta) if k == "piql" else ta)
                found = rx.uniform(size=n) < Dk
            else:
                found = np.ones(n, bool)
            ok1 = ok & found
            out[k]["P_read_H"] += w * np.mean(ok1); out[k]["E_frac_H"] += w * np.mean(ok1 * frac)
            if copies and k in COPYK:
                anyok = ok1.copy(); c = 1; vals = []
                lo_, hi_ = np.log(lo), np.log(hi)
                for nc in NCOP:
                    while c < nc:
                        lj = np.exp(rx.uniform(lo_, hi_, n)) * np.exp(rx.uniform(np.log(0.3), 0.0, n)) / np.log(2) ** 0.5 * (-np.log(rx.uniform(size=n))) ** 0.5
                        fj = (rx.uniform(size=n) < Dk) if disc is not None else True
                        anyok |= (ta <= H) & (lj > ta) & fj; c += 1
                    vals.append(np.mean(anyok))
                cop[k] += w * np.array(vals)
    for k in out:
        for kk in out[k]: out[k][kk] /= wsum
        out[k]["amount_x_P"] = [m[6] for m in MEDIA if m[0] == k][0] * out[k]["E_frac_H"]
    return out, {k: (v / wsum).tolist() for k, v in cop.items()}

HS = (1000.0, 2000.0, 5000.0, 10000.0); VN = ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW")
OUT = {}; COP = {}; CHECK = {}
for vn in VN:
    for s in ("AI", "GEN"):
        sn, sv = NEW[(vn, s)]; TT[s] = tt_new(s, sn, sv)
    for Hh in HS:
        for dn, disc in DISC.items():
            for v141, var in V141.items():
                cp = (vn == "P_MIX") and Hh in (1000.0, 2000.0, 5000.0)
                o, c = run(var, Hh, disc, copies=cp)
                pr = {k: o[k]["P_read_H"] for k in KEYS}; amt = {k: o[k]["amount_x_P"] for k in o}
                key = f"{int(Hh)}|{vn}|{dn}|{v141}"
                OUT[key] = {"P_read_H": pr, "argmax_amount": max(amt, key=amt.get)}
                if cp: COP[key] = c
                if dn == "none":
                    ref = R149["media"][f"{int(Hh)}|{vn}|UNION_1k_10k|{v141}"]["P_read_H"]
                    for k in KEYS:
                        d = abs(pr[k] - ref[k]); CHECK[f"{key}|{k}"] = d; assert d == 0.0, (key, k, d)
                    if cp:
                        for k in COPYK: assert c[k][0] == pr[k], (key, k)
    TT.update(TT0)
g = lambda H, dn, vn="P_MIX", v="main": OUT[f"{H}|{vn}|{dn}|{v}"]["P_read_H"]
allk = [(int(H), vn, dn, v) for H in HS for vn in VN for dn in DISC for v in V141]
# ---- 費用（sources.md）：Rosetta 1 枚 $25,000（2008、手作り）。フィッシュ 24 倍 1 セット＝13,000/98 → 133 枚。
SET_FICHE = int(np.ceil(13000 / 98))
COST = {"rosetta": 25000.0, "fiche_set_master_4.80": SET_FICHE * 4.80, "fiche_set_dup_1.55": SET_FICHE * 1.55}
def interp(k, key, x):  # 部数 x の P（対数の部数で線形補間）
    v = COP[key][k]; return float(np.interp(np.log(x), np.log(NCOP), v))
def kstar(key, target):
    v = COP[key]["fiche24"]
    for a, b, pa, pb in zip(NCOP[:-1], NCOP[1:], v[:-1], v[1:]):
        if v[0] >= target: return 1.0
        if pa < target <= pb: return float(np.exp(np.log(a) + (target - pa) / (pb - pa) * (np.log(b) - np.log(a))))
    return None  # 160 部でも届かない
BUD = {}
for key in COP:
    R1 = COP[key]["rosetta"][0]
    BUD[key] = {"rosetta_1": R1, "rosetta_160": COP[key]["rosetta"][-1], "fiche24_1": COP[key]["fiche24"][0], "fiche24_160": COP[key]["fiche24"][-1],
                "mom_mf_1": COP[key]["mom_mf"][0], "silica_1": COP[key]["silica"][0],
                "kstar_fiche_for_rosetta1": kstar(key, R1),
                "fiche_sets_same_budget": {c: COST["rosetta"] / COST[c] for c in ("fiche_set_master_4.80", "fiche_set_dup_1.55")},
                "fiche_at_budget_master": interp("fiche24", key, min(COST["rosetta"] / COST["fiche_set_master_4.80"], 160)),
                "fiche_at_budget_dup": interp("fiche24", key, min(COST["rosetta"] / COST["fiche_set_dup_1.55"], 160)),
                "fiche_at_budget_rosetta_x0.1": interp("fiche24", key, max(1.0, 0.1 * COST["rosetta"] / COST["fiche_set_master_4.80"]))}
# ---- 仮説 ----
r2fail = [f"{H}|{vn}|{dn}|{v}" for (H, vn, dn, v) in allk if g(H, dn, vn, v)["rosetta"] < max(g(H, dn, vn, v)["fiche48"], g(H, dn, vn, v)["piql_hand"])]
amfail = [f"{H}|{vn}|{dn}|{v}" for (H, vn, dn, v) in allk if OUT[f"{H}|{vn}|{dn}|{v}"]["argmax_amount"] != "silica"]
HY = {"H162a": {"class": "prediction", "value": g(5000, "vault")["silica"], "supported": g(5000, "vault")["silica"] < 0.5},
      "H162b": {"class": "confirmatory", "failures": amfail, "n": len(allk), "supported": not amfail},
      "H162c": {"class": "confirmatory", "failures": r2fail, "n": len(allk), "supported": not r2fail},
      "H162d": {"class": "prediction", "value": BUD["1000|P_MIX|vault|main"]["kstar_fiche_for_rosetta1"],
                "supported": (BUD["1000|P_MIX|vault|main"]["kstar_fiche_for_rosetta1"] or 1e9) <= 3},
      "H162e": {"class": "prediction", "value": g(5000, "vault")["rosetta"] / g(5000, "vault")["fiche24"],
                "none": g(5000, "none")["rosetta"] / g(5000, "none")["fiche24"], "supported": g(5000, "vault")["rosetta"] / g(5000, "vault")["fiche24"] < 1.2}}
RESULT = {"study": "150-find-decay-copies", "note": "P_read は生存確率や復興確率ではない（模型の中の量）",
          "disc_variants": {k: v for k, v in DISC.items()}, "check_max_diff_vs_149": max(CHECK.values()), "n_checks": len(CHECK),
          "media": OUT, "copies_ncop": NCOP, "copies": COP, "cost": COST, "fiche_set_sheets": SET_FICHE, "budget": BUD,
          "hypotheses": HY, "runtime_s": time.time() - T0}
json.dump(RESULT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for dn, ls in zip(DISC, ("-", "--", ":", "-.", (0, (1, 3)))):
    for k, c in zip(("silica", "rosetta", "fiche24", "mom_mf"), ("k", "tab:red", "tab:blue", "tab:green")):
        ax[0].plot(HS, [g(int(H), dn)[k] for H in HS], linestyle=ls, color=c, label=f"{k} {dn}" if dn in ("none", "vault") else None)
ax[0].set_xscale("log"); ax[0].set_xlabel("horizon H (years)"); ax[0].set_ylabel("P_read (heavy, A_XPT_S, P_MIX, main)"); ax[0].legend(fontsize=7)
for k, c in zip(COPYK, ("tab:blue", "tab:green", "tab:red", "k")):
    for dn, ls in (("none", "-"), ("vault", "--")):
        ax[1].plot(NCOP, COP[f"1000|P_MIX|{dn}|main"][k], linestyle=ls, color=c, marker="o", ms=3, label=f"{k} {dn}")
ax[1].set_xscale("log"); ax[1].set_xlabel("copies (independent sites)"); ax[1].set_ylabel("P_read at H=1000"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_find_copies.png", dpi=110)
for dn in DISC:
    print(dn, {int(H): {k: round(g(int(H), dn)[k], 3) for k in ("mom_mf", "fiche24", "fiche48", "rosetta", "piql_hand", "silica")} for H in HS})
for key in ("1000|P_MIX|none|main", "1000|P_MIX|vault|main", "5000|P_MIX|vault|main", "1000|P_MIX|marker|main", "1000|P_MIX|vault|stag_0.9", "1000|P_MIX|vault|A_Ord"):
    print(key, {k: [round(x, 3) for x in COP[key][k]] for k in COPYK}, "k*", BUD[key]["kstar_fiche_for_rosetta1"])
print("cost", COST); print({k: (v["supported"], v.get("value")) for k, v in HY.items()}); print("r2fail", len(r2fail), r2fail[:8], "amfail", len(amfail), amfail[:8])
print("checks", len(CHECK), "max", max(CHECK.values()), "runtime", round(time.time() - T0, 1))
