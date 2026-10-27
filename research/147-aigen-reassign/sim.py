"""研究147：AI・GEN の重い標本の状態を引き直した人口から割り当て直す（337）と、提言の順位が変わるか（338）。
計算のみ・ネット接続なし。入力は 038・141 の sim.py（写しで実行）、038・141・146 の results.json。
事前登録：PREREGISTRATION.md（2026-10-27 11:42 JST、計算の前）。seed 20261146（146 と同じ引き直し）。
注：R_tab・P_read は生存確率や復興確率ではない（模型の中の量）。"""
import json, os, runpy, shutil, time, contextlib, io
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.abspath(os.path.join(HERE, ".."))
R038 = json.load(open(f"{RES}/038-restart-state-evidence/results.json"))
R146 = json.load(open(f"{RES}/146-aigen-survivors/results.json", encoding="utf8"))
R141 = json.load(open(f"{RES}/141-medium-tier-readability/results.json", encoding="utf8"))
W = "/tmp/rep147/research"; os.makedirs(f"{W}/038-restart-state-evidence", exist_ok=True); os.makedirs(f"{W}/141-medium-tier-readability", exist_ok=True)
for d in os.listdir(RES):
    if d not in ("038-restart-state-evidence", "141-medium-tier-readability") and not os.path.exists(f"{W}/{d}"): os.symlink(f"{RES}/{d}", f"{W}/{d}")
shutil.copy(f"{RES}/038-restart-state-evidence/sim.py", f"{W}/038-restart-state-evidence/sim.py")
shutil.copy(f"{RES}/038-restart-state-evidence/results.json", f"{W}/038-restart-state-evidence/results.json")
_cwd = os.getcwd(); os.chdir(f"{W}/038-restart-state-evidence")
with contextlib.redirect_stdout(io.StringIO()):
    G = runpy.run_path(f"{W}/038-restart-state-evidence/sim.py", run_name="m147")
os.chdir(_cwd)
N = G["N"]; EVID = G["EVID"]; HEAVY = G["HEAVY"]; SCEN = G["SCEN"]; DRAWS = G["DRAWS"]; POP_W = G["POP_W"]
survivors, assign, times, recs_gen, T2s_032, weights = G["survivors"], G["assign"], G["times"], G["recs_gen"], G["T2s_032"], G["weights"]
d01, d12, U_lit, lit_mult, f_primer = G["d01"], G["d12"], G["U_lit"], G["lit_mult"], G["f_primer"]
ANCH = ("A_Ord", "A_XPT_S", "A_XPT_E"); RK = ("R_tab", "R_c4", "R_dep", "R_primer_on_tab")
CHECK = {}
SV = {}; ST = {}; PER = {}
for s in EVID:
    sv = survivors(s); st = assign(sv); ST[s] = st; SV[s] = sv; lit = SCEN[s]["lit"]
    PER[s] = recs_gen(st, times(st, sv, lit)[0], DRAWS[s], times(st, sv, lit, primer=True)[0])
def T2p_032(st, lit):
    lit_lost = U_lit < lit
    return np.select([st == 3, st == 4, st == 5, st == 2], [np.zeros(N), d12, np.where(lit_lost, d01 * lit_mult, d01) * f_primer + d12, G["W"]], default=np.zeros(N))
for s in ("AI", "GEN"):
    st = DRAWS[s]["st"]; ST[s] = st; lit = SCEN[s]["lit"]
    PER[s] = recs_gen(st, T2s_032(st, lit), DRAWS[s], T2p_032(st, lit))
def wsum(per, a, rk, mask=None):
    w = weights(a)[0]
    return float(np.mean(sum(w[s] * per[s][rk] * (1 if mask is None else mask[s]) for s in w)))
for a in ANCH:
    for rk in RK:
        d = abs(wsum(PER, a, rk) - R038["weighted_038"][a][rk]); CHECK[f"038_{rk}_{a}"] = d; assert d < 1e-12, (a, rk, d)
# ---- 146 と同じ引き直し ----
POOL = {}
for s in EVID:
    for tgt, ok in ((4, (4,)), (5, (5, 6))):
        idx = np.where(np.isin(ST[s], ok))[0]
        if len(idx) == 0: idx = np.where(np.isin(ST[s], HEAVY))[0]
        POOL[(s, tgt)] = idx
def resample(target, probs, rng):
    out = {k: np.ones(N) for k in ("s", "mcap")}; out["ntot"] = np.full(N, POP_W); out["nmax"] = np.full(N, POP_W)
    st = ST[target]; srcs = [k for k, v in probs.items() if v > 0 and len(POOL[(k, 4)]) + len(POOL[(k, 5)]) > 0]
    pp = np.array([probs[k] for k in srcs]); pp = pp / pp.sum()
    for tgt in (4, 5):
        ii = np.where(st == tgt)[0]
        if len(ii) == 0: continue
        ch = rng.choice(len(srcs), size=len(ii), p=pp)
        for j, src in enumerate(srcs):
            sel = ii[ch == j]; pool = POOL[(src, tgt)]
            if len(sel) == 0 or len(pool) == 0: continue
            pick = pool[rng.integers(0, len(pool), len(sel))]
            for k in ("s", "ntot", "nmax", "mcap"): out[k][sel] = SV[src][k][pick]
    return out
NAMES7 = ["extinct", "T3", "T3_fragile", "T2", "T1", "T0_iron", "T00"]
OUT = {}; NEW = {}
for a in ANCH:
    w = weights(a)[0]
    evw = {s: float(np.mean(w[s] * np.isin(ST[s], HEAVY))) for s in EVID}
    VAR = {"P_BIO": {"BIO": 1.0}, "P_CASC": {"CASC": 1.0}, "P_MIX": {"BIO": 0.5, "CASC": 0.5}, "P_EVIDW": evw}
    for vn, probs in VAR.items():
        rng = np.random.default_rng(20261146)
        per146 = dict(PER); perN = dict(PER); stN = dict(ST); trans = {}
        for s in ("AI", "GEN"):
            sv = resample(s, probs, rng); st = ST[s]; hv = np.isin(st, HEAVY); lit = SCEN[s]["lit"]
            # 146 の版（状態固定）
            T2h = times(st, sv, lit)[0]; per146[s] = recs_gen(st, np.where(hv, T2h, T2s_032(st, lit)), DRAWS[s])
            # 337：状態を割り当て直す
            sn = np.where(hv, assign(sv), st); stN[s] = sn
            T2n = np.where(hv, times(sn, sv, lit)[0], T2s_032(sn, lit)); T2pn = np.where(hv, times(sn, sv, lit, primer=True)[0], T2p_032(sn, lit))
            perN[s] = recs_gen(sn, T2n, DRAWS[s], T2pn)
            trans[s] = {NAMES7[i]: float(np.mean(sn[hv] == i)) for i in range(7)}
            if a == "A_XPT_S": NEW[(vn, s)] = (sn, sv)
        # 146 の再現
        h146 = wsum(per146, a, "R_tab", {s: np.isin(ST[s], HEAVY) for s in w})
        d = abs(h146 - R146["variants"][f"{a}|{vn}"]["R_tab_from_heavy"]); CHECK[f"146_{a}_{vn}"] = d; assert d < 1e-12, (a, vn, d)
        row = {"transition_of_032_heavy": trans}
        for rk in RK:
            v = wsum(perN, a, rk); row[rk] = v; row[f"{rk}_ratio_vs_038"] = v / R038["weighted_038"][a][rk]
        row["R_tab_146_total_ratio"] = (R038["weighted_038"][a]["R_tab"] - R038["weighted_038"][a]["R_tab_from_heavy"] + h146) / R038["weighted_038"][a]["R_tab"]
        row["R_tab_from_heavy_new"] = wsum(perN, a, "R_tab", {s: np.isin(stN[s], HEAVY) for s in w})
        row["R_tab_from_light_new"] = wsum(perN, a, "R_tab", {s: np.isin(stN[s], (1, 2)) for s in w})
        row["heavy_share_of_R_tab"] = row["R_tab_from_heavy_new"] / row["R_tab"]
        row["order"] = sorted(RK, key=lambda k: -row[k])
        OUT[f"{a}|{vn}"] = row
# ---- 338：141 の媒体の表を差し替えた段の時刻で ----
src141 = open(f"{RES}/141-medium-tier-readability/sim.py", encoding="utf8").read()
src141 = src141.replace('/tmp/rep141/038-restart-state-evidence/sim.py', f'{W}/038-restart-state-evidence/sim.py')
cut = src141.index("VARS = {")
NS = {"__file__": f"{W}/141-medium-tier-readability/sim.py", "__name__": "m141"}
_cwd = os.getcwd(); os.chdir(f"{W}/038-restart-state-evidence")
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src141[:cut], "sim141", "exec"), NS)
os.chdir(_cwd)
run141 = NS["run"]; TT = NS["TT"]; TT0 = dict(TT)
V141 = {"main": {}, "A_Ord": {"anchor": "A_Ord"}, "stag_0.9": {"p_stag": 0.9}}
KEYS = ("stone", "mom_eye", "mom_mf", "fiche24", "fiche48", "rosetta", "piql_hand", "piql", "silica", "mdisc")
o, _ = run141({})
for k in KEYS:
    d = abs(o[k]["P_read_H"] - R141["variants"]["main"]["media"][k]["P_read_H"]); CHECK[f"141_main_{k}"] = d; assert d == 0.0, (k, d)
d_heavy = NS["ns"]["d_Fe"]
def tt_new(s, sn, sv):
    lit = SCEN[s]["lit"]; n = np.maximum(sv["nmax"], 1.0); r = G["r_pop"]
    wait = lambda Nk: np.maximum(0.0, np.log(Nk / n) / r)
    d01e = np.where(U_lit < lit, d01 * lit_mult, d01)
    tFe = np.where(sn == 6, np.maximum(G["d_Fe"], wait(G["N_Fe"])), 0.0)
    t1 = np.where(sn >= 5, np.maximum(tFe + d01e, wait(G["N_lens"])), 0.0)
    t2 = np.where(sn >= 4, np.maximum(t1 + d12, wait(G["N_T2e"])), 0.0)
    hv = np.isin(sn, HEAVY)
    assert np.array_equal(t2, times(sn, sv, lit)[1])
    return t1[hv], t2[hv], hv.mean()
MED = {}
for vn in ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW"):
    for s in ("AI", "GEN"):
        sn, sv = NEW[(vn, s)]; TT[s] = tt_new(s, sn, sv)
    for v141, var in V141.items():
        o, tm = run141(var)
        amt = {k: o[k]["amount_x_P"] for k in o}
        MED[f"{vn}|{v141}"] = {"P_read_H": {k: o[k]["P_read_H"] for k in KEYS}, "P_K2e": {k: o[k]["P_K2e"] for k in KEYS},
                               "argmax_amount": max(amt, key=amt.get), "GEN_tier_medians": tm}
    TT.update(TT0)
for v141 in V141:
    o, _ = run141(V141[v141]); MED[f"141|{v141}"] = {"P_read_H": {k: o[k]["P_read_H"] for k in KEYS}, "P_K2e": {k: o[k]["P_K2e"] for k in KEYS}}
# ---- 仮説 ----
r_ord = OUT["A_Ord|P_MIX"]["R_tab_ratio_vs_038"]
orders_ok = all(OUT[k]["order"] == ["R_c4", "R_dep", "R_tab", "R_primer_on_tab"] for k in OUT)
sil = MED["P_MIX|main"]["P_read_H"]["silica"]
d_ok = all(MED[f"{vn}|{v}"]["P_read_H"]["mom_mf"] >= 0.85 and MED[f"{vn}|{v}"]["argmax_amount"] in ("piql", "silica")
           for vn in ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW") for v in ("main", "A_Ord"))
H = {"H159a": {"class": "prediction", "value": r_ord, "ref_146": OUT["A_Ord|P_MIX"]["R_tab_146_total_ratio"], "supported": r_ord <= 0.46},
     "H159b": {"class": "confirmatory", "value": {k: v["order"] for k, v in OUT.items()}, "supported": orders_ok},
     "H159c": {"class": "prediction", "value": sil, "supported": sil < 0.70},
     "H159d": {"class": "confirmatory", "supported": d_ok}}
RESULT = {"study": "147-aigen-reassign", "note": "R_tab・P_read は生存確率や復興確率ではない（模型の中の量）", "check": CHECK,
          "variants": OUT, "media": MED, "hypotheses": H, "runtime_s": time.time() - T0}
json.dump(RESULT, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
# ---- 図 ----
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2)); vs = ("P_BIO", "P_CASC", "P_MIX", "P_EVIDW"); xx = np.arange(4)
for k, a in enumerate(ANCH):
    ax[0].plot(xx, [OUT[f"{a}|{v}"]["R_tab_ratio_vs_038"] for v in vs], "o-", c=f"C{k}", label=f"{a} reassigned")
    ax[0].plot(xx, [OUT[f"{a}|{v}"]["R_tab_146_total_ratio"] for v in vs], "x:", c=f"C{k}", label=f"{a} 146 (state fixed)")
ax[0].set_xticks(xx); ax[0].set_xticklabels(vs); ax[0].set_ylabel("total R_tab / 038"); ax[0].legend(fontsize=7)
mk = ("mom_mf", "fiche24", "rosetta", "piql", "silica", "mdisc"); bw = 0.15
for j, vn in enumerate(("141",) + vs):
    ax[1].bar(np.arange(len(mk)) + j * bw, [MED[f"{vn}|main"]["P_read_H"][k] for k in mk], bw, label=vn)
ax[1].set_xticks(np.arange(len(mk)) + 2 * bw); ax[1].set_xticklabels(mk, fontsize=8); ax[1].set_ylabel("P_read within 1000 y (heavy, A_XPT_S)"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{HERE}/fig_reassign.png", dpi=120)
print("check max", max(CHECK.values()))
for k, v in OUT.items():
    print(k, {kk: round(v[kk], 3) for kk in ("R_tab_ratio_vs_038", "R_tab_146_total_ratio", "R_c4_ratio_vs_038", "R_primer_on_tab_ratio_vs_038", "heavy_share_of_R_tab")}, v["order"][:2],
          {s: {n: round(x, 3) for n, x in t.items() if x > 0.005} for s, t in v["transition_of_032_heavy"].items()})
for k, v in MED.items():
    print(k, {kk: round(x, 3) for kk, x in v["P_read_H"].items()}, v.get("argmax_amount"), "K2e mom_mf", round(v["P_K2e"]["mom_mf"], 3))
print({k: (v["supported"], v.get("value") if k != "H159b" else "") for k, v in H.items()}); print("runtime", round(time.time() - T0, 1))
