"""研究122：1 点あたりに動く接点の数（ops/pt）を最小にする読み取り回路（open_problems 287）。
計算のみ、ネット接続なし（資料は sources/ に保存済み）。事前登録：PREREGISTRATION.md（2026-10-25 09:40 JST）。
121 の sim.py を「# ---------- 仮説」の前まで exec（121 の p_vac と表を差 0 で再現、assert）。
設計ごとに継電器の状態を 1 点ずつ進める離散の模擬で、各継電器が閉じる回数（0→1 の回数＝1 サイクル）を数える。
事実（sources/）：Z3 は約 2,000 継電器のうち 1,400 が記憶（別の箇所では 2,600）、5〜10 Hz。Model 33 の紙テープの読み取りと端末の間は
『completely mechanical』で時計と 8 ビットを金属のレバーで並列に渡す。Model 28 の transmitter-distributor はテープを電気の
インパルスに変える自己完結の機械。設計の回路・カムの扱い・寿命の値は【推測】（120 の値を継ぐ）。
"""
import json, time, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

t0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
P121 = os.path.abspath(os.path.join(HERE, "..", "121-tube-wear-symmetric", "sim.py"))
src = open(P121, encoding="utf8").read(); cut = src.index("\n# ---------- 仮説")
G = {"__file__": P121, "__name__": "sim121"}
exec(compile(src[:cut], P121, "exec"), G)
R121 = json.load(open(os.path.join(os.path.dirname(P121), "results.json"), encoding="utf8"))
dP = max(abs(G["PV"][k] - R121["p_vac_versions"][k]) for k in R121["p_vac_versions"])
assert dP < 1e-12, dP
dT = max(abs(G["TAB"][t]["F_heavy"] - R121["table"][t]["F_heavy"]) for t in R121["table"])
G120 = G["G"]
run, KAPS, fe_d, le_d, t2e_two, L_bulk, PPREC = (G[k] for k in ("run", "KAPS", "fe_d", "le_d", "t2e_two", "L_bulk", "PPREC"))
prelay, B120, DATA, demand = G120["prelay"], G120["B"], G120["DATA"], G120["demand"]
P_TUBE = G["PT"]["main"][0]
OUT = {"study": "122-min-ops-reader", "check_121_pvac_maxdiff": dP, "check_121_table_F_maxdiff_newrng": dT}

# ---------- 1. ビット列：一様乱数（攪拌・圧縮済み）と、攪拌なしの文書の画像 ----------
rng = np.random.default_rng(20261122)
NB = 400_000
bits_rand = rng.integers(0, 2, NB).astype(np.int8)
def page_bits():
    txt = open(os.path.join(HERE, "..", "..", "log.md"), encoding="utf8").read()
    txt = "".join(c if ord(c) < 128 else "#" for c in txt)[:20000]
    lines = [txt[i:i + 90] for i in range(0, len(txt), 90)]
    img = Image.new("1", (640, 12 * len(lines) + 20), 1); d = ImageDraw.Draw(img); f = ImageFont.load_default()
    for i, l in enumerate(lines): d.text((10, 10 + 12 * i), l, fill=0, font=f)
    a = 1 - np.array(img, dtype=np.int8)        # 黒＝1
    return a.reshape(-1)[:NB]
bits_page = page_bits()
OUT["bits"] = {"rand_mean": float(bits_rand.mean()), "page_mean": float(bits_page.mean()), "N": NB}

def cycles(state):  # state: (T, n) の 0/1。0→1 の回数を継電器ごとに
    s = np.vstack([np.zeros((1, state.shape[1]), np.int8), state.astype(np.int8)])
    return (np.diff(s, axis=0) == 1).sum(axis=0)

# ---------- 2. 設計の模擬 ----------
def design_min(b):
    """D1：感知の継電器 1 個が連続の 1 の間は閉じたまま（NRZ）。時刻・送りは機械（カム）。
    出力の磁石はカムの接点で 1 の点ごとに打つ（カムは負荷を切る：cam_ops）。"""
    r = cycles(b[:, None]); return {"relays": 1, "relay_ops": int(r.sum()), "cam_ops": int(b.sum()), "points": len(b)}
def design_rz(b):
    """D2：感知の継電器を点ごとに戻す（RZ）。出力の磁石は継電器の接点が直接打つ。"""
    return {"relays": 1, "relay_ops": int(b.sum()), "cam_ops": 0, "points": len(b)}
def design_realistic(b, copies=5, word=8, counter="relay"):
    """D3：時刻のトラック（点ごとに明暗が入れ替わる）を光導電セル＋継電器で読み、5 重の写しの感知の継電器（NRZ）、
    多数決の継電器、8 ビットのシフトレジスタ（主従 2 個/段）、語ごとに出力の保持の継電器 8 個をセットしリセット、
    ビットの数え（継電器の 2 進の 3 段×2 個、または歩進の切替器の磁石 1 個）。1 出力ビット＝copies 個の点（＋時刻 1 点）。"""
    nbit = len(b) // copies; truth = b[:nbit].copy(); X = np.repeat(truth[:, None], copies, axis=1)   # 同じビットの copies 個の写し
    # 写しの読み誤り：各写しの点を 1% で反転（多数決の働きを見るため）【推測】
    flip = rng.random(X.shape) < 0.01; X = X ^ flip.astype(np.int8)
    ops = {}
    clk = (np.arange(nbit) % 2).astype(np.int8)                        # 時刻のトラック：明暗交互
    ops["clock_sense"] = int(cycles(clk[:, None]).sum())
    ops["copy_sense"] = int(cycles(X).sum())
    maj = (X.sum(axis=1) * 2 > copies).astype(np.int8)
    ops["majority"] = int(cycles(maj[:, None]).sum())
    # シフトレジスタ：各ビット時刻に master_k ← slave_{k-1}、slave_k ← master_k。語の終わりで全段を 0 に
    T2 = 2 * nbit; M = np.zeros((T2, word), np.int8); S = np.zeros((T2, word), np.int8)
    m = np.zeros(word, np.int8); s = np.zeros(word, np.int8)
    for t in range(nbit):
        if t % word == 0: m[:] = 0; s[:] = 0
        m = np.concatenate([[maj[t]], s[:-1]]).astype(np.int8); M[2 * t] = m; S[2 * t] = s
        s = m.copy(); M[2 * t + 1] = m; S[2 * t + 1] = s
    ops["shift_register"] = int(cycles(M).sum() + cycles(S).sum())
    nw = nbit // word
    latch = maj[:nw * word].reshape(nw, word)
    ops["output_latch"] = int(latch.sum())                             # 語ごとに 1 の桁だけ閉じ、次の語の前に開く
    if counter == "relay":
        cnt = np.array([[(t % word) >> k & 1 for k in range(3)] for t in range(nbit)], np.int8)
        ops["bit_counter"] = int(2 * cycles(cnt).sum())                # 段ごとに 2 個
        ncount = 6
    else:
        ops["bit_counter"] = nbit                                       # 歩進の切替器の磁石が 1 ビットに 1 回
        ncount = 1
    ops["word_transfer"] = nw                                          # 語の転送の継電器 1 個
    relays = 1 + copies + 1 + 2 * word + word + ncount + 1
    tot = sum(ops.values())
    return {"relays": relays, "relay_ops": int(tot), "cam_ops": 0, "points": nbit * copies,
            "points_incl_clock": nbit * (copies + 1), "breakdown": ops, "errors_after_majority": int((maj != truth).sum()), "raw_errors_copy0": int(flip[:, 0].sum())}
def design_stepper_distributor(b, copies=5, word=8):
    """D4：D3 からシフトレジスタを外し、多数決の継電器の接点を歩進の切替器の腕で 8 個の保持の継電器に配る。"""
    r = design_realistic(b, copies, word, counter="stepper")
    bd = dict(r["breakdown"]); bd.pop("shift_register")
    r["breakdown"] = bd; r["relay_ops"] = int(sum(bd.values())); r["relays"] -= 2 * word
    return r

DES = {}
for dn, b in (("rand", bits_rand), ("page", bits_page)):
    DES[f"D1_min|{dn}"] = design_min(b)
    DES[f"D2_rz|{dn}"] = design_rz(b)
    DES[f"D3_realistic|{dn}"] = design_realistic(b)
    DES[f"D4_stepper|{dn}"] = design_stepper_distributor(b)
    DES[f"D3_nocopies|{dn}"] = design_realistic(b, copies=1)
for k, v in DES.items():
    v["ops_per_point"] = v["relay_ops"] / v["points"]
    v["ops_per_point_incl_cam"] = (v["relay_ops"] + v["cam_ops"]) / v["points"]
DES["D0_120_assumption"] = {"relays": 50, "ops_per_point": 25.0, "ops_per_point_incl_cam": 25.0}
OUT["designs"] = DES

# 汎用の経路が半分の p_arm を保つ ops/pt の上限：px*k/L <= M50 - N0
px = DATA["reel_120GB"]
k_half = {f"L{L:.0e}": float(B120["M50"] * L / px) for L in (1e5, 1e6, 1e7)}
OUT["ops_per_point_for_S_half_armature"] = k_half

# ---------- 3. p_relay・p_vac・F_heavy の置き直し ----------
def mod(**kw): d = dict(B120); d.update(kw); return d
def from_design(key, **kw):
    v = DES[key]; C = v["relays"]; k = v["ops_per_point_incl_cam" if kw.pop("cam", False) else "ops_per_point"]
    return mod(C=C, a=k / C, **kw)
VERS = {
 "120_main": B120,
 "D3_realistic": from_design("D3_realistic|rand"),
 "D4_stepper": from_design("D4_stepper|rand"),
 "D1_min": from_design("D1_min|rand"),
 "D1_min_incl_cam": from_design("D1_min|rand", cam=True),
 "D1_min_dry_L1e7": from_design("D1_min|rand", L_arm=1e7),
 "D3_page_noscramble": from_design("D3_realistic|page"),
 "D3_L1e5": from_design("D3_realistic|rand", L_arm=1e5),
 "D3_noreed": from_design("D3_realistic|rand", p_NiFe=0.0),
 "D1_noreed": from_design("D1_min|rand", p_NiFe=0.0),
 "D3_pess": from_design("D3_realistic|rand", L_arm=1e5, M50=3e4, p_NiFe=0.3),
}
PR = {k: prelay(v) for k, v in VERS.items()}
PV = {k: 1 - (1 - v["p_relay"]) * (1 - P_TUBE) for k, v in PR.items()}
assert abs(PV["120_main"] - R121["p_vac_versions"]["main"]) < 1e-12
OUT["p_relay_versions"] = PR; OUT["p_vac_versions"] = PV; OUT["p_tube"] = P_TUBE
TAB = {}
for vn in ("120_main", "D3_realistic", "D1_min", "D3_noreed", "D1_noreed", "D3_pess"):
    fT = t2e_two(100.0, L_bulk, PPREC * PV[vn])
    for kn in KAPS:
        TAB[f"{kn}|{vn}"] = run(f"{kn}|{vn}", KAPS[kn], {"Fe": fe_d, "lens": le_d, "T2e": fT}, store=False)
OUT["table"] = TAB

# 走査：ops/pt と p_relay（C は D3 の数、主の他の値）
ks = np.logspace(-2, 1.5, 36); CD3 = DES["D3_realistic|rand"]["relays"]
SC = {"main": [prelay(mod(C=CD3, a=k / CD3))["p_relay"] for k in ks],
      "noreed": [prelay(mod(C=CD3, a=k / CD3, p_NiFe=0.0))["p_relay"] for k in ks]}
OUT["scan_k"] = list(map(float, ks)); OUT["scan_prelay"] = SC

# ---------- 仮説 ----------
H = {}
H["H134a"] = {"class": "confirmatory", "supported": False,
  "value": "Teleprinter・Teletype Model 33・Model 28 の記事に selector magnet（1 個の電磁石で直列を並列に直す）の記述なし。近いのは Model 33 の『読み取りと端末の間は completely mechanical、時計と 8 ビットを金属のレバーで並列』と、start-stop の符号が電気機械の設計と結びついていたという記述"}
H["H134b"] = {"class": "confirmatory", "supported": True, "value": "Z3：Elements 『Around 2,000 relays (1,400 for the memory)』（infobox は 2,600）"}
k1 = DES["D1_min|rand"]["ops_per_point"]
H["H134c"] = {"class": "prediction", "value": k1, "rule": "0.2<=ops/pt<=1.5", "supported": bool(0.2 <= k1 <= 1.5)}
k3 = DES["D3_realistic|rand"]["ops_per_point"]
H["H134d"] = {"class": "prediction", "value": k3, "rule": "1<=ops/pt<=8（25 の 1/3 以下）", "supported": bool(1 <= k3 <= 8)}
fh = TAB["slow_Tasmania|D3_realistic"]["F_heavy"]; p3 = PR["D3_realistic"]["p_relay"]
H["H134e"] = {"class": "prediction", "value": {"p_relay": p3, "F_heavy_slow": fh},
  "rule": "0.36<=p_relay<=0.65 かつ 1.38<=F<=1.58", "supported": bool(0.36 <= p3 <= 0.65 and 1.38 <= fh <= 1.58)}
H["H134f"] = {"class": "descriptive", "supported": None, "value": "設計ごとの継電器数と動作数の表"}
OUT["hypotheses"] = H; OUT["runtime_s"] = time.time() - t0

# ---------- 図 ----------
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
axs[0].plot(ks, SC["main"], label="p_relay (armature + reed)"); axs[0].plot(ks, SC["noreed"], label="p_relay, no reed")
for key, c in (("D1_min|rand", "g"), ("D4_stepper|rand", "m"), ("D3_realistic|rand", "b")):
    axs[0].axvline(DES[key]["ops_per_point"], color=c, lw=0.8, ls="--"); axs[0].text(DES[key]["ops_per_point"] * 1.05, 0.02, key.split("|")[0], fontsize=7, color=c, rotation=90)
axs[0].axvline(25, color="r", lw=0.8); axs[0].text(26, 0.5, "120 assumption", fontsize=7, color="r", rotation=90)
axs[0].set_xscale("log"); axs[0].set_xlabel("relay operations per point read"); axs[0].set_ylabel("p_relay"); axs[0].legend(fontsize=7)
axs[0].set_title("122: reader circuit design vs relay-path probability (one 120 GB reel)", fontsize=9)
vn = [t.split("|")[1] for t in TAB if t.startswith("slow_Tasmania")]; x = np.arange(len(vn))
for j, kn in enumerate(KAPS):
    axs[1].bar(x + (j-1)*0.27, [TAB[f"{kn}|{w}"]["F_heavy"] for w in vn], 0.27, label=kn)
axs[1].set_xticks(x); axs[1].set_xticklabels([f"{w}\n{PV[w]:.3f}" for w in vn], fontsize=6, rotation=30)
axs[1].axhline(1.1, color="k", lw=0.5); axs[1].axhline(1.5, color="k", lw=0.5, ls=":"); axs[1].legend(fontsize=7); axs[1].set_ylabel("F_heavy (A_Ord/U3)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_min_ops.png"), dpi=120); plt.close(fig)
json.dump(OUT, open(os.path.join(HERE, "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
print("check_121", dP, "F newrng diff", dT)
print("bits", OUT["bits"])
for k, v in DES.items(): print("design", k, "relays", v["relays"], "ops/pt", round(v["ops_per_point"], 4), "incl_cam", round(v["ops_per_point_incl_cam"], 4), v.get("breakdown", ""), "err", v.get("errors_after_majority", ""))
print("k_half", k_half)
for k, v in PR.items(): print("p_relay", k, {kk: round(vv, 4) if vv < 10 else f"{vv:.3g}" for kk, vv in v.items()}, "p_vac", round(PV[k], 4))
for t, o in TAB.items(): print(t, "F_heavy", round(o["F_heavy"], 4))
for kk, hh in H.items(): print(kk, hh["class"], hh["supported"], str(hh.get("value"))[:200])
print("runtime", round(OUT["runtime_s"], 1))
