"""研究036 sim.py：≥20,000 星の repo の Debian・Homebrew・nixpkgs での収録と、収録外の中身（文章型か）を測り、
星側の f_orph を Homebrew の install 数を入れて置き直し、034 に戻す。
035 の sim.py を一時ディレクトリで再実行して同じ係数の組（K=4,000）と依存側の f_orph を使う。追加の係数は seed 20261036。"""
import json, re, os, sys, time, shutil, tempfile, runpy, io, contextlib
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T0 = time.time(); HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(HERE)
R = {"seed": 20261036}
def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y)); return float(np.corrcoef(rx, ry)[0, 1])

# ---------- 1. 収録（測定） ----------
M = json.load(open(f"{HERE}/data/match_star.json")); S = M["star"]
st = np.array([float(s["stars"]) for s in S]); w = st / st.sum()
d_deb = np.array([s["deb"] for s in S]); d_brew = np.array([bool(s["brew"]) for s in S]); d_nix = np.array([bool(s["nix"]) for s in S])
d_m4 = np.array([bool(s["m4"]) for s in S]); d_purl = np.array([bool(s["nix_purl"]) for s in S])
inst = np.array([float(s["installs"]) for s in S])
d_any = d_deb | d_brew | d_nix
CONT_LANG = {None, "Markdown", "TeX"}; SOFT_LANG = {None, "Markdown", "HTML", "TeX", "Jupyter Notebook"}
KW = re.compile(r'awesome|tutorial|interview|roadmap|book|cheatsheet|course|guide|resources|list', re.I)
def is_content(s):
    m = s["meta"]; lang = m["lang"]
    if lang in CONT_LANG: return True
    txt = " ".join([s["name"].split('/')[1], m["desc"], " ".join(m["topics"])])
    return bool(KW.search(txt)) and lang in SOFT_LANG
cont = np.array([is_content(s) for s in S])
def share(x, ww=w): return float(ww[x].sum())
R["n"] = {"repos": len(S), "formula": M["n_formula"], "cask": M["n_cask"], "nix": M["n_nix"],
          "an_formula_total": M["an_formula_total"], "an_cask_total": M["an_cask_total"]}
R["coverage_star_weight"] = {"debian": share(d_deb), "brew": share(d_brew), "brew_formula": share(np.array([any(k == "formula" for k, _, _ in s["brew"]) for s in S])),
    "brew_cask": share(np.array([any(k == "cask" for k, _, _ in s["brew"]) for s in S])), "nix": share(d_nix),
    "debian_or_brew": share(d_deb | d_brew), "union": share(d_any), "union_with_m4": share(d_any | d_m4),
    "debian_with_m4": share(d_deb | d_m4)}
R["coverage_count"] = {"debian": float(d_deb.mean()), "brew": float(d_brew.mean()), "nix": float(d_nix.mean()), "union": float(d_any.mean())}
R["content"] = {"share_all": share(cont), "count_all": int(cont.sum()), "share_of_uncovered": float(w[~d_any & cont].sum() / w[~d_any].sum()),
                "uncovered_share": share(~d_any), "uncovered_software_share": share(~d_any & ~cont),
                "coverage_among_software": float(w[d_any & ~cont].sum() / w[~cont].sum()), "coverage_among_content": float(w[d_any & cont].sum() / w[cont].sum())}
lang_unc = {}
for i in np.where(~d_any & ~cont)[0]:
    l = S[i]["meta"]["lang"] or "None"; lang_unc[l] = lang_unc.get(l, 0) + w[i]
R["uncovered_software_by_lang"] = dict(sorted(((k, float(v)) for k, v in lang_unc.items()), key=lambda x: -x[1])[:12])
R["top_uncovered_software"] = [(S[i]["name"], int(st[i]), S[i]["meta"]["lang"]) for i in np.argsort(-st) if not d_any[i] and not cont[i]][:15]
R["top_uncovered_content"] = [(S[i]["name"], int(st[i])) for i in np.argsort(-st) if not d_any[i] and cont[i]][:10]
R["top_brew_installs"] = [(S[i]["name"], int(inst[i])) for i in np.argsort(-inst)[:10]]
cov_b = d_brew & (inst > 0)
R["installs"] = {"n_brew_with_installs": int(cov_b.sum()), "median_installs": float(np.median(inst[cov_b])), "spearman_installs_stars": spearman(inst[cov_b], st[cov_b]),
                 "installs_per_star_median": float(np.median(inst[cov_b] / st[cov_b]))}
# 035 の照合の取りこぼし（M4）
R["m4"] = {"n_new": int((d_m4 & ~d_deb).sum()), "weight_new": share(d_m4 & ~d_deb), "examples": [(S[i]["name"], S[i]["m4"]) for i in np.where(d_m4 & ~d_deb)[0][:12]],
           "m3_only_not_in_brew_nix": int((np.array([bool(s["deb_m3"]) and not s["deb_m1"] for s in S]) & ~d_brew & ~d_nix).sum())}

# ---------- 2. 035 の再実行（同じ係数の組） ----------
tmp = tempfile.mkdtemp(prefix="rep036_035_")
for d in ("024-content-value-reader-guide", "032-collapse-scenarios", "033-scenario-reader-q", "034-t3-vault-value", "035-orphan-share-distro"):
    shutil.copytree(f"{RES}/{d}", f"{tmp}/{d}")
cwd = os.getcwd(); os.chdir(f"{tmp}/035-orphan-share-distro"); sys.argv = ["sim.py"]
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("sim.py", run_name="__main__")
os.chdir(cwd)
P, NREG, REGS, SCEN, K, by_scen, plug, summ = ns["P"], ns["NREG"], ns["REGS"], ns["SCEN"], ns["K"], ns["by_scen"], ns["plug"], ns["summ"]
MAIN35 = ns["MAIN"]; r35 = json.load(open(f"{RES}/035-orphan-share-distro/results.json"))
assert [s["name"] for s in ns["S"]] == [s["name"] for s in S]
assert np.array_equal(ns["dS"]["main"], d_deb) and np.array_equal(ns["st"], st)
g = np.random.default_rng(20261036)
P2 = {"iota": np.exp(g.uniform(np.log(1.0), np.log(10.0), K)), "g_b": g.uniform(0.3, 0.8, K)}

def fstar(r, dd, db, I, use_brew=True):
    Pdeb = 1 - (1 - P["p_m"]) ** NREG[r]; out = np.empty(K)
    for j0 in range(0, K, 250):
        sl = slice(j0, min(K, j0 + 250)); a = (P["a"][sl] * P["s_" + r][sl])[:, None]
        deb = (P["v"][sl] * Pdeb[sl])[:, None] * dd[None, :]
        ins = (1 - P["g"][sl])[:, None] * (1 - np.exp(-st[None, :] * P["kappa"][sl][:, None] * a))
        br = (db[None, :] * (1 - P2["g_b"][sl])[:, None] * (1 - np.exp(-I[None, :] * P2["iota"][sl][:, None] * a))) if use_brew else 0.0
        out[sl] = ((1 - deb) * (1 - br) * (1 - ins)) @ w
    return out
# 再現確認：Homebrew なしなら 035 の星側と一致
chk = {r: float(np.max(np.abs(fstar(r, d_deb, d_brew, inst, use_brew=False) - MAIN35["region"]["star"][r]))) for r in REGS}
R["check_035_star_maxabs"] = chk; assert max(chk.values()) < 1e-12
R["check_035_plug"] = {a: ns["R"]["plug_034"]["main"]["mix"][a]["P_S_gt_TC"] == r35["plug_034"]["main"]["mix"][a]["P_S_gt_TC"] for a in ns["WTS"]}
def model(dd, db, I, use_brew=True):
    fs = {r: fstar(r, dd, db, I, use_brew) for r in REGS}
    fm = {r: 0.5 * MAIN35["region"]["dep"][r] + 0.5 * fs[r] for r in REGS}
    return {"star": by_scen(fs), "mix": by_scen(fm)}
VAR = {"main": model(d_deb, d_brew, inst),
       "no_brew_035": model(d_deb, d_brew, inst, use_brew=False),
       "with_m4": model(d_deb | d_m4, d_brew, inst),
       "nix_as_debian": model(d_deb | d_nix | d_m4, d_brew, inst),
       "all_union_as_debian": model(d_any | d_m4, d_brew, inst)}
# 探索（事前登録外、判定に使わない）：030 の registry の package（≥1,000 被依存）がある repo も Debian と同じ扱いにした上限
REGSET = {(json.loads(l).get("rm_full_name") or "").lower() for l in open(f"{RES}/030-dependency-tail-census/data/pkgs.jsonl")}
d_reg = np.array([s["name"].lower() in REGSET for s in S])
VAR["explore_union_registry_as_debian"] = model(d_any | d_m4 | d_reg, d_brew, inst)
R["explore_registry_share"] = {"registry_any": share(d_reg), "registry_new": share(d_reg & ~d_any), "union_plus_registry": share(d_any | d_reg)}
R["f_orph"] = {k: {wk: {s: summ(x[wk][s]) for s in SCEN} for wk in ("star", "mix")} for k, x in VAR.items()}
R["plug_034"] = {k: {wk: plug(x[wk]) for wk in ("star", "mix")} for k, x in VAR.items()}
R["f_orph_035_ref"] = {"star_BIO": r35["f_orph"]["main"]["star"]["BIO"]["median"], "mix_BIO": r35["f_orph"]["main"]["mix"]["BIO"]["median"],
                       "plug_mix": {a: r35["plug_034"]["main"]["mix"][a]["P_S_gt_TC"] for a in ns["WTS"]}}
# 文章型だけを残したときの床（全ソフトが写される極限）：文章型の重みは配布系で補えない
R["content_floor"] = {"star_weight_content_uncovered": share(cont & ~d_any)}
fm = VAR["main"]["mix"]["BIO"]
R["spearman_mix_BIO"] = {**{k: spearman(P[k], fm) for k in ("a", "g", "kappa", "v", "p_m")}, **{k: spearman(P2[k], fm) for k in P2}}

# ---------- 3. 仮説 ----------
F = R["f_orph"]["main"]["star"]["BIO"]["median"]; F35 = R["f_orph_035_ref"]["star_BIO"]
PS = R["plug_034"]["main"]["mix"]["A_XPT_S"]["P_S_gt_TC"]
R["hypotheses"] = {
 "H48a": {"value": R["coverage_star_weight"]["union"], "supported": R["coverage_star_weight"]["union"] >= 0.35},
 "H48b": {"value": R["content"]["share_of_uncovered"], "supported": R["content"]["share_of_uncovered"] >= 0.5},
 "H48c": {"value": R["m4"]["weight_new"], "supported": R["m4"]["weight_new"] < 0.03},
 "H48d": {"value": {"036": F, "035": F35, "drop": F35 - F}, "supported": (F35 - F) >= 0.1},
 "H48e": {"value": PS, "supported": PS < 0.3},
 "H48f": {"value": R["installs"]["spearman_installs_stars"], "supported": R["installs"]["spearman_installs_stars"] < 0.3},
}

# ---------- 4. 図 ----------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
keys = [("Debian", d_deb), ("Homebrew", d_brew), ("nixpkgs", d_nix), ("Deb∪Brew", d_deb | d_brew), ("union", d_any)]
x = np.arange(len(keys))
ax[0].bar(x, [share(k & ~cont) for _, k in keys], 0.6, label="software repos", color="C0")
ax[0].bar(x, [share(k & cont) for _, k in keys], 0.6, bottom=[share(k & ~cont) for _, k in keys], label="content repos (lists, books, guides)", color="C1")
ax[0].axhline(share(~cont), ls="--", c="C0", lw=1, label=f"all software weight {share(~cont):.2f}")
ax[0].set_xticks(x); ax[0].set_xticklabels([k for k, _ in keys]); ax[0].set_ylabel("share of star weight (>=20k stars, n=1368)")
ax[0].set_title("Packaging coverage of top-starred repos"); ax[0].legend(fontsize=8)
cb = cov_b
ax[1].scatter(st[cb], inst[cb], s=8, alpha=0.6); ax[1].set_xscale("log"); ax[1].set_yscale("log")
ax[1].set_xlabel("GitHub stars"); ax[1].set_ylabel("Homebrew installs, 365 d (analytics, opt-out)")
ax[1].set_title(f"Installs vs stars (Spearman {R['installs']['spearman_installs_stars']:.2f}, n={int(cb.sum())})")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_coverage_content.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
vs = ["no_brew_035", "main", "with_m4", "nix_as_debian", "all_union_as_debian", "explore_union_registry_as_debian"]
for k, v in enumerate(vs):
    ax[0].boxplot([VAR[v]["star"]["BIO"]], positions=[k], widths=0.6, whis=(5, 95), showfliers=False)
ax[0].axhline(r35["breakeven_034"]["A_XPT_S"], ls="--", c="k", lw=1, label="034 break-even A_XPT_S")
ax[0].axhline(r35["breakeven_034"]["A_XPT_E"], ls=":", c="r", lw=1, label="034 break-even A_XPT_E")
ax[0].set_yscale("log"); ax[0].set_xticks(range(len(vs))); ax[0].set_xticklabels(vs, fontsize=7, rotation=15)
ax[0].set_ylabel("star-side f_orph (avg region, 5-95%)"); ax[0].legend(fontsize=8); ax[0].set_title("Star-side unreplaceable share")
xs = np.arange(3); bw = 0.13
for k, v in enumerate(vs):
    ax[1].bar(xs + k * bw, [R["plug_034"][v]["mix"][a]["P_S_gt_TC"] for a in ("A_Ord", "A_XPT_S", "A_XPT_E")], bw, label=v)
ax[1].set_xticks(xs + 2.5 * bw); ax[1].set_xticklabels(["A_Ord", "A_XPT_S", "A_XPT_E"]); ax[1].set_ylim(0, 1); ax[1].set_ylabel("P(S > T+C), mix weight")
ax[1].legend(fontsize=7); ax[1].set_title("Second site vs bundle")
fig.tight_layout(); fig.savefig(f"{HERE}/fig_forph_plug.png", dpi=120); plt.close(fig)

R["runtime_s"] = round(time.time() - T0, 1)
json.dump(R, open(f"{HERE}/results.json", "w"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: R[k] for k in ("coverage_star_weight", "coverage_count", "content", "installs", "m4", "hypotheses")}, ensure_ascii=False, indent=1, default=float))
print("f_orph star BIO", {k: round(R["f_orph"][k]["star"]["BIO"]["median"], 4) for k in VAR})
print("f_orph mix BIO", {k: round(R["f_orph"][k]["mix"]["BIO"]["median"], 4) for k in VAR})
print("plug mix", {k: {a: round(R["plug_034"][k]["mix"][a]["P_S_gt_TC"], 3) for a in ns["WTS"]} for k in VAR})
print("top uncovered software", R["top_uncovered_software"][:10])
print("runtime", R["runtime_s"])
