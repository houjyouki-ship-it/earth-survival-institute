"""研究085 第1部：崩壊の前後の都市の人口の弾力性 β（Reba et al. 2016 の Chandler / Modelski の表）。
局面の定義は PREREGISTRATION.md の登録を、表にある年の列に合わせて具体化したもの（report に逸脱として記す）。"""
import json, os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); S = os.path.join(HERE, "sources")
CH = pd.read_csv(os.path.join(S, "chandlerV2.csv"), encoding="latin1", low_memory=False)
MO = pd.read_csv(os.path.join(S, "modelskiAncientV2.csv"), encoding="latin1")
for _d in (CH, MO):
    for _c in ("Latitude", "Longitude"): _d[_c] = pd.to_numeric(_d[_c], errors="coerce")
TAB = {"chandler": CH, "modelski": MO}
# (名前, 表, 経度の範囲, 緯度の範囲, t0, t1, 種類)
EP = [
 ("W.Roman (Ch 361->622)", "chandler", (-10, 25), (28, 56), "AD_361", "AD_622", "collapse"),
 ("W.Roman (Ch 361->800)", "chandler", (-10, 25), (28, 56), "AD_361", "AD_800", "collapse"),
 ("W.Roman (Mo 200->700)", "modelski", (-10, 25), (28, 56), "AD_200", "AD_700", "collapse"),
 ("Han (Ch BC100->361)", "chandler", (95, 125), (20, 45), "BC_100", "AD_361", "collapse"),
 ("Han (Mo 100->400)", "modelski", (95, 125), (20, 45), "AD_100", "AD_400", "collapse"),
 ("Bronze Age E.Med (Mo BC1300->1000)", "modelski", (20, 50), (25, 42), "BC_1300", "BC_1000", "collapse"),
 ("Akkad/Ur III Mesop. (Mo BC2300->2000)", "modelski", (35, 55), (25, 40), "BC_2300", "BC_2000", "collapse"),
 ("Mongol W.Asia (Ch 1200->1300)", "chandler", (40, 75), (25, 45), "AD_1200", "AD_1300", "collapse"),
 ("Mongol W.Asia (Ch 1200->1400)", "chandler", (40, 75), (25, 45), "AD_1200", "AD_1400", "collapse"),
 ("Europe ctrl (Ch 1700->1800)", "chandler", (-10, 40), (35, 60), "AD_1700", "AD_1800", "control"),
 ("Europe ctrl (Ch 1500->1600)", "chandler", (-10, 40), (35, 60), "AD_1500", "AD_1600", "control"),
 ("China ctrl (Ch 1700->1800)", "chandler", (95, 125), (20, 45), "AD_1700", "AD_1800", "control"),
 ("India ctrl (Ch 1500->1600)", "chandler", (68, 92), (8, 32), "AD_1500", "AD_1600", "control"),
 ("E.Med ctrl (Mo BC500->300)", "modelski", (20, 50), (25, 42), "BC_500", "BC_300", "control"),
 ("Mesop./Iran ctrl (Mo AD700->900)", "modelski", (35, 60), (25, 40), "AD_700", "AD_900", "control"),
]
def fit(x, y):
    X = np.c_[np.ones_like(x), x]; b, *_ = np.linalg.lstsq(X, y, rcond=None); return float(b[1])
def boot(x, y, rg, B=2000):
    n = len(x); out = [fit(x[i], y[i]) for i in (rg.integers(0, n, n) for _ in range(B)) if np.ptp(x[i]) > 0]
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
rg = np.random.default_rng(20261085); OUT = []
for name, t, lon, lat, a, b, kind in EP:
    d = TAB[t]; m = d.Longitude.between(*lon) & d.Latitude.between(*lat)
    p0 = pd.to_numeric(d.loc[m, a], errors="coerce"); p1 = pd.to_numeric(d.loc[m, b], errors="coerce")
    ok0 = p0 > 0; P0 = p0[ok0].values.astype(float); P1 = p1[ok0].values.astype(float)
    th = float(pd.to_numeric(d[b], errors="coerce").loc[m].min()) if (p1 > 0).any() else np.nan  # 地域の t1 の掲載の下限
    surv = P1 > 0; n0 = len(P0); ns = int(surv.sum())
    r = {"episode": name, "table": t, "kind": kind, "t0": a, "t1": b, "n_t0": n0, "n_listed_t1": ns, "share_listed": ns / n0 if n0 else None,
         "cities_t0": d.loc[m, "City"][ok0].tolist(), "threshold_t1": th}
    if ns >= 4 and np.ptp(np.log(P0[surv])) > 0:
        x, y = np.log(P0[surv]), np.log(P1[surv]); r["beta_listed"] = fit(x, y); r["beta_listed_ci"] = boot(x, y, rg)
    if n0 >= 4 and not np.isnan(th):
        P1h = np.where(surv, P1, th / 2); x, y = np.log(P0), np.log(P1h); r["beta_half"] = fit(x, y); r["beta_half_ci"] = boot(x, y, rg)
        o = np.argsort(-P0); top, bot = o[: n0 // 2], o[n0 // 2:]
        r["rho_top_vs_bottom"] = float((P1h[top].sum() / P0[top].sum()) / (P1h[bot].sum() / P0[bot].sum()))
        r["total_ratio_t1_t0"] = float(P1[surv].sum() / P0.sum())
    OUT.append(r)
for r in OUT:
    print(f'{r["episode"]:40s} {r["kind"]:8s} n0={r["n_t0"]:3d} listed={r["n_listed_t1"]:3d} b_list={r.get("beta_listed", float("nan")):.2f} {r.get("beta_listed_ci")} b_half={r.get("beta_half", float("nan")):.2f} {r.get("beta_half_ci")} rho={r.get("rho_top_vs_bottom", float("nan")):.2f}')
def med(kind, key): v = [r[key] for r in OUT if r["kind"] == kind and key in r]; return (float(np.median(v)), len(v)) if v else (None, 0)
SUM = {k: {kind: med(kind, k) for kind in ("collapse", "control")} for k in ("beta_listed", "beta_half", "rho_top_vs_bottom")}
print(json.dumps(SUM))
json.dump({"episodes": OUT, "summary_median": SUM}, open(os.path.join(HERE, "episodes.json"), "w", encoding="utf8"), ensure_ascii=False, indent=1)
