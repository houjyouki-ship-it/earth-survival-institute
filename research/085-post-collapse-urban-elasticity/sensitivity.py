"""研究085 補助（登録外・記述）：消えた都市に入れる値（掲載の下限の 0.25・0.5・1 倍）と、対照から India 1500->1600（1565 年の Vijayanagara の陥落を含む）を除く版。"""
import json, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); g = {"__file__": os.path.join(HERE, "episodes.py")}
src = open(os.path.join(HERE, "episodes.py"), encoding="utf8").read(); exec(compile(src[: src.index("rg = np.random")], "episodes.py", "exec"), g)
TAB, EP, fit = g["TAB"], g["EP"], g["fit"]; out = {}
for f in (0.25, 0.5, 1.0):
    res = {}
    for name, t, lon, lat, a, b, kind in EP:
        d = TAB[t]; m = d.Longitude.between(*lon) & d.Latitude.between(*lat)
        p0 = pd.to_numeric(d.loc[m, a], errors="coerce"); p1 = pd.to_numeric(d.loc[m, b], errors="coerce"); ok = p0 > 0
        P0 = p0[ok].values.astype(float); P1 = p1[ok].values.astype(float)
        if len(P0) < 4 or not (p1 > 0).any(): continue
        th = float(p1[p1 > 0].min()); res[name] = (kind, fit(np.log(P0), np.log(np.where(P1 > 0, P1, th * f))))
    col = [v for k, v in res.values() if k == "collapse"]; ctl = [v for k, v in res.values() if k == "control"]
    ctl2 = [v for n, (k, v) in res.items() if k == "control" and not n.startswith("India")]
    out[str(f)] = {"collapse_median": float(np.median(col)), "control_median": float(np.median(ctl)), "control_median_noIndia": float(np.median(ctl2)), "by_episode": {n: v for n, (k, v) in res.items()}}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "sensitivity.json"), "w"), indent=1)
