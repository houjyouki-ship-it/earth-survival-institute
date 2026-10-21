"""086 の補足（記述）：標的の国の NE の点のうち、標的から 10 km の外にある点の数と人口（sim.py と同じ点と標的）。"""
import json, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RS = os.path.join(HERE, "..")
p74 = os.path.join(RS, "074-within-country-reach", "sim.py"); s = open(p74, encoding="utf8").read()
G = {"__file__": p74, "__name__": "z"}; exec(compile(s[: s.index("def wpts")], p74, "exec"), G)
plat, plon, ppop, pc, A2, hav = G["plat"], G["plon"], G["ppop"], G["pc"], G["A2"], G["hav"]
TGT = json.load(open(os.path.join(RS, "084-nuclear-target-copy-loss", "sources", "targets.json"), encoding="utf8"))
cA2 = np.array([A2[i] for i in pc]); out = {}
for sc in ("xia150", "xia47"):
    tg = []
    for c, n in TGT[sc]["weapons"].items():
        idx = np.where(cA2 == c)[0]; tg += list(idx[np.argsort(-ppop[idx], kind="stable")][:n])
    tg = np.array(tg)
    for c in TGT[sc]["weapons"]:
        idx = np.where(cA2 == c)[0]; d = hav(plat[idx][:, None], plon[idx][:, None], plat[tg][None, :], plon[tg][None, :]).min(1)
        out[f"{sc}:{c}"] = {"ne_points": int(len(idx)), "weapons": TGT[sc]["weapons"][c], "points_outside_10km": int((d > 10).sum()), "pop_share_outside": float(ppop[idx][d > 10].sum() / ppop[idx].sum())}
json.dump(out, open(os.path.join(HERE, "zone_check.json"), "w"), indent=1); print(json.dumps(out, indent=1))
