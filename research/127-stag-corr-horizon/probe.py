import os,time,numpy as np
t0=time.time()
P="../126-stag-fheavy-chain/sim.py"; P=os.path.abspath(P)
s=open(P,encoding="utf8").read(); s=s[:s.index("\n# ---------- 125 と同じ乱数")]
G={"__file__":P,"__name__":"x"}; exec(compile(s,P,"exec"),G)
Vg=G["V_m"].__globals__
print("H",Vg.get("H"), type(Vg.get("life")), np.shape(Vg.get("life")), np.median(Vg["life"]) if "life" in Vg else None)
print("COMPS",G["COMPS"],"FIXED",G["FIXED"])
B=G["B"]
for k in G["COMPS"]:
    st=B[k]["st"]; print(k, {int(v):round(float((st==v).mean()),3) for v in np.unique(st)}, list(B[k].keys())[:10])
print([k for k in Vg if "CASC" in k or "typ" in k.lower() or "BIO" in k][:20])
print(time.time()-t0)
