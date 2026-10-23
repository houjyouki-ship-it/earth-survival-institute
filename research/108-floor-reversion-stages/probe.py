import os,numpy as np,time
t=time.time()
P="/workspace/earth-survival-institute/research/104-ruin-glass-thresholds/sim.py"
src=open(P).read(); cut=src.index("# ---------- 仮説")
G={"__file__":P,"__name__":"x"}; exec(compile(src[:cut],P,"exec"),G)
print(time.time()-t)
print(G["COMPS"], G["FIXED"])
for k in G["COMPS"]:
    if k in G["FIXED"]: continue
    st=G["B"][k]["st"]; sv=G["SRC"][k]["sv"]
    print(k, {i:round(float(np.mean(st==i)),3) for i in range(7)}, np.quantile(sv["nmax"],[.05,.5,.95]))
print(np.quantile(G["r_pop"],[.05,.5,.95]), G["runs"] if "runs" in G else G["R"]["runs"]["both_m0.3_inf"])
