"""研究135: 大きな死の後に国家が解体し人口が減り続ける割合 c_C を死亡率のロジスティックで外挿し、
134 の cC_curve の線形補間で D2rz の F_heavy に変える。F_heavy は生存確率ではない（模型の中の価値の比）。"""
import json, numpy as np
W="https://en.wikipedia.org/wiki/"
# m: 開始から25年以内の死（または人口減）の割合（資料の幅の中点）。Y: 規則 (i)かつ(ii)。X: 征服。flags: long/unk/ambig/city/contested
C=[
 dict(k="England_BlackDeath",m=0.50,Y=0,X=0,fl=[],src=W+"Black_Death_in_England",ev="40–60% widely accepted; 王国は存続"),
 dict(k="Norway_BlackDeath",m=0.625,Y=0,X=0,fl=["ambig"],Yalt=1,src=W+"Black_Death_in_Norway",ev="60–65% died; 1380 Denmark と同君連合、1397 Kalmar、1537 から従属。15世紀の人口は1300の半分未満"),
 dict(k="Egypt_BlackDeath",m=1/3,Y=0,X=0,fl=["city"],src=W+"Black_Death_in_the_Middle_East",ev="Cairo 200,000 = a third; Mamluk は 1517 に Ottoman が征服（170年後）"),
 dict(k="Justinian_EastMed",m=0.25,Y=0,X=0,fl=[],src=W+"Plague_of_Justinian",ev="up to a quarter of the Eastern Mediterranean"),
 dict(k="Athens_430BC",m=0.25,Y=0,X=0,fl=[],src=W+"Plague_of_Athens",ev="around 25% of the population"),
 dict(k="ThirtyYears_Germany",m=0.40,Y=0,X=0,fl=["long"],src=W+"Thirty_Years%27_War",ev="decline of roughly 40% 1600–1650, regained by 1750; 帝国は存続"),
 dict(k="Aztec_1520",m=0.50,Y=1,X=1,fl=[],src=W+"Fall_of_Tenochtitlan",ev="lost approximately 50% from smallpox; 1521 陥落、後の cocoliztli でさらに減少"),
 dict(k="Inca_conquest",m=0.93,Y=1,X=1,fl=["long"],src=W+"Spanish_conquest_of_the_Inca_Empire",ev="decline of 93 percent in the Andes; 1572 Spanish victory"),
 dict(k="Taino_Hispaniola",m=0.85,Y=1,X=1,fl=["long"],src=W+"Ta%C3%ADno_genocide",ev="In thirty years 80–90% died; 1565 200人"),
 dict(k="Cocoliztli_NewSpain",m=0.75,Y=0,X=0,fl=[],src=W+"Cocoliztli_epidemics",ev="60–90% decreased; 副王領は 1821 まで（"+W+"New_Spain）"),
 dict(k="Hawaii_contact",m=1-71/128,Y=0,X=0,fl=["long","ambig"],Yalt=1,src=W+"Hawaiian_Kingdom",ev="128,000 (1778)→71,000 (1853)→24,000 (1920); 王国は 1893 に倒れる（115年後）"),
 dict(k="Paraguay_war",m=0.65,Y=0,X=0,fl=[],src=W+"Paraguayan_War",ev="60–70% died; 1876 まで占領、国は存続"),
 dict(k="Herero_1904",m=0.80,Y=1,X=1,fl=["unk"],src=W+"Herero_and_Nama_genocide",ev="80 percent of pre-war population; 土地と家畜を奪われ 1910年代 約15,000"),
 dict(k="Moriori_1835",m=0.95,Y=1,X=1,fl=[],src=W+"Moriori",ev="killing or displacing nearly 95%; 1862 に 2,000 のうち 101"),
 dict(k="Finland_1695",m=1/3,Y=0,X=0,fl=[],src=W+"Great_Famine_of_1695%E2%80%931697",ev="about a third of the Finnish population; Sweden の王国は存続"),
 dict(k="Bengal_1770",m=0.29,Y=0,X=0,fl=["contested"],src=W+"Great_Bengal_famine_of_1770",ev="a quarter and third (Dutta 4–5%)"),
 dict(k="Russia_1601",m=0.30,Y=0,X=0,fl=["ambig"],Yalt=1,src=W+"Russian_famine_of_1601%E2%80%931603",ev="about 30%; 動乱 1598–1613、1613 に回復"),
]
LOW=[
 dict(k="Antonine",m=0.10,Y=0,X=0,src=W+"Antonine_Plague",ev="about 10%"),
 dict(k="Taiping",m=0.15,Y=0,X=0,src=W+"Taiping_Rebellion",ev="5–10% to 25%（中点 0.15）; 移民で再入植"),
 dict(k="Ireland_famine",m=0.225,Y=0,X=0,src=W+"Great_Famine_(Ireland)",ev="population fall by 20–25% 1841–1871"),
 dict(k="Cambodia_KR",m=0.225,Y=0,X=0,src=W+"Cambodian_genocide",ev="20% to 25% of 1975 population"),
]
cc=json.load(open("../134-modern-downhill/results.json"))["cC_curve"]
assert np.allclose(cc["F"],[1.9110403512681224,1.9646378388278007,2.0218922586443555,2.1664031895428884])
Fof=lambda c: float(np.interp(c,cc["c"],cc["F"]))
sig=lambda z:1/(1+np.exp(-z))
def fit(m,y,firth=False,it=200):
    X=np.c_[np.ones_like(m),m]; b=np.zeros(2)
    for _ in range(it):
        p=sig(X@b); Wd=p*(1-p); I=X.T@(X*Wd[:,None])+1e-10*np.eye(2)
        if firth:
            Ii=np.linalg.inv(I); h=Wd*np.einsum('ij,jk,ik->i',X,Ii,X); U=X.T@(y-p+h*(0.5-p))
        else: U=X.T@(y-p)
        st=np.linalg.solve(I,U); st=np.clip(st,-5,5); b=b+st
        if np.max(np.abs(st))<1e-9: break
    return b
def separated(m,y):
    if y.min()==y.max(): return True
    return m[y==1].min()>m[y==0].max() or m[y==1].max()<m[y==0].min()
def est(cases,Ykey=lambda c:c["Y"],nb=2000,seed=135):
    m=np.array([c["m"] for c in cases]); y=np.array([Ykey(c) for c in cases],float)
    sep=separated(m,y); b=fit(m,y,firth=sep)
    rng=np.random.default_rng(seed); P8=[];P9=[];B=[]
    for _ in range(nb):
        i=rng.integers(0,len(m),len(m)); mi,yi=m[i],y[i]
        bb=fit(mi,yi,firth=separated(mi,yi)); B.append(bb[1]); P8.append(sig(bb[0]+bb[1]*0.8)); P9.append(sig(bb[0]+bb[1]*0.9))
    P8=np.array(P8); F8=np.array([Fof(p) for p in P8])
    q=lambda a:[float(np.percentile(a,5)),float(np.percentile(a,95))]
    return dict(n=len(m),k1=int(y.sum()),firth=bool(sep),a=float(b[0]),b=float(b[1]),
        c80=float(sig(b[0]+b[1]*0.8)),c90=float(sig(b[0]+b[1]*0.9)),c80_90ci=q(P8),c90_90ci=q(np.array(P9)),
        P_b_pos=float(np.mean(np.array(B)>0)),F80=Fof(float(sig(b[0]+b[1]*0.8))),F80_90ci=q(F8),F80_bootmean=float(F8.mean()))
V={}
V["main"]=est(C)
V["noconq"]=est(C,lambda c:0 if c["X"] else c["Y"])
V["low"]=est(C+LOW)
V["nolong"]=est([c for c in C if "long" not in c["fl"]])
V["nounk"]=est([c for c in C if "unk" not in c["fl"]])
V["ambig1"]=est(C,lambda c:c.get("Yalt",c["Y"]))
V["noconq_ambig1"]=est(C,lambda c:0 if c["X"] else c.get("Yalt",c["Y"]))
V["nocontested_city"]=est([c for c in C if not set(c["fl"])&{"contested","city"}])
# 記述：征服の有無で分けた割合と、征服なしの m>=0.5 の件数（0 件の崩れへのルール・オブ・スリー）
nc=[c for c in C if not c["X"]]; nc50=[c for c in nc if c["m"]>=0.5]
desc=dict(Y_given_X1=f'{sum(c["Y"] for c in C if c["X"])}/{sum(c["X"] for c in C)}',
          Y_given_X0=f'{sum(c["Y"] for c in nc)}/{len(nc)}',
          noconq_m50=[c["k"] for c in nc50], rule_of_three_upper95=3/len(nc50),
          Y_ambig1_given_X0=f'{sum(c.get("Yalt",c["Y"]) for c in nc)}/{len(nc)}')
bounds=dict(F_c0=Fof(0),F_c1=Fof(1))
H=dict(H147a=V["main"]["b"]>0 and V["main"]["P_b_pos"]>=0.8, H147b=V["main"]["c80"]>=0.5,
       H147c=V["noconq"]["c80"]<V["main"]["c80"], H147d=V["main"]["F80"]>=2.02)
out=dict(note="F_heavy は生存確率ではない。c_C は史実の破局の二値の符号化からの外挿。",cases=C,low=LOW,versions=V,desc=desc,bounds=bounds,hyp=H)
json.dump(out,open("results.json","w"),ensure_ascii=False,indent=1)
for k,v in V.items(): print(f'{k:18s} n={v["n"]:2d} k1={v["k1"]} firth={v["firth"]!s:5s} b={v["b"]:7.2f} P(b>0)={v["P_b_pos"]:.2f} c80={v["c80"]:.3f} [{v["c80_90ci"][0]:.3f},{v["c80_90ci"][1]:.3f}] c90={v["c90"]:.3f} F80={v["F80"]:.3f} [{v["F80_90ci"][0]:.3f},{v["F80_90ci"][1]:.3f}]')
print(desc); print(bounds); print(H)
# 図
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4.2)); g=np.linspace(0,1,201)
for c in C+LOW:
    ax[0].scatter(c["m"],c["Y"]+np.random.default_rng(hash(c["k"])%999).uniform(-.04,.04),c="r" if c["X"] else "k",marker="x" if "long" in c.get("fl",[]) else "o",s=30)
for k,ls in [("main","-"),("noconq","--"),("ambig1",":"),("noconq_ambig1","-.")]:
    ax[0].plot(g,sig(V[k]["a"]+V[k]["b"]*g),ls,label=f'{k} (c80={V[k]["c80"]:.2f})')
ax[0].axvline(0.8,c="gray",lw=.6); ax[0].set_xlabel("mortality m"); ax[0].set_ylabel("P(state dissolves & decline)"); ax[0].legend(fontsize=7); ax[0].set_title("red = conquest")
ks=list(V); ax[1].barh(ks,[V[k]["F80"] for k in ks],xerr=np.array([[V[k]["F80"]-V[k]["F80_90ci"][0],V[k]["F80_90ci"][1]-V[k]["F80"]] for k in ks]).T,color="steelblue")
ax[1].axvline(Fof(0),c="k",ls="--",lw=.7); ax[1].axvline(Fof(1),c="k",ls="--",lw=.7); ax[1].set_xlim(1.88,2.2); ax[1].set_xlabel("F_heavy (D2rz) at m=0.8")
plt.tight_layout(); plt.savefig("fig_cC_logit.png",dpi=120)
