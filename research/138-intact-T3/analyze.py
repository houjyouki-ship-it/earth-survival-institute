"""研究138：150 Tg で無傷の 4 か国が T3（計算機の使える社会）を保つ割合 q_T3 を発電の国内自給（OWID energy-data）で置き、
032 の NUC_G の T3 保持と突き合わせ、重い崩壊の標本の中の条件付き P_ext・F_heavy を出す。
F_heavy・c_C・q_T3 は模型の中の量で、生存確率でも復興確率でもない。"""
import json, math, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
B='../136-conquest-pressure/'
X=json.load(open(B+'xia_tables.json'))
CEN={n:(a,b) for c,a,b,n in json.load(open('centroids.json'))}
ALIAS={'Republic of Congo':'Congo [Republic]','Myanmar':'Myanmar [Burma]'}
cc=json.load(open('../134-modern-downhill/results.json'))['cC_curve']
assert [round(f,4) for f in cc['F']]==[1.911,1.9646,2.0219,2.1664]
def Fh(c): return float(np.interp(c,cc['c'],cc['F']))
C_PRESS=0.723; C_NO=0.09; COLS=X['S4']['cols']
def hav(p,q):
    la1,lo1,la2,lo2=map(math.radians,(*p,*q))
    h=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*6371*math.asin(math.sqrt(h))
def run(table,col,reach,s_int=0.25,s_heavy=0.75,popmin=10.0):   # 137 と同じ
    r=X[table]['rows']; j=COLS.index(col); nat=[]
    for k,v in r.items():
        if k=='Total': continue
        pop=v[0]; s=min(v[j]/pop,1.0) if pop>0 else 0
        nat.append(dict(n=k,pop=pop,s=s,ll=CEN[ALIAS.get(k,k)]))
    intact=[d for d in nat if d['s']<s_int and d['pop']>=popmin]
    heavy=[d for d in nat if d['s']>=s_heavy]
    W=P=0.0
    for d in heavy:
        w=d['pop']*(1-d['s']); ok=any(reach(e['n']) is None or hav(d['ll'],e['ll'])<=reach(e['n']) for e in intact)
        W+=w; P+=w*ok
    return P/W
r137=json.load(open('../137-conquest-reach/results.json'))
Dh=r137['median']['D_home']; Ds=r137['median']['D_stage']; SS=r137['selfsuf']
P_main=run('S4','150',lambda n: Dh if SS.get(n,0)>=0.8 else Ds)
assert abs(P_main-r137['main']['Pext'])<1e-12          # 再現：137 の主
P_stage=run('S4','150',lambda n: Ds)
# ---- (1) 発電の国内自給（OWID energy-data、2024 年：4 か国の全列がそろう最新の年）
d=pd.read_csv('sources/owid-energy-data.csv')
C4=['Argentina','Australia','Sri Lanka','Dominican Republic']; YR=2024
x=d[d.country.isin(C4)&(d.year==YR)].set_index('country')
OUT={'note':'q_T3・F_heavy・c_C は模型の中の量。発電は OWID energy-data 2024 年（TWh）。','year':YR,'countries':{}}
for n in C4:
    r=x.loc[n].fillna({'coal_production':0,'gas_production':0,'oil_production':0,'gas_electricity':0})
    flags=[f for f in ['coal_production','gas_production','oil_production'] if pd.isna(x.loc[n,f])]
    s={}
    for f in ['coal','gas','oil']:
        cons=r[f+'_consumption']; s[f]=0.0 if (pd.isna(cons) or cons<=0) else min(1.0,r[f+'_production']/cons)
    gen=r['electricity_generation']
    fossil_dom=sum(r[f+'_electricity']*s[f] for f in ['coal','gas','oil'])
    dom=(r['renewables_electricity']+r['nuclear_electricity']+fossil_dom)/gen
    dom_nonuc=(r['renewables_electricity']+fossil_dom)/gen
    OUT['countries'][n]=dict(gen=gen,ren=r['renewables_electricity'],nuc=r['nuclear_electricity'],coal=r['coal_electricity'],gas=r['gas_electricity'],oil=r['oil_electricity'],
        net_imports=r['net_elec_imports'],s=s,d=dom,d_nonuclear=dom_nonuc,s_oil137=SS[n],prod_missing_as0=flags)
    print(n,{k:round(v,3) for k,v in s.items()},'d',round(dom,3),'d_noNuc',round(dom_nonuc,3),'missing->0',flags)
D={n:OUT['countries'][n]['d'] for n in C4}
RULES={'main':lambda n:D[n],
       'threshold':lambda n:1.0 if D[n]>=0.8 else (0.5 if D[n]>=0.5 else 0.0),
       'strict_oil':lambda n:D[n]*SS[n]**0.5 if SS[n]<1 else D[n],
       'nuclear_foreign':lambda n:OUT['countries'][n]['d_nonuclear']}
T3_032=0.65+0.15; HEAVY_032=0.14+0.04+0.01
OUT['q']={}
for rn,f in RULES.items():
    q={n:min(1.0,f(n)) for n in C4}
    lose_ind=float(np.prod([1-q[n] for n in C4])); lose_com=float(min(1-q[n] for n in C4))
    OUT['q'][rn]=dict(q=q,Q_independent=1-lose_ind,Q_comonotone=1-lose_com,P_all_lose_ind=lose_ind,P_all_lose_comono=lose_com)
    print(rn,{k:round(v,3) for k,v in q.items()},'Q ind',round(1-lose_ind,4),'Q comono',round(1-lose_com,3))
# ---- (2) 条件付きの P_ext・F_heavy（重い崩壊の標本＝4 か国とも T3 を失った）
cond=[]
for pi in [0,0.25,0.5,1.0]:
    P=pi*P_main+(1-pi)*P_stage; c=P*C_PRESS+(1-P)*C_NO; cond.append(dict(pi=pi,Pext=P,cC=c,F_heavy=Fh(c)))
cLB=C_NO; cond.append(dict(pi='no_reach',Pext=0.0,cC=cLB,F_heavy=Fh(cLB)))
for g in cond: print('pi',g['pi'],round(g['Pext'],3),round(g['F_heavy'],4))
Fs=[g['F_heavy'] for g in cond]; OUT['conditional']=cond; OUT['F_range']=[min(Fs),max(Fs)]
OUT['P_main137']=P_main; OUT['P_stage']=P_stage
# 032 との突き合わせ：NUC_G の重い崩壊の割合の上限
M=OUT['q']['main']
OUT['compare032']=dict(T3_032=T3_032,heavy_032=HEAVY_032,heavy_upper_ind=M['P_all_lose_ind'],heavy_upper_comono=M['P_all_lose_comono'])
# ---- 判定
dd=OUT['countries']
OUT['hyp']={'H150a':dict(val=[n for n in C4 if dd[n]['d']>=0.8],ok=sum(dd[n]['d']>=0.8 for n in C4)>=2),
 'H150b':dict(val=M['Q_independent'],ok=M['Q_independent']>=T3_032),
 'H150c':dict(val=max(Fs)-min(Fs),ok=(max(Fs)-min(Fs))<0.04),
 'H150d':dict(val=dd['Dominican Republic']['d'],ok=dd['Dominican Republic']['d']<0.5)}
for k,v in OUT['hyp'].items(): print(k,v)
json.dump(OUT,open('results.json','w'),ensure_ascii=False,indent=1,default=float)
fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].bar(C4,[dd[n]['d'] for n in C4]); ax[0].axhline(0.8,ls='--',c='k'); ax[0].set_ylabel('domestic share of electricity d (2024)'); ax[0].tick_params(axis='x',labelsize=7)
ax[1].plot([g['pi'] for g in cond[:4]],[g['F_heavy'] for g in cond[:4]],'o-',label='Argentina expedition prob. pi')
ax[1].axhline(cond[4]['F_heavy'],ls=':',label='no reach (P_ext=0)'); ax[1].set_xlabel('pi'); ax[1].set_ylabel('conditional F_heavy (model)'); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig('fig_intact_T3.png',dpi=110)
