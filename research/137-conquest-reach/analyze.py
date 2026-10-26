"""研究137：征服の届く距離を史実（135 の征服つき 5 件）で置き、無傷の国の燃料の自給で遠征の能力を分けて P_ext・c_C・F_heavy を出す。
F_heavy・c_C は模型の中の量で、生存確率でも復興確率でもない。"""
import json, math, numpy as np, sys
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
# 再現：136 の主（S4、150、2,000 km、0.75）
r136=json.load(open(B+'results.json'))['main']
# ---- (1) 史実の距離（座標の出典は report.md）
CHATHAM=(-44.00,-176.50)        # Wikipedia Chatham Islands {{coord|44.00|S|176.50|W}}
WELLINGTON=(-(41+17/60+20/3600),174+46/60+38/3600)  # Wikipedia Wellington
C=lambda n:CEN[n]
cases=[
 dict(k='Aztec_1520', home=('Spain',C('Spain')), stage=('Cuba',C('Cuba')), tgt=('Mexico',C('Mexico')), flag=''),
 dict(k='Inca_conquest', home=('Spain',C('Spain')), stage=('Panama',C('Panama')), tgt=('Peru',C('Peru')), flag=''),
 dict(k='Taino_Hispaniola', home=('Spain',C('Spain')), stage=None, tgt=('Dominican Republic',C('Dominican Republic')), flag='nostage'),
 dict(k='Herero_1904', home=('Germany',C('Germany')), stage='in-situ', tgt=('Namibia',C('Namibia')), flag='colony'),
 dict(k='Moriori_1835', home=('New Zealand',C('New Zealand')), stage=('Wellington',WELLINGTON), tgt=('Chatham',CHATHAM), flag=''),
]
for c in cases:
    c['D_home']=hav(c['home'][1],c['tgt'][1])
    c['D_stage']=c['D_home'] if c['stage'] is None else (0.0 if c['stage']=='in-situ' else hav(c['stage'][1],c['tgt'][1]))
Dh=float(np.median([c['D_home'] for c in cases])); Ds=float(np.median([c['D_stage'] for c in cases]))
# Herero の版：決め手の増援は本国から（資料：After reinforcements arrived from Germany）
Ds_alt=float(np.median([c['D_home'] if c['k']=='Herero_1904' else c['D_stage'] for c in cases]))
for c in cases: print(c['k'],round(c['D_home']),round(c['D_stage']),c['flag'])
print('median D_home',round(Dh),'D_stage',round(Ds),'D_stage(Herero=home)',round(Ds_alt))
# ---- (2) 燃料の自給（EIA 2026-01 の原油生産 千 bbl/日、Wikipedia 一覧；消費 bbl/日、Global Firepower 2026、Wikipedia 一覧）
prod={'Argentina':873,'Australia':241,'Sri Lanka':0,'Dominican Republic':0}   # 一覧に無い国は 0
cons={'Argentina':749,'Australia':1151,'Sri Lanka':100,'Dominican Republic':146}
selfsuf={k:prod[k]/cons[k] for k in prod}
print('self-sufficiency',{k:round(v,3) for k,v in selfsuf.items()})
def run(table,col,reach,s_int=0.25,s_heavy=0.75,popmin=10.0):
    r=X[table]['rows']; j=COLS.index(col); nat=[]
    for k,v in r.items():
        if k=='Total': continue
        pop=v[0]; s=min(v[j]/pop,1.0) if pop>0 else 0
        nat.append(dict(n=k,pop=pop,s=s,ll=CEN[ALIAS.get(k,k)]))
    intact=[d for d in nat if d['s']<s_int and d['pop']>=popmin]
    heavy=[d for d in nat if d['s']>=s_heavy]
    W=P=0.0; press=[]
    for d in heavy:
        w=d['pop']*(1-d['s']); ok=False
        for e in intact:
            R=reach(e['n'])
            if R is None or hav(d['ll'],e['ll'])<=R: ok=True
        W+=w; P+=w*ok
        if ok: press.append(d['n'])
    Pext=P/W; cC=Pext*C_PRESS+(1-Pext)*C_NO
    return dict(table=table,col=col,s_heavy=s_heavy,intact=[e['n'] for e in intact],Pext=Pext,cC=cC,F_heavy=Fh(cC),pressured=press)
# 再現の確認
rep=run('S4','150',lambda n:2000)
assert abs(rep['Pext']-r136['Pext'])<1e-12 and abs(rep['F_heavy']-r136['F_heavy'])<1e-12
cap=lambda n: selfsuf.get(n,0)>=0.8
RULES={'main':lambda n: Dh if cap(n) else Ds,
       'all_home':lambda n: Dh, 'all_stage':lambda n: Ds,
       'cap_unlimited':lambda n: None if cap(n) else Ds,
       'main_HereroHome':lambda n: Dh if cap(n) else Ds_alt,
       'cap_only_home':lambda n: Dh if cap(n) else -1}
OUT={'note':'F_heavy・c_C は模型の中の量。距離は国の重心の大圏距離（136 と同じ）。',
     'cases':[{k:(v if k not in ('home','stage','tgt') else (v if isinstance(v,(str,type(None))) else v[0])) for k,v in c.items()} for c in cases],
     'median':dict(D_home=Dh,D_stage=Ds,D_stage_HereroHome=Ds_alt),'selfsuf':selfsuf,'repro136':rep['Pext']}
grid=[]
for t in X:
  for col in ['150','150hw','150tw']:
    for sh in [0.5,0.75,0.9]:
      for rn,rf in RULES.items():
        g=run(t,col,rf,s_heavy=sh); g['rule']=rn; grid.append(g)
main=[g for g in grid if g['table']=='S4' and g['col']=='150' and g['s_heavy']==0.75]
for g in main: print(g['rule'],round(g['Pext'],3),round(g['cC'],3),round(g['F_heavy'],3),g['pressured'] if g['rule']=='main' else '')
M=[g for g in main if g['rule']=='main'][0]
Fm=[g['F_heavy'] for g in grid if g['rule']=='main']; Fa=[g['F_heavy'] for g in grid]
print('main rule all tables/thresholds F',round(min(Fm),3),round(max(Fm),3),' all rules',round(min(Fa),3),round(max(Fa),3))
# (3) T3 を保つ場合：NUC_G の標本のうち重い崩壊に残るのは (1-q_T3)。条件付きの F_heavy は P_ext をそのまま使う。
T3=[dict(q_T3=q,weight_left=1-q) for q in [0,0.5,0.9,1.0]]
H={'H149a':dict(val=Dh,ok=Dh>2000),'H149b':dict(val=Ds,ok=Ds<2000),
   'H149c':dict(val=selfsuf,ok=[k for k,v in selfsuf.items() if v>=0.8]==['Argentina']),
   'H149d':dict(val=M['F_heavy'],ok=1.93<=M['F_heavy']<2.0)}
print('hyp',json.dumps(H,ensure_ascii=False,default=str))
OUT.update(main=M,grid=grid,range_main_rule=[min(Fm),max(Fm)],range_all=[min(Fa),max(Fa)],T3=T3,hyp=H)
json.dump(OUT,open('results.json','w'),ensure_ascii=False,indent=1,default=str)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4))
ks=[c['k'] for c in cases]; x=np.arange(len(ks))
ax[0].bar(x-0.2,[c['D_home'] for c in cases],0.4,label='home->target');ax[0].bar(x+0.2,[c['D_stage'] for c in cases],0.4,label='staging->target')
ax[0].axhline(2000,ls='--',c='k');ax[0].set_xticks(x);ax[0].set_xticklabels(ks,rotation=30,fontsize=7);ax[0].set_ylabel('km');ax[0].legend(fontsize=7)
for rn in RULES:
  v=[g['F_heavy'] for g in grid if g['rule']==rn]; ax[1].plot([rn]*len(v),v,'o',alpha=.5)
ax[1].tick_params(axis='x',rotation=30,labelsize=7);ax[1].set_ylabel('F_heavy (model quantity)')
plt.tight_layout();plt.savefig('fig_conquest_reach.png',dpi=110)
