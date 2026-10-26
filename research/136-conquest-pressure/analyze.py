"""研究136：Xia 2022 の国別の餓死の表（補足 S4〜S9）から外の征服の圧力 P_ext を出し、c_C と F_heavy に変える。
F_heavy は生存確率ではない（模型の中の量）。"""
import json, math, numpy as np
X=json.load(open('xia_tables.json'))
CEN={n:(a,b) for c,a,b,n in json.load(open('centroids.json'))}
ALIAS={'Republic of Congo':'Congo [Republic]','Myanmar':'Myanmar [Burma]'}
cc=json.load(open('../134-modern-downhill/results.json'))['cC_curve']
assert [round(f,4) for f in cc['F']]==[1.911,1.9646,2.0219,2.1664]
def Fh(c): return float(np.interp(c,cc['c'],cc['F']))
C_PRESS=0.723; C_NO=0.09
COLS=X['S4']['cols']
def hav(p,q):
    la1,lo1,la2,lo2=map(math.radians,(*p,*q))
    h=math.sin((la2-la1)/2)**2+math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return 2*6371*math.asin(math.sqrt(h))
def run(table,col,D=2000,s_int=0.25,s_heavy=0.75,popmin=10.0):
    r=X[table]['rows']; j=COLS.index(col)
    nat=[]
    for k,v in r.items():
        if k=='Total': continue
        pop=v[0]; s=min(v[j]/pop,1.0) if pop>0 else 0
        nat.append(dict(n=k,pop=pop,s=s,ll=CEN[ALIAS.get(k,k)]))
    intact=[d for d in nat if d['s']<s_int and d['pop']>=popmin]
    heavy=[d for d in nat if d['s']>=s_heavy]
    W=P=0.0; press=[]
    for d in heavy:
        w=d['pop']*(1-d['s'])
        if D is None: ok=len(intact)>0
        else: ok=any(hav(d['ll'],e['ll'])<=D for e in intact)
        W+=w; P+=w*ok
        if ok: press.append(d['n'])
    Pext=P/W if W>0 else float('nan')
    cC=Pext*C_PRESS+(1-Pext)*C_NO
    poptot=sum(d['pop'] for d in nat)
    return dict(table=table,col=col,D=D,s_int=s_int,s_heavy=s_heavy,popmin=popmin,
        n_intact=len(intact),intact=sorted([(e['n'],round(e['pop'],1),round(e['s'],3),round(e['ll'][0],1)) for e in intact],key=lambda x:-x[1]),
        n_heavy=len(heavy),heavy_survivors_M=round(W,1),Pext=Pext,cC=cC,F_heavy=Fh(cC),
        share_pop_intact=sum(e['pop'] for e in intact)/poptot,
        share_pop_intact_any=sum(d['pop'] for d in nat if d['s']<s_int)/poptot,
        pressured=press)
OUT={'note':'F_heavy は生存確率ではない。P_ext は国の重心の距離による代理。','main':run('S4','150')}
m=OUT['main']
print('main',{k:m[k] for k in ['n_intact','n_heavy','heavy_survivors_M','Pext','cC','F_heavy','share_pop_intact']})
print('intact',m['intact']); print('pressured',m['pressured'])
grid=[]
for t in X:
  for col in ['150','150hw','150tw','47','37','27']:
    for D in [1000,2000,4000,None]:
      for sh in [0.5,0.75,0.9]:
        g=run(t,col,D=D,s_heavy=sh); g.pop('pressured'); g.pop('intact'); grid.append(g)
OUT['grid']=grid
def pick(**kw): return [g for g in grid if all(g[k]==v for k,v in kw.items())]
print('\n150Tg main thresholds, by table x D:')
for t in X:
  print(t,[ (D, round(pick(table=t,col='150',D=D,s_heavy=0.75)[0]['Pext'],3)) for D in [1000,2000,4000,None]])
for col in ['47','37','27']:
  g=pick(table='S4',col=col,D=2000,s_heavy=0.75)[0]
  print(col,'share_pop_intact',round(g['share_pop_intact'],3),'any size',round(g['share_pop_intact_any'],3),'Pext',round(g['Pext'],3),'nheavy',g['n_heavy'])
# 仮説
F150=[g['F_heavy'] for g in grid if g['col'].startswith('150')]
P150=[g['Pext'] for g in grid if g['col'].startswith('150') and g['D'] is not None]
g47=pick(table='S4',col='47',D=2000,s_heavy=0.75)[0]
H={'H148a':dict(val=m['Pext'],ok=m['Pext']<0.5),
   'H148b':dict(val=[(e[0],e[3]) for e in m['intact']],ok=all(e[3]<30 for e in m['intact'])),
   'H148c':dict(val=m['F_heavy'],ok=m['F_heavy']<2.0),
   'H148d':dict(val=g47['share_pop_intact'],ok=g47['share_pop_intact']>=0.5)}
OUT['hyp']=H; OUT['range_150']=dict(Pext=[min(P150),max(P150)],F=[min(F150),max(F150)])
print('\nhyp',json.dumps(H,ensure_ascii=False)); print('range150',OUT['range_150'])
json.dump(OUT,open('results.json','w'),ensure_ascii=False,indent=1,default=str)
# 図
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(11,4))
for t in X:
  ax[0].plot([1000,2000,4000,8000],[pick(table=t,col='150',D=D,s_heavy=0.75)[0]['Pext'] for D in [1000,2000,4000,None]],'o-',label=t)
ax[0].set_xscale('log');ax[0].set_xticks([1000,2000,4000,8000]);ax[0].set_xticklabels(['1000','2000','4000','any'])
ax[0].set_xlabel('distance to intact nation (km)');ax[0].set_ylabel('P_ext (150 Tg)');ax[0].legend(fontsize=7)
cs=np.linspace(0,1,50);ax[1].plot(cs,[Fh(c*C_PRESS+(1-c)*C_NO) for c in cs]);ax[1].set_xlabel('P_ext');ax[1].set_ylabel('F_heavy (model quantity)')
ax[1].axvline(m['Pext'],ls='--',c='r')
plt.tight_layout();plt.savefig('fig_conquest_pressure.png',dpi=110)
