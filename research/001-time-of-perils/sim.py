"""001: 「危険の時代」ハザードモデルにおける長期存続確率と介入効果の比較
再現: python3 sim.py  (seed=20261014) -> results.json, fig_*.png
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng=np.random.default_rng(20261014)
N=200_000
# 文献アンカー: Ord(2020)今世紀1/6 -> 年率 h0 = 1-(5/6)^(1/100)
h0_ord=1-(5/6)**(1/100)
# 自然ハザード床: Snyder-Beattie et al.(2019) 上限 1/87,000/yr を「上限」として使用
h_nat=1/87000
# 不確実性(仮定): h0 ~ lognormal(median=h0_ord, sigma=1.0); 減衰時定数 tau ~ lognormal(median=200y, sigma=1.0)
h0=h0_ord*np.exp(rng.normal(0,1.0,N)); tau=200*np.exp(rng.normal(0,1.0,N))
def cumhaz(T,h0,tau,f_h=1.0,f_tau=1.0,hn=h_nat):
    # h(t)=f_h*h0*exp(-t/(f_tau*tau)) + hn  (連続近似, 年率<<1)
    t2=f_tau*tau
    return f_h*h0*t2*(1-np.exp(-T/t2))+hn*T
Ts=np.array([100,1e3,1e4,1e5,1e6])
def S(T,**k): return float(np.mean(np.exp(-cumhaz(T,h0,tau,**k))))
res={"seed":20261014,"N":N,"h0_ord_annual":h0_ord,"h_nat_upper":h_nat}
res["baseline"]={int(T):S(T) for T in Ts}
# 点推定(不確実性なし)
res["point_estimate"]={int(T):float(np.exp(-cumhaz(T,h0_ord,200.0))) for T in Ts}
# 介入比較: A=現在ハザード半減, B=危険の時代を半分の長さに(tau半減), C=自然ハザード床ゼロ
res["A_halve_h0"]={int(T):S(T,f_h=0.5) for T in Ts}
res["B_halve_tau"]={int(T):S(T,f_tau=0.5) for T in Ts}
res["C_no_natural"]={int(T):S(T,hn=0) for T in Ts}
# 仮説H1: 不確実性はS(1e6)を点推定より大きくする(ジェンセン)、かつ生存はh0*tau低位の尾に集中
surv=np.exp(-cumhaz(1e4,h0,tau)); x=h0*tau
q=np.quantile(x,[.25,.5,.75])
res["share_survival_from_lowest_quartile_h0tau"]=float(surv[x<=q[0]].sum()/surv.sum())
# 感度: sigma
sens={}
for s in [0.0,0.5,1.0,1.5,2.0]:
    r=np.random.default_rng(1); a=h0_ord*np.exp(r.normal(0,s,N)); b=200*np.exp(r.normal(0,s,N))
    sens[s]=float(np.mean(np.exp(-cumhaz(1e4,a,b))))
res["sensitivity_sigma_S1e4"]=sens
# 平均一致(mean-matched)版: E[h0*tau]を点推定に固定 (中央値をexp(-sigma^2)倍)
mm={}
for s_ in [0.0,0.5,1.0,1.5,2.0]:
    r=np.random.default_rng(1); a=h0_ord*np.exp(r.normal(0,s_,N)-s_**2/2); b=200*np.exp(r.normal(0,s_,N)-s_**2/2)
    mm[s_]=float(np.mean(np.exp(-cumhaz(1e4,a,b))))
res["sensitivity_sigma_S1e4_mean_matched"]=mm
res["S_1e4_peril_only_point"]=float(np.exp(-h0_ord*200*(1-np.exp(-50))))
json.dump(res,open("results.json","w"),indent=2,ensure_ascii=False)
# 図
tt=np.logspace(1,6,60)
plt.figure(figsize=(7,4.5))
for lab,k in [("baseline",{}),("A: h0 x0.5",{"f_h":.5}),("B: tau x0.5",{"f_tau":.5}),("C: natural=0",{"hn":0})]:
    plt.semilogx(tt,[S(T,**k) for T in tt],label=lab)
plt.semilogx(tt,np.exp(-cumhaz(tt,h0_ord,200.0)),"k--",label="point estimate")
plt.xlabel("years from now"); plt.ylabel("P(survival)"); plt.legend(); plt.grid(alpha=.3)
plt.title("Time-of-perils model (seed 20261014)"); plt.tight_layout(); plt.savefig("fig_survival.png",dpi=130)
plt.figure(figsize=(6,4)); plt.hist(np.log10(x),bins=80,weights=surv,density=True,alpha=.6,label="weighted by survival 1e4y")
plt.hist(np.log10(x),bins=80,density=True,alpha=.4,label="prior"); plt.xlabel("log10(h0*tau) = cumulative peril hazard"); plt.legend(); plt.tight_layout(); plt.savefig("fig_posterior_tail.png",dpi=130)
print(json.dumps(res,indent=1))
