"""Xia 2022 の 150 Tg の CLM5 格子収量（095 の sources/nw150.nc, OSF https://osf.io/yrbse/）から、年ごとの4大作物の世界の生産の比（収量×cos緯度の代理、面積なし）を出す。"""
import json, numpy as np, netCDF4 as nc, os
S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "095-farming-halt-correlation", "sources") + "/"
L = lambda f: np.ma.filled(nc.Dataset(S+f)['yield'][:].astype(float), np.nan)
Y = L('nw150.nc'); C = np.concatenate([L('ctrl03.nc'), L('ctrl03m02.nc')])
lat = np.asarray(nc.Dataset(S+'nw150.nc')['lat'][:]); w = np.cos(np.deg2rad(lat))[:, None]
crops = {"Corn": (1, 2), "Rice": (4, 5), "Soy": (7, 8), "Wheat": (10, 11)}
Cm = np.nanmean(C, axis=0)
out = {}
for nm, j in crops.items():
    num = np.zeros(Y.shape[0]); den = 0.0
    for jj in j:
        c = np.nan_to_num(Cm[jj]); m = c > 0.1
        num += np.array([np.sum(np.nan_to_num(Y[t, jj]) * w * m) for t in range(Y.shape[0])]); den += np.sum(c * w * m)
    out[nm] = (num / den).tolist()
eq = np.mean([out[k] for k in crops], axis=0).tolist()
def ctrl_ratio(t):
    v=[]
    for nm,j in crops.items():
        num=sum(np.sum(np.nan_to_num(C[t,jj])*w*(np.nan_to_num(Cm[jj])>0.1)) for jj in j); den=sum(np.sum(np.nan_to_num(Cm[jj])*w*(np.nan_to_num(Cm[jj])>0.1)) for jj in j); v.append(num/den)
    return float(np.mean(v))
cr=[ctrl_ratio(t) for t in range(C.shape[0])]; cv = float(np.min(cr))
res = {"per_crop_ratio": out, "equal_mean_ratio": eq, "ctrl_min_ratio": float(cv),
       "years_below": {th: int(sum(1 for x in eq if x < th)) for th in (1.0, 0.95, 0.9, 0.8, 0.5)},
       "argmin_year": int(np.argmin(eq)) + 1, "min": float(min(eq))}
json.dump(res, open("nw_years.json", "w"), indent=1); print(json.dumps(res, indent=1))
