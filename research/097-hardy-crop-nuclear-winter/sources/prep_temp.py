"""NWDC の OSF（https://osf.io/e28gq/）の CAM の月の出力から TS・TSMN を抜き出す。
nw_ur_150_07: https://osf.io/download/wnvxf/ 、nw_cntrl_03: https://osf.io/download/w6qb5/"""
import numpy as np, netCDF4 as nc, glob
D='/workspace/nwdata/scratch/alpine/joco6825/dwinstead_data/'
def load(run, years):
    TS=[];TN=[]
    for y in years:
        for m in range(1,13):
            d=nc.Dataset(f'{D}{run}/{run}.cam.h0.{y:04d}-{m:02d}.nc')
            TS.append(d['TS'][0]); TN.append(d['TSMN'][0])
    return np.array(TS,np.float32).reshape(len(years),12,96,144), np.array(TN,np.float32).reshape(len(years),12,96,144), d
ts,tn,d=load('nw_ur_150_07',range(5,12))
cs,cn,_=load('nw_cntrl_03',range(1,22))
np.savez_compressed('temps.npz', ts150=ts, tn150=tn, tsctl=cs, tnctl=cn, lat=d['lat'][:], lon=d['lon'][:])
print(ts.shape, cs.shape)
