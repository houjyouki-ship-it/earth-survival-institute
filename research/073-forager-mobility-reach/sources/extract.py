"""CRAN binford 0.1.0 の LRB.rdata から移動・面積・座標の列を CSV に抜き出す（pyreadr）。"""
import pyreadr, os
H = os.path.dirname(os.path.abspath(__file__))
d = pyreadr.read_r(os.path.join(H, "binford", "data", "LRB.rdata"))
df = list(d.values())[0]
cols = [c for c in ["society", "name", "wc.area", "latitude", "longitude", "kmov", "dismov", "nomov", "kspmov", "area", "density", "forcol", "polpos"] if c in df.columns]
print(df.shape, cols)
df[cols].to_csv(os.path.join(H, "lrb_mobility.csv"), index=False)
