import runpy, json, sys, os, numpy as np
os.chdir('/tmp/rep035/031-lowtier-dependency-inclusion'); sys.argv=['sim.py']
ns = runpy.run_path('sim.py', run_name='__main__')
U = ns['build_U']("main")
out=[]
for k in range(len(U['v'])):
    out.append({"name":U['names'][k],"w":float(U['w'][k]),"D1":float(U['D'][k]),"v":float(U['v'][k]),"arch":bool(U['arch'][k]),"tier":str(U['tier'][k]),"cls":str(U['cls'][k])})
json.dump(out, open('/tmp/rep035/U_main.json','w'))
print(len(out), sum(o['arch'] for o in out))
