import re,json
L=open('sources/s1.txt').read().split('\n')
starts=[i for i,l in enumerate(L) if re.match(r'\s*Table S[4-9][.:]',l)]
starts.append(len(L))
out={}
num=r'(-?\d+(?:\.\d+)?)'
for a,b in zip(starts,starts[1:]):
    t=re.match(r'\s*Table (S\d)',L[a]).group(1)
    cap=' '.join(x.strip() for x in L[a:a+6])
    rows={}
    for l in L[a:b]:
        m=re.match(r'^\s*([A-Za-z][A-Za-z .,\'()&\-]*?)\s+'+r'\s+'.join([num]*9)+r'\s*$',l)
        if m:
            nm=m.group(1).strip()
            if nm=='China' and 'China' in rows: nm='Taiwan'
            if nm=='Taiwan, province of China': nm='Taiwan'
            rows[nm]=[float(x) for x in m.groups()[1:]]
    if not rows: continue
    out[t]={'caption':cap[:400],'cols':['pop','5','16','27','37','47','150','150hw','150tw'],'rows':rows}
    tot=rows.get('Total')
    print(t,len(rows),tot, cap[150:260])
json.dump(out,open('xia_tables.json','w'),ensure_ascii=False,indent=0)
