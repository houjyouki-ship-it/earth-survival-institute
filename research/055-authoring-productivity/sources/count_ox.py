import re,glob,os,json,html
out={}
for col in sorted(glob.glob('osbooks-*/collections/*.collection.xml')):
    repo=col.split('/')[0]; c=open(col,encoding='utf8').read()
    mods=re.findall(r'document="(m\d+)"',c)
    ch=ch_nomath=figs=0; miss=0
    for m in mods:
        p=f'{repo}/modules/{m}/index.cnxml'
        if not os.path.exists(p): miss+=1; continue
        t=open(p,encoding='utf8').read()
        t=re.sub(r'<metadata.*?</metadata>','',t,flags=re.S)
        figs+=len(re.findall(r'<figure\b',t))
        nm=re.sub(r'<m:math.*?</m:math>|<math.*?</math>',' ',t,flags=re.S)
        def txt(s): s=html.unescape(re.sub(r'<[^>]+>',' ',s)); return len(re.sub(r'\s+',' ',s).strip())
        ch+=txt(t); ch_nomath+=txt(nm)
    out[os.path.basename(col)]={"repo":repo,"modules":len(mods),"missing":miss,"chars_text_incl_math":ch,"chars_text_excl_math":ch_nomath,"figures":figs}
json.dump(out,open('openstax_counts.json','w'),indent=1); print(json.dumps(out,indent=1))
