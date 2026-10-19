import json,re,html,os
H=os.path.dirname(os.path.abspath(__file__))
def text(f):
    t=open(os.path.join(H,f),encoding='utf8',errors='ignore').read()
    if f.endswith('.json'): t=list(json.loads(t)['query']['pages'].values())[0]['extract']
    elif f.endswith('.html'): t=html.unescape(re.sub(r'<[^>]+>',' ',re.sub(r'<script.*?</script>|<style.*?</style>','',t,flags=re.S))) if '2023-07-01' not in t or True else t
    return re.sub(r'\s+',' ',t.replace('\u2019',"'").replace('-\n',''))
ok=0; ex=json.load(open(os.path.join(H,'excerpts.json')))
raw={}
for e in ex:
    t=text(e['file']); q=re.sub(r'\s+',' ',e['quote'].replace('\u2019',"'"))
    hit = q in t or q in re.sub(r'\s+',' ',open(os.path.join(H,e['file']),encoding='utf8',errors='ignore').read())
    ok+=hit; print('OK ' if hit else 'MISS', e['file'], q[:70])
print(ok,'/',len(ex))
