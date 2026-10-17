"""研究035 collect.py：Debian の索引（data/*.xz、deb.debian.org から 2026-10-17 取得）と U・028 census を照合する。
外部への接続はしない（索引は取得済み）。出力 data/match_<suite>.json"""
import lzma, re, json, sys, collections
HERE = __file__.rsplit('/', 1)[0] or '.'
R30 = f"{HERE}/../030-dependency-tail-census/data/pkgs.jsonl"
CEN = f"{HERE}/../028-important-repo-dualwrite/data/census.json"
GH = re.compile(r'github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)', re.I)

def stanzas(path):
    buf = []
    with lzma.open(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.strip() == '':
                if buf: yield buf; buf = []
            else: buf.append(line.rstrip('\n'))
    if buf: yield buf
def fields(st):
    d = {}; key = None
    for l in st:
        if l[:1] in (' ', '\t') and key: d[key] += ' ' + l.strip()
        elif ':' in l:
            key, v = l.split(':', 1); d[key] = v.strip()
    return d
def slug(o, r):
    r = re.sub(r'\.git$', '', r); return f"{o}/{r}".lower()

def index(suite):
    src, binn, prov, ghmap = set(), set(), set(), collections.defaultdict(set)
    for st in stanzas(f"{HERE}/data/Sources_{suite}.xz"):
        d = fields(st); p = d.get('Package');
        if not p: continue
        src.add(p)
        for b in d.get('Binary', '').split(','): binn.add(b.strip())
        for k in ('Homepage', 'Vcs-Git', 'Vcs-Browser'):
            for o, r in GH.findall(d.get(k, '')): ghmap[slug(o, r)].add(p)
    for st in stanzas(f"{HERE}/data/Packages_{suite}.xz"):
        d = fields(st); p = d.get('Package')
        if p: binn.add(p)
        for x in d.get('Provides', '').split(','):
            x = x.strip().split(' ')[0]
            if x: prov.add(x)
        for o, r in GH.findall(d.get('Homepage', '')): ghmap[slug(o, r)].add(d.get('Source', p).split(' ')[0])
    return src, binn, prov, ghmap

def cand_names(reg, name):
    n = name.lower(); c = []
    if reg == 'npmjs.org':
        m = n.lstrip('@').replace('/', '-'); c += ['node-' + m]
        if n.startswith('@'): c += ['node-' + n.split('/')[-1]]
    elif reg == 'pypi.org':
        m = re.sub(r'[-_.]+', '-', n); c += ['python-' + m, 'python3-' + m]
    elif reg == 'rubygems.org': c += ['ruby-' + n.replace('_', '-')]
    elif reg == 'crates.io': c += ['rust-' + n.replace('_', '-'), 'librust-' + n.replace('_', '-') + '-dev']
    elif reg == 'proxy.golang.org':
        m = n.replace('/', '-').replace('.', '-'); c += ['golang-' + m + '-dev']
        p = n.split('/')
        if p[0] == 'github.com' and len(p) >= 3: c += ['golang-github-' + p[1].replace('.', '-') + '-' + p[2].replace('.', '-') + '-dev']
        if p[0] == 'golang.org' and len(p) >= 3: c += ['golang-golang-x-' + p[2] + '-dev']
    elif reg == 'packagist.org': c += ['php-' + n.split('/')[-1]]
    return c

def run(suite):
    src, binn, prov, ghmap = index(suite)
    allnames = src | binn | prov
    pk = collections.defaultdict(list)
    for l in open(R30):
        d = json.loads(l)
        if d.get('rm_full_name'): pk[d['rm_full_name'].lower()].append((d['registry'], d['name']))
    U = json.load(open(f"{HERE}/data/U_main.json"))
    out = {"suite": suite, "n_src": len(src), "n_bin": len(binn), "n_prov": len(prov), "n_gh_slugs": len(ghmap), "U": [], "star": []}
    for it in U:
        if not it['arch']: continue
        s = it['name'].lower(); m1 = sorted(ghmap.get(s, []))
        names = [x for reg, nm in pk.get(s, []) for x in cand_names(reg, nm)]
        m2 = sorted(set(x for x in names if x in allnames))
        out["U"].append({"name": it['name'], "v": it['v'], "tier": it['tier'], "regs": sorted(set(r for r, _ in pk.get(s, []))), "m1": m1, "m2": m2, "n_cand": len(names)})
    cen = json.load(open(CEN))
    for r in cen['repos']:
        s = r['full_name'].lower(); m1 = sorted(ghmap.get(s, []))
        rn = s.split('/')[1]; m3 = sorted(set(x for x in (rn, rn.replace('_', '-'), rn.replace('.', '-')) if x in src or x in binn))
        out["star"].append({"name": r['full_name'], "stars": r.get('stars_now') or 0, "status": r.get('status'), "m1": m1, "m3": m3})
    json.dump(out, open(f"{HERE}/data/match_{suite}.json", 'w'))
    print(suite, out["n_src"], out["n_bin"], out["n_prov"], out["n_gh_slugs"], len(out["U"]), len(out["star"]))

if __name__ == '__main__':
    for s in sys.argv[1:] or ['trixie', 'sid']: run(s)
