"""研究036 collect.py：Homebrew（formula・cask・365日 install）、nixpkgs（nixos-unstable packages.json）、
GitHub GraphQL のメタデータ（gh、読み取りのみ・キャッシュ data/gh_meta.json）を、028 の ≥20,000 星 1,368件と照合する。
Debian の照合は 035 の data/match_trixie.json と Sources/Packages をそのまま使う。出力 data/match_star.json"""
import json, re, os, sys, subprocess, collections, lzma, brotli
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.dirname(HERE)
D35 = f"{RES}/035-orphan-share-distro"
GH = re.compile(r'github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)', re.I)
def slugs(s):
    out = set()
    if isinstance(s, list): s = " ".join(map(str, s))
    for o, r in GH.findall(s or ''):
        r = re.sub(r'\.git$', '', r); out.add(f"{o}/{r}".lower())
    return out
cen = json.load(open(f"{RES}/028-important-repo-dualwrite/data/census.json"))["repos"]
names = [r["full_name"] for r in cen]

# ---- GitHub メタデータ（キャッシュ） ----
metap = f"{HERE}/data/gh_meta.json"
meta = json.load(open(metap)) if os.path.exists(metap) else {}
todo = [n for n in names if meta.get(n) is None and n not in meta.get("_notfound", [])]
for attempt in range(3):
    todo = [n for n in names if meta.get(n) is None and n not in meta.get("_notfound", [])]
    if not todo: break
    for i in range(0, len(todo), 25):
        q = []; batch = todo[i:i + 25]
        for j, n in enumerate(batch):
            o, r = n.split('/', 1)
            q.append(f'r{j}: repository(owner:{json.dumps(o)}, name:{json.dumps(r)}) {{ nameWithOwner primaryLanguage {{ name }} '
                     f'repositoryTopics(first:20) {{ nodes {{ topic {{ name }} }} }} forkCount isArchived description stargazerCount }}')
        res = subprocess.run(["gh", "api", "graphql", "-f", "query={" + " ".join(q) + "}"], capture_output=True, text=True)
        try: js = json.loads(res.stdout or "{}")
        except Exception: js = {}
        d = js.get("data") or {}
        for j, n in enumerate(batch):
            x = d.get(f"r{j}")
            if x is None:
                if d and any(e.get("type") == "NOT_FOUND" and e.get("path") == [f"r{j}"] for e in js.get("errors", [])):
                    meta.setdefault("_notfound", []).append(n)
                continue
            meta[n] = {"nwo": x["nameWithOwner"], "lang": (x["primaryLanguage"] or {}).get("name"),
                     "topics": [t["topic"]["name"] for t in x["repositoryTopics"]["nodes"]], "forks": x["forkCount"],
                     "archived": x["isArchived"], "desc": x["description"] or "", "stars_api": x["stargazerCount"]}
        print("gh", attempt, i, len(d), file=sys.stderr)
    json.dump(meta, open(metap, "w"), ensure_ascii=False)

# ---- Homebrew ----
F = json.load(open(f"{HERE}/data/formula.json")); C = json.load(open(f"{HERE}/data/cask.json"))
def an(path, key):
    c = collections.Counter()
    for it in json.load(open(path))["items"]:
        c[it[key].split(' ')[0]] += int(it["count"].replace(',', ''))
    return c
AF = an(f"{HERE}/data/an_formula_365.json", "formula"); AC = an(f"{HERE}/data/an_cask_365.json", "cask")
brew = collections.defaultdict(list)   # slug -> [(kind, name, installs)]
for f in F:
    s = slugs(f.get("homepage")) | slugs(json.dumps(f.get("urls") or {})) | slugs(json.dumps(f.get("head") or ""))
    for x in s: brew[x].append(("formula", f["name"], AF.get(f["name"], 0)))
for c in C:
    s = slugs(c.get("homepage")) | slugs(c.get("url"))
    for x in s: brew[x].append(("cask", c["token"], AC.get(c["token"], 0)))
# ---- nixpkgs ----
nx = json.loads(brotli.decompress(open(f"{HERE}/data/nix_packages.json.br", "rb").read()))["packages"]
nix = collections.defaultdict(set); nixp = collections.defaultdict(set)
for attr, p in nx.items():
    m = p.get("meta", {}); hp = m.get("homepage")
    hp = " ".join(hp) if isinstance(hp, list) else (hp or "")
    for x in slugs(hp) | slugs(m.get("downloadPage")) | slugs(m.get("changelog")): nix[x].add(p.get("pname") or attr)
    for pu in (m.get("identifiers", {}) or {}).get("purls", []) or []:      # 探索用（事前登録外）
        mm = re.match(r'pkg:github/([^/@]+)/([^/@?]+)', pu)
        if mm: nixp[f"{mm.group(1)}/{mm.group(2)}".lower()].add(p.get("pname") or attr)
# ---- Debian の名前集合（M4 用、035 と同じ索引） ----
def debnames(suite):
    s = set()
    for fn in (f"{D35}/data/Sources_{suite}.xz", f"{D35}/data/Packages_{suite}.xz"):
        with lzma.open(fn, "rt", encoding="utf-8", errors="replace") as fh:
            for l in fh:
                if l.startswith("Package:") or l.startswith("Binary:"):
                    for x in l.split(":", 1)[1].split(","): s.add(x.strip())
    return s
DEB = debnames("trixie")
m35 = {s["name"]: s for s in json.load(open(f"{D35}/data/match_trixie.json"))["star"]}
out = {"n_formula": len(F), "n_cask": len(C), "n_nix": len(nx), "an_formula_total": sum(AF.values()), "an_cask_total": sum(AC.values()), "star": []}
for r in cen:
    n = r["full_name"]; s = n.lower(); b = brew.get(s, []); xs = sorted(nix.get(s, []))
    cands = set(x for _, x, _ in b) | set(xs)
    cands |= set(c.lower() for c in cands)
    m4 = sorted(x for x in cands if x in DEB)
    d35 = m35[n]
    out["star"].append({"name": n, "stars": r.get("stars_now") or 0, "deb": bool(d35["m1"] or d35["m3"]), "deb_m1": d35["m1"], "deb_m3": d35["m3"],
                        "brew": b, "installs": sum(i for _, _, i in b), "nix": xs, "nix_purl": sorted(nixp.get(s, [])), "m4": m4, "meta": meta.get(n)})
json.dump(out, open(f"{HERE}/data/match_star.json", "w"), ensure_ascii=False)
st = out["star"]
print("formula", len(F), "cask", len(C), "nix", len(nx), "brew_hit", sum(bool(x["brew"]) for x in st), "nix_hit", sum(bool(x["nix"]) for x in st),
      "deb_hit", sum(x["deb"] for x in st), "meta_missing", sum(x["meta"] is None for x in st))
