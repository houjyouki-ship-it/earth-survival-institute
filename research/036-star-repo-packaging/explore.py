"""研究036 explore.py（事前登録外、判定には使わない）：文章型の分類の妥当性、改名後の名前・nix の purl での追加一致。"""
import json, re, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
M = json.load(open(f"{HERE}/data/match_star.json")); S = M["star"]
st = np.array([float(s["stars"]) for s in S]); w = st / st.sum()
d_any = np.array([s["deb"] or bool(s["brew"]) or bool(s["nix"]) for s in S])
KW = re.compile(r'awesome|tutorial|interview|roadmap|book|cheatsheet|course|guide|resources|list|learn|primer|algorithms|curriculum|free-|papers|notes|questions|examples|100-days|exercises', re.I)
CONT_LANG = {None, "Markdown", "TeX"}; SOFT_LANG = {None, "Markdown", "HTML", "TeX", "Jupyter Notebook"}
def txt(s): m = s["meta"]; return " ".join([s["name"].split('/')[1], m["desc"], " ".join(m["topics"])])
c_pre = np.array([s["meta"]["lang"] in CONT_LANG or (bool(re.search(r'awesome|tutorial|interview|roadmap|book|cheatsheet|course|guide|resources|list', txt(s), re.I)) and s["meta"]["lang"] in SOFT_LANG) for s in S])
c_wide = np.array([s["meta"]["lang"] in CONT_LANG or bool(KW.search(txt(s))) for s in S])
E = {}
E["content_wide"] = {"share_all": float(w[c_wide].sum()), "share_of_uncovered": float(w[~d_any & c_wide].sum() / w[~d_any].sum()),
                     "uncovered_software_share": float(w[~d_any & ~c_wide].sum()), "coverage_among_software": float(w[d_any & ~c_wide].sum() / w[~c_wide].sum())}
# 上位50件の未収録（星順）を両方の分類で並べる（目で確かめるため）
top = [i for i in np.argsort(-st) if not d_any[i]][:50]
E["top50_uncovered"] = [{"name": S[i]["name"], "stars": int(st[i]), "lang": S[i]["meta"]["lang"], "pre": bool(c_pre[i]), "wide": bool(c_wide[i]),
                         "desc": S[i]["meta"]["desc"][:90]} for i in top]
E["top50_uncovered_share_wide_content"] = float(sum(w[i] for i in top if c_wide[i]) / sum(w[i] for i in top))
# 改名・nix の purl での追加一致
nwo = {}
B = json.load(open(f"{HERE}/data/formula.json")); C = json.load(open(f"{HERE}/data/cask.json"))
GH = re.compile(r'github\.com[/:]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)', re.I)
bs = set()
for f in B: bs |= {f"{o}/{re.sub(r'.git$', '', r)}".lower() for o, r in GH.findall(f.get("homepage", "") or "" ) + GH.findall(json.dumps(f.get("urls") or {}))}
for c in C: bs |= {f"{o}/{re.sub(r'.git$', '', r)}".lower() for o, r in GH.findall((c.get("homepage") or "") + " " + (c.get("url") or ""))}
renamed = [i for i, s in enumerate(S) if s["meta"]["nwo"].lower() != s["name"].lower()]
add_rn = [i for i in renamed if not d_any[i] and S[i]["meta"]["nwo"].lower() in bs]
add_purl = [i for i, s in enumerate(S) if not d_any[i] and s["nix_purl"]]
E["renamed"] = {"n": len(renamed), "n_new_brew_hit": len(add_rn), "weight_new": float(w[add_rn].sum()) if add_rn else 0.0, "names": [S[i]["name"] + "→" + S[i]["meta"]["nwo"] for i in add_rn]}
E["nix_purl"] = {"n_new": len(add_purl), "weight_new": float(w[add_purl].sum()) if add_purl else 0.0}
# fork 数（GitHub 上の写し）と星の比：文章型とソフトで比べる
fk = np.array([s["meta"]["forks"] for s in S], float)
E["forks_per_star_median"] = {"content_wide": float(np.median(fk[c_wide] / st[c_wide])), "software": float(np.median(fk[~c_wide] / st[~c_wide])),
                              "uncovered_software": float(np.median(fk[~c_wide & ~d_any] / st[~c_wide & ~d_any])), "covered": float(np.median(fk[d_any] / st[d_any]))}
json.dump(E, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in E.items() if k != "top50_uncovered"}, ensure_ascii=False, indent=1))
for x in E["top50_uncovered"][:50]: print(x["stars"], x["name"], x["lang"], "pre" if x["pre"] else "-", "wide" if x["wide"] else "-", "|", x["desc"])

# ---- 追加（同じく事前登録外）：名前一致（brew・nix）と registry の package（030 の pkgs.jsonl）による取りこぼしの見積もり ----
import brotli
fn = {f["name"].lower() for f in B} | {c["token"].lower() for c in C}
nx = json.loads(brotli.decompress(open(f"{HERE}/data/nix_packages.json.br", "rb").read()))["packages"]
pn = {(p.get("pname") or "").lower() for p in nx.values()}
def rn(s): return s["name"].split('/')[1].lower()
nm = np.array([(rn(s) in fn or rn(s) in pn) for s in S])
reg = {}
for l in open(f"{HERE}/../030-dependency-tail-census/data/pkgs.jsonl"):
    d = json.loads(l)
    if d.get("rm_full_name"): reg.setdefault(d["rm_full_name"].lower(), []).append(d["registry"])
rg = np.array([s["name"].lower() in reg for s in S])
c3 = ~c_wide
E["recall_probe"] = {"name_match_new_weight": float(w[nm & ~d_any].sum()), "name_match_new_examples": [S[i]["name"] for i in np.argsort(-st) if nm[i] and not d_any[i]][:15],
    "registry_new_weight": float(w[rg & ~d_any].sum()), "registry_new_examples": [S[i]["name"] for i in np.argsort(-st) if rg[i] and not d_any[i]][:15],
    "union_plus_name_plus_registry": float(w[d_any | nm | rg].sum()),
    "coverage_among_software_wide_plus": float(w[(d_any | nm | rg) & c3].sum() / w[c3].sum())}
json.dump(E, open(f"{HERE}/explore_results.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(E["recall_probe"], ensure_ascii=False, indent=1))
