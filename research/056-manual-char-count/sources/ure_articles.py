"""Split Ure's Dictionary of Arts, Manufactures and Mines (PG #44562) into headword articles and sum
characters (whitespace collapsed) for stage topics. Headword = line starting with >=3 capitals followed by
space/comma/period/paren. Run: python3 ure_articles.py"""
import re, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
raw = open(os.path.join(HERE, "gutenberg", "pg44562.txt"), encoding="utf-8").read()
s = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\n", raw); e = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK", raw)
body = raw[s.end():e.start()]
lines = body.split("\n")
hw = re.compile(r"^([A-Z][A-Z’'\-]{2,}(?: [A-Z’'\-]{2,}){0,4})(?=[ ,.(;])")
arts = []; cur = None
for ln in lines:
    m = hw.match(ln)
    if m and not ln.isupper():
        cur = {"head": m.group(1), "text": [ln]}; arts.append(cur)
    elif cur is not None: cur["text"].append(ln)
for a in arts: a["chars"] = len(re.sub(r"\s+", " ", "\n".join(a["text"])).strip())
STAGES = {
 "Fe_metal": ["IRON", "STEEL", "CAST IRON", "BLAST", "FURNACE", "FORGE", "ORES", "SMELTING", "COPPER", "TIN", "LEAD", "ASSAY", "METALLURGY", "CHARCOAL", "COKE"],
 "glass": ["GLASS", "GLASS-MAKING", "GLASS CUTTING", "FLINT GLASS", "CROWN GLASS", "PLATE GLASS", "BOTTLE"],
 "lens_optics": ["LENS", "SPECULUM", "OPTICAL", "TELESCOPE", "MIRROR", "SILVERING"],
 "chem": ["SULPHURIC ACID", "NITRIC ACID", "MURIATIC ACID", "SODA", "POTASH", "ALKALI", "SOAP", "LIME", "CHLORINE", "BLEACHING", "GUNPOWDER", "NITRE", "AMMONIA", "SALT", "ALUM"],
 "elec": ["ELECTRICITY", "ELECTRO", "GALVANIC", "VOLTAIC", "MAGNET", "TELEGRAPH"],
 "mach": ["LATHE", "MACHINE", "TURNING", "SCREW", "STEAM-ENGINE", "STEAM ENGINE", "BOILER", "MILL", "WATER-WHEEL", "PUMP"],
 "ceramic_build": ["POTTERY", "PORCELAIN", "BRICK", "MORTAR", "CEMENT", "KILN"],
}
def match(head, keys): return any(head == k or head.startswith(k + " ") or head.startswith(k + "-") or head.startswith(k + ",") for k in keys)
res = {"n_articles": len(arts), "total_chars": sum(a["chars"] for a in arts), "stages": {}}
for st, keys in STAGES.items():
    hits = [(a["head"], a["chars"]) for a in arts if match(a["head"], keys)]
    res["stages"][st] = {"chars": sum(c for _, c in hits), "n": len(hits), "top": sorted(hits, key=lambda x: -x[1])[:8]}
json.dump(res, open(os.path.join(HERE, "ure_stage_counts.json"), "w"), indent=1, ensure_ascii=False)
print(res["n_articles"], res["total_chars"])
for st, v in res["stages"].items(): print(st, v["chars"], v["n"], v["top"][:6])
