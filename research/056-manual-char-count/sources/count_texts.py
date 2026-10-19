"""Count characters in Project Gutenberg plain texts (body between START/END markers).
chars = length after collapsing every whitespace run to one space. Run: python3 count_texts.py"""
import re, json, hashlib, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for p in sorted(glob.glob(os.path.join(HERE, "gutenberg", "pg*.txt"))):
    raw = open(p, encoding="utf-8", errors="replace").read()
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\n", raw)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", raw)
    body = raw[s.end():e.start()] if s and e else raw
    title = re.search(r"Title: (.*)", raw); title = title.group(1).strip() if title else "?"
    flat = re.sub(r"\s+", " ", body).strip()
    # footnote text: Gutenberg convention "[Footnote ...]" blocks, or lines in a FOOTNOTES section
    fn = sum(len(re.sub(r"\s+", " ", m)) for m in re.findall(r"\[Footnote[^\]]*\]", body, flags=re.S))
    ill = len(re.findall(r"\[Illustration", body, flags=re.I))
    figs = len(re.findall(r"\bFig(?:ure|\.)\s*\d+", body))
    out[os.path.basename(p)[2:-4]] = {"title": title, "chars": len(flat), "chars_footnote_blocks": fn,
        "illustration_markers": ill, "fig_refs": figs, "sha256": hashlib.sha256(raw.encode("utf-8", "replace")).hexdigest(),
        "markers_found": bool(s and e)}
json.dump(out, open(os.path.join(HERE, "gutenberg_counts.json"), "w"), indent=1, ensure_ascii=False)
for k, v in out.items(): print(k, v["chars"], v["chars_footnote_blocks"], v["illustration_markers"], v["fig_refs"], v["markers_found"], v["title"][:70])
