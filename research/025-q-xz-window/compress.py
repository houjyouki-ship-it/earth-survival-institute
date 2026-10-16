"""研究025 圧縮の部分標本：スナップショットのコミットの tarball を取得し、LZMA2 の辞書を変えて圧縮後バイト数を測る。
入力 data/sample.json、出力 data/compress.json（tarball は /tmp に置き、保存しない）。seed=20261025。"""
import json, os, io, tarfile, lzma, random, math, subprocess, time
random.seed(20261025)
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(f"{HERE}/data/sample.json"))
BIN_EXT = set("png jpg jpeg gif bmp ico tif tiff psd pdf zip gz tgz bz2 xz 7z rar jar war class so dll exe dylib a o lib bin dat db sqlite sqlite3 mp3 mp4 wav ogg flac avi mov mkv webm woff woff2 ttf otf eot pyc pkl npy npz h5 hdf5 onnx pt pth ckpt pb tflite apk ipa dmg iso img deb rpm msi swf blend fbx unitypackage xlsx xls docx doc pptx ppt keystore jks wasm node".split())
DICTS = [4096, 65536, 262144, 1 << 20, 8 << 20]
ok = [r for r in S["repos"] if r.get("status") == "ok" and not r["truncated"] and 0 < r["tar_bytes"] <= 40e6]
edges = [0, 1e4, 1e5, 1e6, 4e6, 1.6e7, 4.1e7]
pick = []
for a, b in zip(edges[:-1], edges[1:]):
    pool = sorted([r for r in ok if a < r["tar_bytes"] <= b], key=lambda r: r["full_name"])
    random.shuffle(pool); pick += pool[:5]
out = []
for r in pick:
    url = f"https://codeload.github.com/{r['full_name']}/tar.gz/{r['commit']}"
    p = subprocess.run(["curl", "-sfL", "--max-time", "300", url], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        out.append({"full_name": r["full_name"], "status": "download_failed"}); continue
    src = tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz")
    buf = io.BytesIO(); dst = tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT if False else tarfile.GNU_FORMAT)
    for m in src.getmembers():
        if m.isfile():
            ext = m.name.rsplit(".", 1)[-1].lower() if "." in m.name.rsplit("/", 1)[-1] else ""
            if r["stars_now"] < 250 and m.size > 100_000 and ext in BIN_EXT: continue
            dst.addfile(m, src.extractfile(m))
        elif m.isdir():
            dst.addfile(m)
    dst.close(); tb = buf.getvalue()
    sizes = {}
    for d in DICTS:
        t0 = time.time()
        c = lzma.compress(tb, format=lzma.FORMAT_XZ, filters=[{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": d}])
        sizes[str(d)] = len(c)
    rec = {"full_name": r["full_name"], "stratum": r["stratum"], "stars_now": r["stars_now"], "tar_bytes_tree_est": r["tar_bytes"],
           "tar_bytes_actual": len(tb), "xz_bytes_by_dict": sizes, "status": "ok"}
    print(rec, flush=True); out.append(rec)
json.dump({"dicts": DICTS, "repos": out}, open(f"{HERE}/data/compress.json", "w"), indent=1)
