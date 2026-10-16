"""研究026：025 の compress.py と同じ選び方（seed=20261025）・同じ TAR 再構成で30件の tarball を /tmp/tar026 に置く。
GitHub の公開 codeload から読み取りのみ。TAR は保存しない（大きさと SHA-256 だけ data/tars.json に記録）。"""
import json, os, io, tarfile, random, subprocess, hashlib
random.seed(20261025)
HERE = os.path.dirname(os.path.abspath(__file__)); R025 = f"{HERE}/../../025-q-xz-window"
S = json.load(open(f"{R025}/data/sample.json")); C25 = {r["full_name"]: r for r in json.load(open(f"{R025}/data/compress.json"))["repos"]}
BIN_EXT = set("png jpg jpeg gif bmp ico tif tiff psd pdf zip gz tgz bz2 xz 7z rar jar war class so dll exe dylib a o lib bin dat db sqlite sqlite3 mp3 mp4 wav ogg flac avi mov mkv webm woff woff2 ttf otf eot pyc pkl npy npz h5 hdf5 onnx pt pth ckpt pb tflite apk ipa dmg iso img deb rpm msi swf blend fbx unitypackage xlsx xls docx doc pptx ppt keystore jks wasm node".split())
ok = [r for r in S["repos"] if r.get("status") == "ok" and not r["truncated"] and 0 < r["tar_bytes"] <= 40e6]
edges = [0, 1e4, 1e5, 1e6, 4e6, 1.6e7, 4.1e7]
pick = []
for a, b in zip(edges[:-1], edges[1:]):
    pool = sorted([r for r in ok if a < r["tar_bytes"] <= b], key=lambda r: r["full_name"])
    random.shuffle(pool); pick += pool[:5]
os.makedirs("/tmp/tar026", exist_ok=True); out = []
for i, r in enumerate(pick):
    url = f"https://codeload.github.com/{r['full_name']}/tar.gz/{r['commit']}"
    p = subprocess.run(["curl", "-sfL", "--max-time", "300", url], capture_output=True)
    if p.returncode != 0 or not p.stdout:
        out.append({"full_name": r["full_name"], "status": "download_failed"}); print("fail", r["full_name"], flush=True); continue
    src = tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz")
    buf = io.BytesIO(); dst = tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT)
    for m in src.getmembers():
        if m.isfile():
            ext = m.name.rsplit(".", 1)[-1].lower() if "." in m.name.rsplit("/", 1)[-1] else ""
            if r["stars_now"] < 250 and m.size > 100_000 and ext in BIN_EXT: continue
            dst.addfile(m, src.extractfile(m))
        elif m.isdir():
            dst.addfile(m)
    dst.close(); tb = buf.getvalue()
    fn = f"/tmp/tar026/{i:02d}.tar"; open(fn, "wb").write(tb)
    rec = {"idx": i, "full_name": r["full_name"], "stratum": r["stratum"], "stars_now": r["stars_now"], "commit": r["commit"],
           "tar_bytes_tree_est": r["tar_bytes"], "tar_bytes": len(tb), "sha256": hashlib.sha256(tb).hexdigest(),
           "tar_bytes_025": C25.get(r["full_name"], {}).get("tar_bytes_actual"), "status": "ok"}
    print(rec["idx"], rec["full_name"], rec["tar_bytes"], rec["tar_bytes_025"], flush=True); out.append(rec)
json.dump({"repos": out}, open(f"{HERE}/../data/tars.json", "w"), indent=1)
