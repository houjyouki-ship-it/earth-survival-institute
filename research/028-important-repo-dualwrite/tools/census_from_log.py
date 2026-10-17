"""研究028 逸脱の処理：collect.py は chromium/chromium の truncated な木の再帰走査で API 上限を使い切りかけたため
03:07 UTC 頃に停止した（census.json は終了時にしか書かれない設計だった）。標準出力のログ data/collect_log.txt から
census.json を再構成する。ログ1行 = full_name stars status tar_bytes truncated。commit は compress.py が必要時に取り直す。"""
import json, os
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURED = ["dotnet/core", "torvalds/linux", "python/cpython", "bitcoin/bitcoin", "rails/rails", "docker/machine",
            "openssl/openssl", "nodejs/node", "Homebrew/brew", "php/php-src", "twbs/bootstrap", "microsoft/TypeScript",
            "apache/hadoop", "Alamofire/Alamofire", "v8/v8"]
ALIAS = {"docker-archive-public/docker.machine": "docker/machine"}  # repos/docker/machine は改名先を返した
lines = open(f"{HERE}/data/collect_log.txt").read().splitlines()
assert lines[0].startswith("listed 1366 of 1366")
repos = []
for ln in lines[1:]:
    p = ln.split()
    if len(p) != 5: continue
    name, stars, status, tb, tr = p
    d = {"full_name": name, "stars_now": int(stars), "status": status,
         "featured": name in FEATURED or ALIAS.get(name) in FEATURED}
    if status == "ok": d.update(tar_bytes=int(tb), truncated=(tr == "True"))
    repos.append(d)
repos.append({"full_name": "chromium/chromium", "stars_now": 25045, "api_size_kb_now": 67517178, "status": "walk_aborted", "featured": False, "note": "木の再帰走査を中断。TAR は不明。sim.py は主設定 3e9 B【推測】、上限感度は api_size_kb_now×1024"})  # stars・size は 03:08 UTC に repos/chromium/chromium で取得
json.dump({"until": "2020-02-02T00:00:00Z", "search_total_count": 1366, "n_listed": 1366, "featured": FEATURED, "alias": ALIAS,
           "reconstructed_from_log": True, "repos": repos}, open(f"{HERE}/data/census.json", "w"), indent=1)
print(len(repos), sum(r["status"] == "ok" for r in repos), sum(r["featured"] for r in repos))
