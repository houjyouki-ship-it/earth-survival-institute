"""研究071 の資料の取得（2026-10-20 に実行したコマンドの記録。再実行するとネットに接続する）。
Wikipedia 英語版の MediaWiki API から本文の平文（extracts）と座標（coordinates）を取る。"""
import subprocess, json, urllib.parse, sys
UA = "EarthSurvivalInstitute/1.0 (research; contact via operator)"
PAGES = ["Xuanzang", "Yijing_(monk)", "Faxian", "Japanese_missions_to_Tang_China", "Kibi_no_Makibi", "Kūkai", "Ennin",
         "Poggio_Bracciolini", "Giovanni_Aurispa", "Gerard_of_Cremona", "Adelard_of_Bath", "Hunayn_ibn_Ishaq", "Thonmi_Sambhota",
         "Marpa_Lotsawa", "Song_Yun", "Benedict_Biscop", "Lupus_Servatus", "Hyecho", "Petrarch", "Basilios_Bessarion",
         "Translation_movement", "House_of_Wisdom", "Japanese_missions_to_Sui_China", "Saichō", "Francesco_Filelfo", "Guarino_da_Verona",
         "Ceolfrith", "Codex_Amiatinus", "Atiśa", "Rinchen_Zangpo"]
PLACES = ["Xi'an", "Nalanda", "Nara,_Nara", "Kyoto", "Florence", "Rome", "Constantinople", "Toledo,_Spain", "Baghdad", "Lhasa",
          "Monkwearmouth–Jarrow_Abbey", "Ferrières-en-Gâtinais", "York", "Cluny_Abbey", "Abbey_of_Saint_Gall", "Fulda", "Cremona",
          "Bath,_Somerset", "Gyeongju", "Luoyang", "Venice", "Palembang", "Srivijaya", "Vikramashila", "Mount_Hiei", "Mount_Kōya",
          "Langres", "Montecassino", "Tamralipta", "Sri_Lanka", "Pataliputra", "Lhodrak_County", "Hangzhou", "Ningbo", "Yangzhou",
          "Kanchipuram", "Peshawar", "Udyana", "Tours", "Cologne", "Mosul", "Alexandria"]
def get(title, kind):
    q = {"action": "query", "titles": title.replace("_", " "), "format": "json", "formatversion": 2, "redirects": 1}
    if kind == "text": q.update(prop="extracts", explaintext=1)
    else: q.update(prop="coordinates")
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(q)
    out = subprocess.run(["curl", "-sL", "-A", UA, url], capture_output=True).stdout
    return json.loads(out)
if __name__ == "__main__" and "--places2" not in sys.argv:
    T = {p: get(p, "text") for p in PAGES}
    json.dump(T, open("wiki_texts.json", "w"), ensure_ascii=False, indent=0)
    C = {p: get(p, "coord") for p in PLACES}
    json.dump(C, open("wiki_coords.json", "w"), ensure_ascii=False, indent=0)
PLACES2 = ["Nalanda_district", "Bargaon,_Nalanda", "Jarrow", "Sunderland", "Verona", "Savona", "Srinagar", "Bad_Hersfeld", "Nanjing", "Chengdu",
           "Tholing_Monastery", "Mount_Tiantai", "Konstanz", "Reichenau_Island", "Shaoxing", "Bodh_Gaya", "Tamluk", "Kathmandu", "Dunhuang", "Gyantse"]
if __name__ == "__main__" and "--places2" in sys.argv:
    C2 = {p: get(p, "coord") for p in PLACES2}
    json.dump(C2, open("wiki_coords2.json", "w"), ensure_ascii=False, indent=0)
