"""本回開いた Wikipedia の本文（MediaWiki API の plain text、2026-10-18 16:4xZ 取得、fetch.py）から引用を照合して excerpts.json を作る。"""
import json, os
H = os.path.dirname(os.path.abspath(__file__))
Q = [
 ("wp_Aboriginal_Tasmanians.txt", "This left the island's Aboriginal communities geographically isolated from other Aboriginal Australian groups for about 8,000 years."),
 ("wp_Aboriginal_Tasmanians.txt", "Estimates of the population before colonisation generally range from 3,000 to 15,000 people."),
 ("wp_Aboriginal_Tasmanians.txt", "Approximately 4,000 years ago, Aboriginal Tasmanians largely dropped scaled fish from their diet"),
 ("wp_Aboriginal_Tasmanians.txt", "The significance of the disappearance of bone tools (believed to have been primarily used for fishing) and fish in the diet is heavily debated."),
 ("wp_Aboriginal_Tasmanians.txt", "others argue that the change was economic, as large areas of scrub at that time were changing to grassland"),
 ("wp_Sakoku.txt", "Japan was able to acquire the imported goods it required through intermediary trade with the Dutch and through the Ryukyu Islands."),
 ("wp_Sakoku.txt", "Western scientific, technical and medical innovations flowed into Japan through Rangaku"),
 ("wp_Dejima.txt", "From 1715, only two ships were permitted every year, which was reduced to one ship in 1790, and again increased to two ships in 1799."),
 ("wp_Rhodesia.txt", "After the Rhodesian Front began introducing incentives accorded to domestic production, industrial output expanded dramatically."),
 ("wp_Rhodesia.txt", "Portugal served as a conduit for Rhodesian goods, which it exported through Mozambique with false certificates of origin. South Africa, too, refused to observe the UN sanctions."),
 ("wp_Rhodesia.txt", "These economic successes owed a lot to the immigration of skilled labour and backing from South Africa."),
 ("wp_Economy_of_North_Korea.txt", "lack of foreign exchange to purchase spare parts and oil for electricity generation left many factories idle."),
 ("wp_Economy_of_North_Korea.txt", "Industrial capital stock is nearly beyond repair as a result of years of underinvestment and shortages of spare parts."),
 ("wp_Economy_of_North_Korea.txt", "China is North Korea's largest trading partner."),
 ("wp_Tredegar_Iron_Works.txt", "Tredegar supplied about half the artillery used by the Confederate States Army"),
 ("wp_Tredegar_Iron_Works.txt", "Even in the summer of 1861, soon after the beginning of the Civil War, metal was so scarce that the iron works failed to produce a single piece of artillery for an entire month."),
 ("wp_Confederate_States_of_America.txt", "The value of farm implements and machinery in the 1860 Census was $81 million; by 1870, it had diminished by 40 percent and was worth just $48 million. Many old tools had broken through heavy use; new tools were rarely available, and even repairs were difficult"),
 ("wp_José_Gaspar_Rodríguez_de_Francia.txt", "he imposed a ruthless isolation upon Paraguay, interdicting all external trade, and he fostered national industries."),
 ("wp_José_Gaspar_Rodríguez_de_Francia.txt", "Contrary to popular belief, Paraguay was not completely isolated."),
]
out = []
for f, q in Q:
    t = open(os.path.join(H, f), encoding="utf8").read()
    out.append({"url": t.splitlines()[0].replace("URL: ", ""), "file": "sources/" + f, "quote": q, "found_in_fetched_text": q in t})
json.dump({"retrieved": "2026-10-18T16:40-16:50Z (MediaWiki API, en.wikipedia.org)", "excerpts": out}, open(os.path.join(H, "excerpts.json"), "w"), ensure_ascii=False, indent=1)
print(sum(not o["found_in_fetched_text"] for o in out), "not found of", len(out))
for o in out:
    if not o["found_in_fetched_text"]: print(o["quote"][:80])
