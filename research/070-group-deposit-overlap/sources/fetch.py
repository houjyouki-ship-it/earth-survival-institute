"""研究070 の資料の取得（2026-10-20 14:38〜14:42Z に実行したコマンドの記録。再実行するとネットに接続する）。"""
import subprocess
UA = "EarthSurvivalInstitute/1.0 (research)"
CMDS = [
 ["curl", "-sL", "-A", "Mozilla/5.0", "https://www.nature.com/articles/s43016-022-00573-0", "-o", "xia2022.html"],
 ["curl", "-sL", "-A", "Mozilla/5.0", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43016-022-00573-0/MediaObjects/43016_2022_573_MOESM1_ESM.pdf", "-o", "xia_moesm1.pdf"],
 ["curl", "-sL", "-A", "Mozilla/5.0", "https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs43016-022-00573-0/MediaObjects/43016_2022_573_MOESM2_ESM.pdf", "-o", "xia_moesm2.pdf"],
 ["curl", "-sL", "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/master/all/all.csv", "-o", "iso_regions.csv"],
]
for p in ["List_of_largest_libraries", "List_of_national_libraries", "List_of_national_and_state_libraries", "Southern_Hemisphere", "Legal_deposit", "Arctic_World_Archive", "Svalbard_Global_Seed_Vault"]:
    CMDS.append(["curl", "-sL", "-A", UA, f"https://en.wikipedia.org/w/api.php?action=parse&page={p}&prop=wikitext&format=json&formatversion=2", "-o", f"wiki_{p}.json"])
for pmid in (36464495, 33886124):
    CMDS.append(["curl", "-s", f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pmid}&rettype=abstract&retmode=text", "-o", f"pm_{pmid}.txt"])
if __name__ == "__main__":
    for c in CMDS: subprocess.run(c, check=True)
    subprocess.run(["pdftotext", "-layout", "xia_moesm1.pdf", "xia_moesm1.txt"], check=True)
