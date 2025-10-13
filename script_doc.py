import requests
from bs4 import BeautifulSoup
import os, urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://www.aca.ntu.edu.tw/w/aca/CDRules"
save_dir = "./ntu_rules_pdfs"
os.makedirs(save_dir, exist_ok=True)

resp = requests.get(url, verify=False)  # ⚠️ 關閉 SSL 驗證
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")

links = []
for a in soup.find_all("a", href=True):
    href = a["href"]
    text = a.get_text(strip=True)
    if href.lower().endswith((".pdf", ".odt")):
        full_url = requests.compat.urljoin(url, href)
        links.append((text, full_url))

for i, (text, link) in enumerate(links, 1):
    print(f"{i}. {text}: {link}")
    fname = os.path.basename(link.split("?")[0])
    path = os.path.join(save_dir, fname)
    if not os.path.exists(path):
        r = requests.get(link, verify=False)
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"   -> Saved to {path}")
