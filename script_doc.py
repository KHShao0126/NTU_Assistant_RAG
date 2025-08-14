import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# 目標頁面
url = "https://www.aca.ntu.edu.tw/w/aca/RulesSet?subMenuId=21060316015673770"

# 建立儲存資料夾
os.makedirs("ntu_rules_pdfs", exist_ok=True)

# 發送 GET 請求
headers = {
    "User-Agent": "Mozilla/5.0"
}
res = requests.get(url, headers=headers)
res.encoding = "utf-8"  # 確保中文正確顯示
soup = BeautifulSoup(res.text, "html.parser")

# 抓出所有 PDF 連結
pdf_links = soup.find_all("a", href=True)
count = 0

for link in pdf_links:
    href = link["href"]
    if href.endswith(".pdf"):
        pdf_url = urljoin(url, href)
        pdf_name = pdf_url.split("/")[-1]

        print(f"下載中：{pdf_name}")
        pdf_res = requests.get(pdf_url)
        if pdf_res.status_code == 200:
            with open(os.path.join("ntu_rules_pdfs", pdf_name), "wb") as f:
                f.write(pdf_res.content)
                count += 1
        else:
            print(f"⚠️ 無法下載：{pdf_url}")

print(f"\n✅ 完成，共下載 {count} 個 PDF 檔案！")