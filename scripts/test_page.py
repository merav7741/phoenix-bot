import sys
import requests
from bs4 import BeautifulSoup

url = sys.argv[1]
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; HackathonResearchBot/1.0)"}, timeout=30)
print("Status:", r.status_code, "| HTML size:", len(r.text))

soup = BeautifulSoup(r.text, "lxml")
for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()
text = soup.get_text("\n", strip=True)
print("Text length:", len(text))
print("---- first 1500 chars ----")
print(text[:1500])