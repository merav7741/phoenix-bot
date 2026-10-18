import re
import json
import time
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter

import requests
from bs4 import BeautifulSoup

BASE = "https://www.fnx.co.il"
OUT = Path("data")
OUT.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HackathonResearchBot/1.0)",
    "Accept-Language": "he,en;q=0.8",
}


def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r


def save_robots():
    print("== robots.txt ==")
    r = fetch(f"{BASE}/robots.txt")
    (OUT / "robots.txt").write_text(r.text, encoding="utf-8")
    print(r.text[:3000])
    sitemaps = re.findall(r"(?im)^sitemap:\s*(\S+)", r.text)
    return sitemaps


def collect_urls(sitemap_url, depth=0, seen=None):
    """Reads a sitemap. If it is an index, follows the child sitemaps."""
    seen = seen if seen is not None else set()
    if sitemap_url in seen or depth > 3:
        return []
    seen.add(sitemap_url)

    print("  " * depth + f"reading: {sitemap_url}")
    try:
        r = fetch(sitemap_url)
    except Exception as e:
        print("  " * depth + f"  failed: {e}")
        return []

    soup = BeautifulSoup(r.content, "xml")
    urls = []

    for sm in soup.find_all("sitemap"):
        loc = sm.find("loc")
        if loc:
            time.sleep(1)  # polite crawl rate
            urls += collect_urls(loc.text.strip(), depth + 1, seen)

    for u in soup.find_all("url"):
        loc = u.find("loc")
        if loc:
            urls.append(loc.text.strip())

    return urls


def main():
    sitemaps = save_robots()
    if not sitemaps:
        sitemaps = [f"{BASE}/sitemap.xml"]
        print("\nNo Sitemap line in robots.txt, trying the default one.")

    print("\n== sitemap ==")
    all_urls = []
    for sm in sitemaps:
        all_urls += collect_urls(sm)

    all_urls = sorted(set(all_urls))
    (OUT / "all_urls.txt").write_text("\n".join(all_urls), encoding="utf-8")
    print(f"\nTotal unique URLs: {len(all_urls)}")

    # How the pages are distributed by first folder in the path
    sections = Counter()
    for u in all_urls:
        parts = [p for p in urlparse(u).path.split("/") if p]
        sections[parts[0] if parts else "(root)"] += 1

    print("\n== Top sections ==")
    for name, count in sections.most_common(30):
        print(f"{count:6d}  {name}")

    (OUT / "sections.json").write_text(
        json.dumps(sections.most_common(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()