import json
import re
import time
import hashlib
from datetime import date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# URLS_FILE = Path("data/test_urls.txt")
URLS_FILE = Path("data/selected_urls.txt")
OUT_DIR = Path("data/pages")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; HackathonResearchBot/1.0)",
    "Accept-Language": "he,en;q=0.8",
}
DELAY = 1.0
MIN_TEXT_LEN = 300

SKIP_PATH_PARTS = ("tzmekifa", "finncial-reports", "gemel-kitzva",
                   "mekifa-deals-with-related-parties", "financial-report")

# Everything after this heading is teasers of other articles
RELATED_MARKER = "### עוד בנושא"
# UI lines and the legal disclaimer, not real content
JUNK_LINE_PREFIXES = ("לרכישת/ לקבלת הצעת", "סוכן הביטוח שלך", "סרטון",
                      "מידע זה הינו כללי")


def page_id(url):
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def text_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def clean(html):
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else ""

    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "form", "iframe", "svg"]):
        tag.decompose()

    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    lines = []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if el.name in ("h1", "h2", "h3", "h4"):
            level = int(el.name[1])
            lines.append("\n" + "#" * level + " " + text)
        elif el.name == "li":
            lines.append("- " + text)
        else:
            lines.append(text)

    body = "\n".join(lines)
    if RELATED_MARKER in body:
        body = body.split(RELATED_MARKER)[0]
    body = "\n".join(
        ln for ln in body.split("\n")
        if not ln.strip().startswith(JUNK_LINE_PREFIXES)
    )
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return title, body


def load_seen_hashes():
    seen = set()
    for f in OUT_DIR.glob("*.json"):
        try:
            seen.add(text_hash(json.loads(f.read_text(encoding="utf-8"))["text"]))
        except Exception:
            pass
    return seen


def main():
    urls = URLS_FILE.read_text(encoding="utf-8").splitlines()
    urls = [u for u in urls if not any(p in u for p in SKIP_PATH_PARTS)]
    print(f"Crawling {len(urls)} URLs")

    seen = load_seen_hashes()
    failed_urls, duplicates = [], []
    ok = skipped = 0

    for i, url in enumerate(urls, 1):
        out = OUT_DIR / f"{page_id(url)}.json"
        if out.exists():
            skipped += 1
            continue
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            r.encoding = "utf-8"
            title, body = clean(r.text)
            if len(body) < MIN_TEXT_LEN:
                print(f"[{i}] too short, skipped: {url}")
                skipped += 1
            elif text_hash(body) in seen:
                print(f"[{i}] duplicate content, skipped: {url}")
                duplicates.append(url)
            else:
                seen.add(text_hash(body))
                out.write_text(json.dumps({
                    "url": url,
                    "title": title,
                    "text": body,
                    "crawled_at": date.today().isoformat(),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                ok += 1
                print(f"[{i}/{len(urls)}] ok ({len(body)} chars)")
        except Exception as e:
            failed_urls.append(f"{url}\t{e}")
            print(f"[{i}] FAILED {url}: {e}")
        time.sleep(DELAY)

    Path("data/failed.txt").write_text("\n".join(failed_urls), encoding="utf-8")
    Path("data/duplicates.txt").write_text("\n".join(duplicates), encoding="utf-8")
    print(f"\nDone. saved={ok} skipped={skipped} "
          f"duplicates={len(duplicates)} failed={len(failed_urls)}")


if __name__ == "__main__":
    main()