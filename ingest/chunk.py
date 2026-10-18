import json
import re
from pathlib import Path

PAGES_DIR = Path("data/pages")
OUT_FILE = Path("data/chunks.jsonl")

MIN_CHARS = 200    # smaller sections get merged into the previous one
MAX_CHARS = 1800   # larger sections get split on paragraph boundaries

# Boilerplate headings that carry no answerable content
JUNK_HEADINGS = {
    "כל הדרכים לדבר איתנו", "בוואטסאפ", "בצ'אט", "בטלפון", "שירות לקוחות",
    "לבצע שינויים", "למצוא טופס או מסמך", "כדאי לדעת", "מסמכים חשובים",
}

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def clean_title(title):
    return re.sub(r"\s*\|\s*הפניקס\s*$", "", title).strip()


def dedupe_lines(lines):
    """Removes consecutive duplicate lines (list items appear twice)."""
    out = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if out and (s == out[-1] or s.lstrip("- ").strip() == out[-1].lstrip("- ").strip()):
            continue
        out.append(s)
    return out


def split_sections(text):
    """Returns a list of (heading, body_lines)."""
    sections = []
    heading, body = "", []
    for ln in text.split("\n"):
        m = HEADING_RE.match(ln.strip())
        if m:
            if heading or body:
                sections.append((heading, body))
            heading, body = m.group(2).strip(), []
        else:
            body.append(ln)
    if heading or body:
        sections.append((heading, body))
    return sections


def split_long(body, max_chars):
    """Splits a long body into pieces on line boundaries."""
    pieces, cur = [], ""
    for ln in body:
        if cur and len(cur) + len(ln) + 1 > max_chars:
            pieces.append(cur)
            cur = ""
        cur += ("\n" if cur else "") + ln
    if cur:
        pieces.append(cur)
    return pieces


def chunk_page(page):
    title = clean_title(page["title"])
    sections = split_sections(page["text"])

    prepared = []
    for heading, body in sections:
        if heading in JUNK_HEADINGS:
            continue
        body = dedupe_lines(body)
        text = "\n".join(body).strip()
        if not text:
            continue
        prepared.append([heading, text])

    # Merge tiny sections into the previous one
    merged = []
    for heading, text in prepared:
        if merged and len(text) < MIN_CHARS:
            label = f"{heading}: " if heading else ""
            merged[-1][1] += "\n" + label + text
        else:
            merged.append([heading, text])

    chunks = []
    for heading, text in merged:
        for part in split_long(text.split("\n"), MAX_CHARS):
            header = title if not heading or heading == title else f"{title} > {heading}"
            chunks.append({"header": header, "body": part})
    return chunks


def main():
    files = sorted(PAGES_DIR.glob("*.json"))
    total = 0
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for f in files:
            page = json.loads(f.read_text(encoding="utf-8"))
            for i, c in enumerate(chunk_page(page)):
                rec = {
                    "id": f"{f.stem}-{i}",
                    "url": page["url"],
                    "page_title": clean_title(page["title"]),
                    "header": c["header"],
                    "text": c["header"] + "\n" + c["body"],
                    "crawled_at": page["crawled_at"],
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
    print(f"Pages: {len(files)} | Chunks: {total}")


if __name__ == "__main__":
    main()