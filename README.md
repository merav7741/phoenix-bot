# Phoenix RAG Bot

A Hebrew-language RAG chatbot built for the Open Code AI Hackathon 2026
(Phoenix challenge). It answers customer questions strictly from the public
content of the Phoenix website (fnx.co.il), cites the source page in every
answer, and says "this isn't on the site" when the information is missing.

## Principles
- **Single source of truth:** only public pages from fnx.co.il. No outside knowledge.
- **Citations:** every answer links to the page it came from. No source, no answer.
- **No hallucination:** when retrieval is weak, the bot declines and refers the user to customer service.
- **Boundaries:** no personal advice and no access to customer data.

## Status
| Component | State |
|---|---|
| Crawl, clean, dedupe | Done |
| Chunking by headings | Done |
| Vector index (Voyage + Chroma) | In progress |
| Retrieval (Voyage + Chroma) | Implemented; short follow-ups use conversation context; threshold needs calibration |
| Answer layer (Gemini) with retrieved citations, grounding verification, and abstention | Implemented; pending evaluation |
| Streamlit UI | Connected to the RAG pipeline; pending end-to-end validation |
| Evaluation set (30 questions) | Written, not yet run |

## Architecture
```
sitemap -> filter -> crawl -> clean -> chunk -> embed -> Chroma
                                                          |
question -> contextual rewrite when needed -> retrieve -> threshold -> LLM (chunks only) -> verify cited claims -> answer + source URL
```

## Project structure
```
app.py            Streamlit chat UI (RTL)
rag.py            retrieve chunks and generate grounded answers
ingest/
  map_site.py     read robots.txt and sitemap.xml
  filter_urls.py  choose sections, respect robots.txt rules
  crawl.py        download pages, clean HTML, drop duplicates
  chunk.py        split pages into chunks by headings
  index.py        embed chunks and store them in Chroma
eval/
  questions.json          30 questions: answerable / personal / unanswerable
  negative_questions.json off-topic and manipulation questions
tests/
  test_rag.py             offline tests for answer boundaries and citations
scripts/          small helper scripts
data/             generated files (not committed)
```

## Setup
1. Python 3.12. Create and activate a virtual environment:
```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
```
2. Install dependencies:
```
   pip install requests beautifulsoup4 lxml python-dotenv streamlit chromadb voyageai google-genai
```
3. Create a `.env` file (never commit it):
```
   GEMINI_API_KEY=...
   VOYAGE_API_KEY=...
```

## Run
Run everything from the project root:
```
python ingest/map_site.py     # list URLs from the sitemap
python ingest/filter_urls.py  # select relevant sections
python ingest/crawl.py        # download and clean pages
python ingest/chunk.py        # create data/chunks.jsonl
python ingest/index.py        # embed and index (resumable)
streamlit run app.py          # start the chat UI
```

## How content was collected
- URLs come from the site's `sitemap.xml` (1,643 URLs). We selected the
  customer-facing sections (claims, customer service, apartment, car, health,
  travel, life, pets, mortgage, business, pension, savings, loans).
- Rules in `robots.txt` are respected, with a one-second delay between requests.
- Pages are cleaned to text with headings, and duplicate content is dropped.
- Result: 698 pages saved, split into 4,019 chunks. Every chunk keeps its
  source URL, page title, section heading and crawl date.

## How chunks are chosen
Pages are split by headings. Every chunk starts with the page title and its
heading, so retrieval keeps the context. Very short chunks are not indexed.
Retrieval, the abstention threshold and the answer layer are still being built.

## Limitations
- Some pages could not be fetched: about 47 failed (network filtering and
  pages listed in the sitemap that no longer exist).
- Not crawled: `investor-relations`, the English site, and tag pages.
- Some chunks may still contain teaser text from related articles.
- Free API tiers have rate limits, so indexing is slow and resumable.
- Evaluation results will be added after the first run.

## Refreshing the content
Every chunk stores its crawl date. To refresh, delete `data/pages` and
`data/chunks.jsonl`, then rerun `crawl.py`, `chunk.py` and `index.py`.