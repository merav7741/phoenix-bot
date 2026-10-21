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
- **Conversation context:** recent turns help resolve follow-up questions; when the
  reference is ambiguous, the bot asks a clarifying question before searching.
  The full transcript is kept only in the active Streamlit session.
- **Typos and tone:** colloquial Hebrew and clear typing/spelling mistakes are
  normalized for search only when retrieval is weak; answers should be clear,
  considerate, and grounded in retrieved sources without guessing or inventing
  contact details.

## Status
| Component | State |
|---|---|
| Crawl, clean, dedupe | Done |
| Chunking by headings | Done |
| Vector index (Voyage + Chroma) | In progress |
| Retrieval (Voyage + Chroma) | Implemented; recent context and low-confidence typo recovery; threshold needs calibration |
| Answer layer (Gemini) with retrieved citations, grounding verification, and abstention | Implemented; pending evaluation |
| Streamlit UI | Phoenix-inspired RTL landing-page demo with a floating chat, linked sources, background answer generation, and guarded message submission |
| Evaluation set (30 questions) | Written, not yet run |

## Architecture
```
sitemap -> filter -> crawl -> clean -> chunk -> embed -> Chroma
                                                          |
question + recent conversation -> contextualize or clarify -> retrieve -> threshold -> LLM (chunks only) -> verify cited claims -> answer + source URL
```

## Project structure
```
app.py            Streamlit landing-page demo and floating chat UI (RTL)
rag.py            retrieve chunks and generate grounded answers
.streamlit/
  config.toml     Phoenix-inspired light theme
  blank.svg       Transparent browser-tab icon
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
data/
  chroma_voyage/  deployment index (included in Git; all other generated data is ignored)
```

## Setup
1. Python 3.12. Create and activate a virtual environment:
```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
```
2. Install dependencies:
```
   pip install -r requirements.txt
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

## Deploying to Streamlit Community Cloud

The app entry point is `app.py`. Generated data and the Chroma index under
`data/` are excluded from Git. Do not publish the index in this public
repository: it contains text collected from the Phoenix website. A private
index-delivery method must be configured before deploying the app.

In the Community Cloud app settings, add `GEMINI_API_KEY` and `VOYAGE_API_KEY`
as Streamlit secrets. Do not commit API keys or `.streamlit/secrets.toml`.

## Deploying to Render

The app is configured as a Render Docker Blueprint in `render.yaml`. The local
Chroma index is excluded from Git and the Docker build context.
For deployment, store a ZIP of `data/chroma_voyage` in a **private** Cloudflare
R2 bucket; the service downloads it on startup and checks its SHA-256 hash.

1. Build the archive locally after the index is ready:
   `python scripts/package_index.py`
2. Upload `data/phoenix-index.zip` to a private R2 bucket.
3. In Render, create a Blueprint from this repository and configure the
   variables listed in `render.yaml`. Set `R2_INDEX_SHA256` to the hash printed
   by the packaging script. Keep API keys and R2 credentials in Render's
   environment settings, never in Git.

Render's free service may sleep when idle. After a restart it may need to
download the index again; no scheduled pings are configured to keep it awake.
Chat history is held in the Streamlit session and is not stored in Redis or a
database.

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
Retrieval and the grounded answer layer are implemented. The abstention
threshold and answer quality still need calibration against the evaluation set.

## Limitations
- Some pages could not be fetched: about 47 failed (network filtering and
  pages listed in the sitemap that no longer exist).
- NetFree blocked 11 pages during the crawl with HTTP 418 responses. Ordinary
  HTTP/HTTPS requests are not guaranteed to pass through network filters.
- Not crawled: `investor-relations`, the English site, and tag pages.
- Some chunks may still contain teaser text from related articles.
- Free API tiers have rate limits, so indexing is slow and resumable.
- Evaluation results will be added after the first run.

## Refreshing the content
Every chunk stores its crawl date. To refresh, delete `data/pages` and
`data/chunks.jsonl`, then rerun `crawl.py`, `chunk.py` and `index.py`.