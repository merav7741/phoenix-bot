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
| Streamlit UI | Phoenix-inspired RTL landing page and floating chat with separate question and pre-claim guidance modes |
| Evaluation set (30 questions) | Written, not yet run |

## Architecture
```
sitemap -> filter -> crawl -> clean -> chunk -> embed -> Chroma
                                                          |
question + recent conversation -> contextualize or clarify -> retrieve -> threshold -> LLM (chunks only) -> verify cited claims -> answer + source URL
```

## Project structure
```
app.py            legacy Streamlit demo (kept available)
server/
  api.py          FastAPI endpoints and production delivery for the Angular app
  guide_catalog.py shared list of supported insurance guidance flows
  rag.py          retrieval, grounded answers, and claim-guide excerpts
frontend/
  src/app/        standalone Angular components for the Hebrew RTL experience
  proxy.conf.json local API proxy for Angular development
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
  test_api.py             FastAPI contract and validation tests
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

## Run locally
Install Python dependencies from the project root and start the API:
```
pip install -r requirements.txt
python -m uvicorn server.api:app --reload --port 8000
```

In another terminal, run the Angular development server:
```
cd frontend
npm ci
npm start
```

The Angular development server proxies `/api` requests to FastAPI. Use npm 11
for dependency installation (`npx npm@11 ci`) to match the pinned Docker build.
In production, Angular is compiled into the Docker image and served by FastAPI
from the same origin as the API. The legacy Streamlit demo can still be
started separately with `streamlit run app.py`.

To rebuild the public source index when needed:
```
python ingest/map_site.py
python ingest/filter_urls.py
python ingest/crawl.py
python ingest/chunk.py
python ingest/index.py
```

### Add pages supplied separately

Place each verified Phoenix page JSON in `data/pages/` with the fields
`url`, `title`, `text`, and `crawled_at`. Keep the article body and its policy
disclaimers; remove navigation cards, unrelated recommended articles, and
editorial notes copied into the text. Then rebuild and index:

```powershell
python ingest/chunk.py
python ingest/index.py
```

`data/pages/` and `data/chunks.jsonl` are ignored by Git. The resulting
`data/chroma_voyage/` index is tracked and must be included in the commit for
the new pages to reach a deployment. Indexing requires `VOYAGE_API_KEY`; the
script skips chunk IDs already present, so it can resume after an interruption.
The current local corpus contains 709 page files and generates 4,216 chunks;
chunks shorter than the indexer's 150-character minimum are skipped.

## Deploying to Streamlit Community Cloud

The Community Cloud app continues to use the legacy Streamlit entry point
`app.py`; it is separate from the Angular + FastAPI frontend. The active
Chroma index is stored in
`data/chroma_voyage` and included in Git so Community Cloud receives it with
the repository. It contains text collected from public Phoenix website pages.
On first use, the app copies the index to a temporary directory before opening
it, keeping Chroma's runtime writes away from the checked-in index. Other
generated files under `data/` remain ignored.

In the Community Cloud app settings, add `GEMINI_API_KEY` and `VOYAGE_API_KEY`
as Streamlit secrets. Do not commit API keys or `.streamlit/secrets.toml`.
For service continuity, you may also configure up to eight additional Gemini
keys as `GEMINI_API_KEY1` through `GEMINI_API_KEY8` (or with an underscore
before the number). The app skips unset or duplicate keys and tries configured
keys in order. It advances to the next key when Gemini reports an invalid key
(`API_KEY_INVALID` or 401) or a quota/rate-limit response (429); other API
errors are returned without trying another key. This is failover between keys
you are authorized to use, not a way to evade Google AI Studio limits. The key
list is cached for the lifetime of the app process, so restart or redeploy the
service after changing environment variables.

The floating chat has two modes: **Questions** uses the RAG answer flow, while
**Guidance** starts a conversational pre-claim flow: it asks for an insurance
type, then accepts an incident description or a follow-up question. It searches
the selected official Phoenix claims section in Chroma using local lexical
matching and returns source excerpts with links; this flow does not call Gemini
or Voyage. If no relevant indexed content is found, the app says so and links
to the claims center rather than inventing instructions. The guide provides
general public information only and does not determine claim eligibility or
request personal details.

## Deploying the Angular app to Render

The Angular + FastAPI app is configured as a Render Docker Blueprint in
`render.yaml`. Docker compiles `frontend/` and starts the API in
`server/api.py`; `/api/health` is the service health check. The Chroma index in
`data/chroma_voyage` is included in the Docker image, so deployment does not
require a separate object-storage account or persistent disk.

Create a Blueprint from this repository and configure the Gemini and Voyage
API keys in Render's environment settings if they are available. Never put
these keys in Angular or commit them to Git. No R2 credentials are needed. The
free Render service may sleep when idle, and its local files are temporary;
each new image contains the checked-in index. Chat history is kept in browser
session memory and is not stored in a database.

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
