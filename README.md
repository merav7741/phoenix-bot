# Phoenix RAG Bot

**Live demo:** https://phoenix-bot-2meaqsw7hcfzcfdhzjwwnf.streamlit.app/
(The submitted version is the Streamlit app, `app.py`. An Angular + FastAPI version also exists in the repo.)

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
| Vector index (Voyage + Chroma) | Done |
| Retrieval with conversation context and typo recovery | Done |
| Answer layer (Gemini) with citations, grounding check and abstention | Done |
| Streamlit UI (deployed on Streamlit Community Cloud) | Done |
| Pre-claim guidance mode (original feature) | Done |
| Evaluation set (30 questions: answerable / personal / unanswerable) | Written |

## Architecture
```
sitemap -> filter -> crawl -> clean -> chunk -> embed -> Chroma
                                                          |
question + recent conversation -> contextualize or clarify -> retrieve -> threshold -> LLM (chunks only) -> verify cited claims -> answer + source URL
```

## Project structure
```
app.py            Streamlit app (the deployed, submitted version)
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
  test_app.py             Streamlit UI tests
  test_deployment.py      deployment and index packaging tests
scripts/          small helper scripts
Dockerfile        image for the Angular + FastAPI version
render.yaml       Render Blueprint for the Angular + FastAPI version
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
Streamlit app (the submitted version):
```
streamlit run app.py
```

Optional Angular + FastAPI version: start the API:
```
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
from the same origin as the API.

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

The submitted cloud app runs the Streamlit entry point `app.py`. The active
Chroma index is stored in `data/chroma_voyage` and included in Git so Community
Cloud receives it with the repository. It contains text collected from public
Phoenix website pages. On first use, the app copies the index to a temporary
directory before opening it, keeping Chroma's runtime writes away from the
checked-in index. Other generated files under `data/` remain ignored.

In the Community Cloud app settings, add `GEMINI_API_KEY` and `VOYAGE_API_KEY`
as Streamlit secrets. Do not commit API keys or `.streamlit/secrets.toml`.
For service continuity, you may also configure up to eight additional Gemini
keys as `GEMINI_API_KEY1` through `GEMINI_API_KEY8` (or with an underscore
before the number). The app skips unset or duplicate keys and tries configured
keys in order. Any Gemini API error (exhausted quota, invalid key, a model the
key cannot use, or an overloaded model) moves on to the next key. If every key
fails with `gemini-2.5-flash`, all keys are tried again with the fallback model
`gemini-3.8-flash`; only when everything fails is an error shown. The key list is cached for the
lifetime of the app process, so restart or redeploy the service after changing
environment variables.

## Original feature: pre-claim guidance

The floating chat has two modes: **Questions** uses the RAG answer flow, while
**Guidance** starts a conversational pre-claim flow: it asks for an insurance
type, then accepts an incident description or a follow-up question. It searches
the selected official Phoenix claims section in Chroma using local lexical
matching. Gemini then writes short practical steps using only those official
excerpts and must cite the excerpts it relies on; answers that cite nothing
supplied are rejected. If no relevant indexed content is found, the app says so
and links to the claims center. The guide provides general public information
only and does not determine claim eligibility or request personal details.

Try it: open the bot → **ליווי** → choose an insurance type → describe what
happened (e.g. "הייתה נזילה בדירה, מה כדאי לבדוק קודם?").

## Deploying the Angular app to Render (optional)

The Angular + FastAPI app is configured as a Render Docker Blueprint in
`render.yaml`. Docker compiles `frontend/` and starts the API in
`server/api.py`; `/api/health` is the service health check. The Chroma index in
`data/chroma_voyage` is included in the Docker image, so deployment does not
require a separate object-storage account or persistent disk. Configure the
Gemini and Voyage API keys in Render's environment settings; never put them in
Angular or commit them to Git.

## How content was collected
- URLs come from the site's `sitemap.xml` (1,643 URLs). We selected the
  customer-facing sections (claims, customer service, apartment, car, health,
  travel, life, pets, mortgage, business, pension, savings, loans).
- Rules in `robots.txt` are respected, with a one-second delay between requests.
- Pages are cleaned to text with headings, and duplicate content is dropped.
- Result: 709 pages saved, split into 4,216 chunks. Every chunk keeps its
  source URL, page title, section heading and crawl date.

## How chunks are chosen
Pages are split by headings. Every chunk starts with the page title and its
heading, so retrieval keeps the context. Very short chunks are not indexed.
When retrieval similarity is below the threshold, the bot abstains instead of
answering.

## Limitations
- Some pages could not be fetched: about 47 failed (network filtering and
  pages listed in the sitemap that no longer exist).
- A network filter blocked 11 pages during the crawl with HTTP 418 responses.
- Not crawled: `investor-relations`, the English site, and tag pages.
- Some chunks may still contain teaser text from related articles.
- Free API tiers have rate limits, so indexing is slow and resumable.

## Refreshing the content
Every chunk stores its crawl date. To refresh, back up any pages added by hand,
delete `data/pages`, `data/chunks.jsonl` and `data/chroma_voyage`, then rerun
`crawl.py`, `chunk.py` and `index.py`. Deleting the index is required because
`index.py` skips chunk IDs it already has, so it would keep outdated text.