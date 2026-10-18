# Phoenix RAG Bot

A Hebrew-language RAG chatbot built for the Open Code AI Hackathon 2026
(Phoenix challenge). It answers customer questions strictly from the public
content of the Phoenix website, cites the source page in every answer, and
says "this isn't on the site" when the information is missing.

## Principles
- **Single source of truth:** only public pages from fnx.co.il. No outside knowledge.
- **Citations:** every answer links to the page it came from. No source, no answer.
- **No hallucination:** when retrieval is weak, the bot declines and refers the user to customer service.
- **Boundaries:** no personal advice and no access to customer data.

## Architecture
Crawl -> clean HTML -> chunk by headings -> embeddings -> vector index
-> hybrid retrieval + rerank -> answer with citations -> chat UI

## Status
Work in progress (3-day build).

## Setup
Instructions will be added as the project progresses.