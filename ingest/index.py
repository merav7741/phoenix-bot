import os
import time
from pathlib import Path

import chromadb
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

CHUNKS_FILE = Path("data/chunks.jsonl")
DB_DIR = "data/chroma"
MODEL = "gemini-embedding-001"
BATCH = 20
MIN_CHARS = 150
MAX_CHARS = 6000  # very long chunks are trimmed for embedding only


def embed(texts, retries=6):
    for attempt in range(retries):
        try:
            res = client.models.embed_content(
                model=MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in res.embeddings]
        except Exception as e:
            wait = 5 * (attempt + 1)
            print(f"  embed error ({e}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("embedding failed repeatedly")


def main():
    chunks = [json.loads(x) for x in CHUNKS_FILE.read_text(encoding="utf-8").splitlines()]
    chunks = [c for c in chunks if len(c["text"]) >= MIN_CHARS]
    print(f"Chunks to index: {len(chunks)}")

    db = chromadb.PersistentClient(path=DB_DIR)
    col = db.get_or_create_collection("fnx", metadata={"hnsw:space": "cosine"})

    done = set(col.get(include=[])["ids"])
    todo = [c for c in chunks if c["id"] not in done]
    print(f"Already indexed: {len(done)} | Remaining: {len(todo)}")

    for i in range(0, len(todo), BATCH):
        batch = todo[i : i + BATCH]
        vectors = embed([c["text"][:MAX_CHARS] for c in batch])
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=vectors,
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "url": c["url"],
                    "page_title": c["page_title"],
                    "header": c["header"],
                    "crawled_at": c["crawled_at"],
                }
                for c in batch
            ],
        )
        print(f"  indexed {min(i + BATCH, len(todo))}/{len(todo)}")
        time.sleep(15)

    print("Done. Total in index:", col.count())


if __name__ == "__main__":
    main()