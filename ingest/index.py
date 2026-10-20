import json
import os
import time
from pathlib import Path

import chromadb
import voyageai
from dotenv import load_dotenv

load_dotenv()
client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

CHUNKS_FILE = Path("data/chunks.jsonl")
DB_DIR = "data/chroma_voyage"
MODEL = "voyage-4"  # model names change, see the note below
BATCH = 6
MIN_CHARS = 150
MAX_CHARS = 2500

def embed(texts, retries=6):
    for attempt in range(retries):
        try:
            res = client.embed(texts, model=MODEL, input_type="document")
            return res.embeddings
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"  embed error ({e}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("embedding failed repeatedly")


def main():
    lines = CHUNKS_FILE.read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(x) for x in lines]
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
        time.sleep(20)

    print("Done. Total in index:", col.count())


if __name__ == "__main__":
    main()