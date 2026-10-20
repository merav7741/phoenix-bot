import argparse
import hashlib
import zipfile
from pathlib import Path


INDEX_DIR = Path("data/chroma_voyage")


def package_index(output_path):
    database_file = INDEX_DIR / "chroma.sqlite3"
    if not database_file.is_file() or database_file.stat().st_size == 0:
        raise FileNotFoundError(
            f"Chroma index not found at {database_file}. Build the index first."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(INDEX_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, Path("chroma_voyage") / path.relative_to(INDEX_DIR))

    digest = hashlib.sha256()
    with output_path.open("rb") as archive_file:
        for block in iter(lambda: archive_file.read(1024 * 1024), b""):
            digest.update(block)

    print(f"Created {output_path} ({output_path.stat().st_size:,} bytes)")
    print(f"SHA-256: {digest.hexdigest()}")


def main():
    parser = argparse.ArgumentParser(
        description="Package the local Chroma index for private R2 storage."
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="data/phoenix-index.zip",
        help="ZIP output path (default: data/phoenix-index.zip)",
    )
    args = parser.parse_args()
    package_index(args.output)


if __name__ == "__main__":
    main()
