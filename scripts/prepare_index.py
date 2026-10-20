import hashlib
import hmac
import os
import re
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError


INDEX_DIR = Path("data/chroma_voyage")
CHROMA_DATABASE = INDEX_DIR / "chroma.sqlite3"
CHUNK_SIZE = 1024 * 1024
MAX_ARCHIVE_BYTES = 250 * 1024 * 1024
MAX_EXTRACTED_BYTES = 500 * 1024 * 1024


def _required_settings():
    names = (
        "R2_ACCOUNT_ID",
        "R2_BUCKET",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_OBJECT_KEY",
        "R2_INDEX_SHA256",
    )
    settings = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise RuntimeError(
            "Chroma index is missing and these Render environment variables "
            f"are not configured: {', '.join(missing)}"
        )
    if not re.fullmatch(r"[a-fA-F0-9]{64}", settings["R2_INDEX_SHA256"]):
        raise RuntimeError("R2_INDEX_SHA256 must be a 64-character SHA-256 hash.")
    return settings


def _download_archive(settings, archive_path):
    client = boto3.client(
        "s3",
        endpoint_url=f"https://{settings['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=settings["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=settings["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )
    digest = hashlib.sha256()
    try:
        response = client.get_object(
            Bucket=settings["R2_BUCKET"],
            Key=settings["R2_OBJECT_KEY"],
        )
        if response.get("ContentLength", 0) > MAX_ARCHIVE_BYTES:
            raise RuntimeError(
                f"The Chroma index archive exceeds the "
                f"{MAX_ARCHIVE_BYTES // (1024 * 1024)} MB limit."
            )
        with response["Body"] as body, archive_path.open("wb") as output:
            for block in body.iter_chunks(chunk_size=CHUNK_SIZE):
                if block:
                    digest.update(block)
                    output.write(block)
    except BotoCoreError as exc:
        raise RuntimeError(
            "Could not download the Chroma index from private R2 storage. "
            "Check the bucket, object key, and R2 credentials."
        ) from exc

    if not hmac.compare_digest(
        digest.hexdigest(), settings["R2_INDEX_SHA256"].lower()
    ):
        raise RuntimeError(
            "The downloaded Chroma index SHA-256 does not match "
            "R2_INDEX_SHA256; refusing to install the file."
        )


def _extract_archive(archive_path, destination):
    destination.mkdir(parents=True)
    found_database = False
    extracted_bytes = 0
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"Index archive failed its CRC check: {bad_member}")

            for member in archive.infolist():
                name = member.filename
                path = PurePosixPath(name)
                if (
                    not name
                    or "\\" in name
                    or path.is_absolute()
                    or ".." in path.parts
                    or not path.parts
                    or path.parts[0] != "chroma_voyage"
                ):
                    raise RuntimeError(
                        f"Unexpected path in Chroma index archive: {name!r}"
                    )

                relative_path = Path(*path.parts[1:])
                if not relative_path.parts:
                    continue
                target = destination / relative_path
                file_type = stat.S_IFMT(member.external_attr >> 16)
                if file_type == stat.S_IFLNK:
                    raise RuntimeError(
                        f"Symbolic links are not allowed in the index archive: {name!r}"
                    )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                extracted_bytes += member.file_size
                if extracted_bytes > MAX_EXTRACTED_BYTES:
                    raise RuntimeError(
                        "The extracted Chroma index exceeds the "
                        f"{MAX_EXTRACTED_BYTES // (1024 * 1024)} MB limit."
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=CHUNK_SIZE)
                if relative_path == Path("chroma.sqlite3"):
                    found_database = target.stat().st_size > 0
    except zipfile.BadZipFile as exc:
        raise RuntimeError("The downloaded Chroma index is not a valid ZIP file.") from exc

    if not found_database:
        raise RuntimeError(
            "The ZIP does not contain a non-empty chroma_voyage/chroma.sqlite3."
        )


def prepare_index():
    if CHROMA_DATABASE.is_file() and CHROMA_DATABASE.stat().st_size > 0:
        print(f"Using existing Chroma index at {INDEX_DIR}")
        return

    settings = _required_settings()
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="phoenix-index-", dir=INDEX_DIR.parent) as temp:
        temp_dir = Path(temp)
        archive_path = temp_dir / "phoenix-index.zip"
        extracted_dir = temp_dir / "chroma_voyage"
        _download_archive(settings, archive_path)
        _extract_archive(archive_path, extracted_dir)

        if INDEX_DIR.exists():
            raise RuntimeError(
                f"An incomplete Chroma index already exists at {INDEX_DIR}. "
                "Remove that specific directory and restart the service."
            )
        extracted_dir.replace(INDEX_DIR)
    print(f"Downloaded and verified Chroma index at {INDEX_DIR}")


if __name__ == "__main__":
    prepare_index()
