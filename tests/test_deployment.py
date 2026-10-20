import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import package_index, prepare_index


class PackageIndexTests(unittest.TestCase):
    def test_packages_index_and_prints_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            index_dir = root / "source"
            index_dir.mkdir()
            (index_dir / "chroma.sqlite3").write_bytes(b"index")
            (index_dir / "segment" / "data.bin").parent.mkdir()
            (index_dir / "segment" / "data.bin").write_bytes(b"vector data")
            output = root / "phoenix-index.zip"

            with patch.object(package_index, "INDEX_DIR", index_dir):
                package_index.package_index(output)

            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    archive.namelist(),
                    ["chroma_voyage/chroma.sqlite3", "chroma_voyage/segment/data.bin"],
                )

    def test_missing_index_fails_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(package_index, "INDEX_DIR", Path(temp) / "missing"):
                with self.assertRaisesRegex(FileNotFoundError, "Build the index first"):
                    package_index.package_index(Path(temp) / "index.zip")


class PrepareIndexTests(unittest.TestCase):
    def test_missing_r2_settings_are_reported(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "R2_ACCOUNT_ID"):
                prepare_index._required_settings()

    def test_invalid_sha256_is_rejected(self):
        settings = {
            name: "configured"
            for name in (
                "R2_ACCOUNT_ID",
                "R2_BUCKET",
                "R2_ACCESS_KEY_ID",
                "R2_SECRET_ACCESS_KEY",
                "R2_OBJECT_KEY",
            )
        }
        settings["R2_INDEX_SHA256"] = "not-a-hash"
        with patch.dict("os.environ", settings, clear=True):
            with self.assertRaisesRegex(RuntimeError, "64-character SHA-256"):
                prepare_index._required_settings()

    def test_extracts_valid_index_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "index.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("chroma_voyage/chroma.sqlite3", b"index")
                archive.writestr("chroma_voyage/segment/data.bin", b"vectors")

            destination = root / "extracted"
            prepare_index._extract_archive(archive_path, destination)

            self.assertEqual(
                (destination / "chroma.sqlite3").read_bytes(),
                b"index",
            )
            self.assertEqual(
                (destination / "segment" / "data.bin").read_bytes(),
                b"vectors",
            )

    def test_rejects_paths_outside_index_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_path = root / "index.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("chroma_voyage/../../outside.txt", b"unsafe")

            with self.assertRaisesRegex(RuntimeError, "Unexpected path"):
                prepare_index._extract_archive(archive_path, root / "extracted")


if __name__ == "__main__":
    unittest.main()
