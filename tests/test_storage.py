from pathlib import Path

import pytest

from storage import delete_uploaded_file, resolve_uploaded_file


def test_resolve_uploaded_file_rejects_traversal(tmp_path: Path):
    with pytest.raises(ValueError):
        resolve_uploaded_file(tmp_path, "../outside.pdf")


def test_delete_uploaded_file_removes_file(tmp_path: Path):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.4")
    delete_uploaded_file(tmp_path, "doc.pdf")
    assert not f.exists()
