"""Tests for LMDB image export (requires the optional synthetic-faces deps)."""

import io
from pathlib import Path

import pytest
from PIL import Image

from llm_spark_exp.synthetic_faces.generation.vec2face_plus import export_lmdb_images

lmdb = pytest.importorskip("lmdb")
msgpack = pytest.importorskip("msgpack")


def _jpeg_bytes(color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), color=color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _make_lmdb(path: Path, records: dict[bytes, bytes]) -> None:
    env = lmdb.open(str(path), map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        for key, value in records.items():
            txn.put(key, value)
    env.close()


def test_export_dict_records_groups_by_label(tmp_path: Path) -> None:
    lmdb_path = tmp_path / "data.lmdb"
    _make_lmdb(
        lmdb_path,
        {
            b"0": msgpack.packb({"image": _jpeg_bytes((200, 0, 0)), "label": 0}),
            b"1": msgpack.packb({"image": _jpeg_bytes((0, 200, 0)), "label": 0}),
            b"2": msgpack.packb({"image": _jpeg_bytes((0, 0, 200)), "label": 1}),
        },
    )
    output_dir = tmp_path / "out"

    result = export_lmdb_images(lmdb_path=lmdb_path, output_dir=output_dir)

    assert result.images == 3
    assert sorted(p.name for p in output_dir.iterdir()) == ["000000", "000001"]
    assert sorted(p.name for p in (output_dir / "000000").iterdir()) == ["000.jpg", "001.jpg"]
    assert [p.name for p in (output_dir / "000001").iterdir()] == ["000.jpg"]


def test_export_supports_sequence_records(tmp_path: Path) -> None:
    lmdb_path = tmp_path / "seq.lmdb"
    _make_lmdb(
        lmdb_path,
        {b"0": msgpack.packb([_jpeg_bytes((1, 2, 3)), None, 5])},
    )
    output_dir = tmp_path / "out"

    result = export_lmdb_images(lmdb_path=lmdb_path, output_dir=output_dir)

    assert result.images == 1
    assert (output_dir / "000005" / "000.jpg").exists()


def test_export_respects_limit(tmp_path: Path) -> None:
    lmdb_path = tmp_path / "data.lmdb"
    _make_lmdb(
        lmdb_path,
        {
            bytes(str(i), "ascii"): msgpack.packb({"image": _jpeg_bytes((i, i, i)), "label": i})
            for i in range(5)
        },
    )
    output_dir = tmp_path / "out"

    result = export_lmdb_images(lmdb_path=lmdb_path, output_dir=output_dir, limit=2)

    assert result.images == 2


def test_export_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="LMDB path does not exist"):
        export_lmdb_images(lmdb_path=tmp_path / "missing.lmdb", output_dir=tmp_path / "out")


def test_export_rejects_non_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="limit must be at least 1"):
        export_lmdb_images(lmdb_path=tmp_path / "x.lmdb", output_dir=tmp_path / "out", limit=0)
