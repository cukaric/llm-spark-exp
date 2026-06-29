"""CPU-only unit tests for worker logic that needs no GPU or external repos."""

import pytest
from PIL import Image

try:
    import torch

    from llm_spark_exp.synthetic_faces.generation.center_worker import (
        parse_float_tuple,
        sample_nearby_vectors,
        save_images,
    )
    from llm_spark_exp.synthetic_faces.restoration.flux_refine_worker import (
        iter_images,
        torch_dtype,
    )
    from llm_spark_exp.synthetic_faces.restoration.restore_worker import _make_vqvae_args
except ImportError:  # optional synthetic-faces deps (torch/imageio) not installed
    pytest.skip("synthetic-faces optional deps not installed", allow_module_level=True)


# --------------------------------------------------------------------------- #
# center_worker
# --------------------------------------------------------------------------- #


def test_sample_nearby_vectors_preserves_shape_and_row_norm() -> None:
    base = torch.randn(10, 512)
    out = sample_nearby_vectors(base, epsilons=(0.05, 0.1), percentages=(0.5, 0.5))
    assert out.shape == base.shape
    # each row is renormalized back to its original norm
    assert torch.allclose(out.norm(dim=1), base.norm(dim=1), atol=1e-4)


def test_sample_nearby_vectors_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        sample_nearby_vectors(torch.randn(4, 8), epsilons=(0.1,), percentages=(0.5, 0.5))


def test_sample_nearby_vectors_rejects_weights_not_summing_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        sample_nearby_vectors(torch.randn(4, 8), epsilons=(0.1, 0.2), percentages=(0.5, 0.6))


def test_save_images_groups_by_identity_with_dense_numbering(tmp_path) -> None:
    import numpy as np

    images = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    save_images(images, [0, 0, 1], root=tmp_path, name="run")

    base = tmp_path / "run"
    assert sorted(p.name for p in base.iterdir()) == ["0000", "0001"]
    assert sorted(p.name for p in (base / "0000").iterdir()) == ["000.jpg", "001.jpg"]
    assert [p.name for p in (base / "0001").iterdir()] == ["000.jpg"]


def test_parse_float_tuple_parses_and_validates() -> None:
    assert parse_float_tuple("0.1, 0.2 ,0.3", name="x") == (0.1, 0.2, 0.3)
    with pytest.raises(ValueError, match="at least one"):
        parse_float_tuple("  ,  ", name="x")
    with pytest.raises(ValueError, match="positive"):
        parse_float_tuple("0.1,-0.2", name="x")


# --------------------------------------------------------------------------- #
# restore_worker
# --------------------------------------------------------------------------- #


def test_make_vqvae_args_defaults() -> None:
    args = _make_vqvae_args("/weights/associate_2.ckpt")
    assert args.img_encoder_weight == "/weights/associate_2.ckpt"
    assert args.cat_prompt_embedding is False
    assert args.use_pos_embedding is False
    assert args.use_att_pool is False
    assert args.learnable_pos_emb is False


# --------------------------------------------------------------------------- #
# flux_refine_worker
# --------------------------------------------------------------------------- #


def _make_identity_tree(root, identities: int, per_identity: int) -> None:
    for i in range(identities):
        identity_dir = root / f"{i:04d}"
        identity_dir.mkdir(parents=True)
        for j in range(per_identity):
            Image.new("RGB", (4, 4)).save(identity_dir / f"{j:03d}.jpg")


def test_iter_images_orders_and_bounds(tmp_path) -> None:
    _make_identity_tree(tmp_path, identities=3, per_identity=3)

    all_items = list(iter_images(tmp_path, identity_limit=None, max_images=None))
    assert len(all_items) == 9
    assert all_items[0][0] == "0000"  # sorted identity order

    limited_identities = {
        ident for ident, _ in iter_images(tmp_path, identity_limit=2, max_images=None)
    }
    assert limited_identities == {"0000", "0001"}

    capped = list(iter_images(tmp_path, identity_limit=None, max_images=4))
    assert len(capped) == 4


def test_torch_dtype_mapping() -> None:
    assert torch_dtype("bfloat16") is torch.bfloat16
    assert torch_dtype("float16") is torch.float16
    assert torch_dtype("float32") is torch.float32
    assert torch_dtype("anything-else") is torch.float32
