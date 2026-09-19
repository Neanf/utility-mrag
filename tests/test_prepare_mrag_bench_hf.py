"""Tests for MRAG-Bench candidate-pool construction.

Luo et al. (ACL 2026) §5.1 build each pool as the union of the annotated
ground-truth images and the retrieved images, with duplicate images removed.
Counting the released dataset shows the difference is material: keeping every
copy yields 13,122 candidates where the paper reports 11,523 (Table 6), because
1,508 ground-truth images are also returned by the retriever.
"""

from __future__ import annotations

import io

from PIL import Image

from scripts.prepare_mrag_bench_hf import build_examples


def _png(color: tuple[int, int, int]) -> dict:
    """An image struct shaped like the HuggingFace release."""
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


def _row(gt_images: list[dict], retrieved_images: list[dict]) -> dict:
    return {
        "id": "q1",
        "question": "Which animal is this?",
        "answer_choice": "A",
        "A": "a",
        "B": "b",
        "C": "c",
        "D": "d",
        "scenario": "Angle",
        "aspect": None,
        "image": _png((0, 0, 0)),
        "gt_images": gt_images,
        "retrieved_images": retrieved_images,
    }


def _build(row: dict, image_dir, **kwargs):
    kwargs.setdefault("num_candidates", None)
    kwargs.setdefault("include_gt", True)
    (example,) = list(build_examples([row], image_dir, **kwargs))
    return example


def test_retrieved_copy_of_a_ground_truth_image_is_dropped(tmp_path):
    gt = _png((255, 0, 0))
    example = _build(_row([gt], [dict(gt), _png((0, 255, 0))]), tmp_path)

    assert len(example.candidate_images) == 2


def test_surviving_copy_keeps_its_ground_truth_provenance(tmp_path):
    gt = _png((255, 0, 0))
    example = _build(_row([gt], [dict(gt)]), tmp_path)

    assert [c.source for c in example.candidate_images] == ["gt"]
    assert example.gt_image_ids == [example.candidate_images[0].image_id]


def test_distinct_images_are_all_kept(tmp_path):
    example = _build(_row([_png((255, 0, 0))], [_png((0, 255, 0)), _png((0, 0, 255))]), tmp_path)

    assert len(example.candidate_images) == 3


def test_dedup_can_be_switched_off_to_rebuild_the_released_pool(tmp_path):
    gt = _png((255, 0, 0))
    example = _build(_row([gt], [dict(gt)]), tmp_path, dedup=False)

    assert len(example.candidate_images) == 2
