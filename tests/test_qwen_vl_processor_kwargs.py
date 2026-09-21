"""Tests for handing the Qwen3-VL processor a pixel budget.

Luo et al. never state a resolution cap, and the released wrapper builds the
processor with no keyword arguments at all, so every image reaches the vision
tower at whatever the checkpoint's default budget allows -- 16,777,216 pixels.

The first attempt at this passed ``max_pixels``. That was wrong, and wrong in
the worst way: ``Qwen2VLImageProcessorFast`` accepts ``max_pixels``, stores it
as an attribute, and never reads it. Resizing is driven by
``size["longest_edge"]``, which stayed at the default. The cap therefore did
nothing, silently -- no error, no warning. It was caught by a sweep in which a
4 MP cap changed neither the runtime (663.0s vs 657.7s) nor a single one of 20
Top-3 selections, and confirmed by inspecting the processor directly.

``size`` must carry both keys: passing ``longest_edge`` alone raises
``ValueError: size must contain 'shortest_edge' and 'longest_edge' keys.``
``shortest_edge`` is therefore pinned to the checkpoint's own default so that
capping the ceiling does not silently move the floor.

Only ``processor_kwargs_from_config`` is exercised here, not ``load()``.
``load()`` does ``from transformers import AutoProcessor`` inside the function
body, and ``transformers`` is a ``_LazyModule``: patching the attribute is
visible to ``getattr`` but *not* to that ``from``-import, so a unit test of the
call site would silently assert against a stand-in that never runs while the
real weights load from disk. No test in this repo loads a model.

That gap is why the first version shipped broken, so the call site is now
verified end to end instead: score the same questions capped and uncapped and
compare both the runtime and the Top-3 sets.
"""

from __future__ import annotations

from utility_mrag.models.base import ModelConfig
from utility_mrag.models.qwen_vl import (
    SHORTEST_EDGE_DEFAULT,
    processor_kwargs_from_config,
)

ONE_MEGAPIXEL = 1_048_576


def _config(**extra) -> ModelConfig:
    return ModelConfig(
        family="qwen3_vl",
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        role="surrogate",
        dtype="bfloat16",
        extra=extra,
    )


def test_unset_budget_keeps_the_released_behaviour():
    assert processor_kwargs_from_config(_config()) == {}


def test_configured_budget_sets_the_key_the_processor_actually_reads():
    assert processor_kwargs_from_config(_config(max_pixels=ONE_MEGAPIXEL)) == {
        "size": {
            "longest_edge": ONE_MEGAPIXEL,
            "shortest_edge": SHORTEST_EDGE_DEFAULT,
        }
    }


def test_budget_never_emits_max_pixels_which_the_processor_ignores():
    kwargs = processor_kwargs_from_config(_config(max_pixels=ONE_MEGAPIXEL))

    assert "max_pixels" not in kwargs


def test_size_always_carries_both_keys_or_the_processor_raises():
    size = processor_kwargs_from_config(_config(max_pixels=ONE_MEGAPIXEL))["size"]

    assert set(size) == {"longest_edge", "shortest_edge"}


def test_budget_written_as_a_yaml_string_becomes_an_int():
    kwargs = processor_kwargs_from_config(_config(max_pixels=str(ONE_MEGAPIXEL)))

    assert kwargs["size"]["longest_edge"] == ONE_MEGAPIXEL
    assert isinstance(kwargs["size"]["longest_edge"], int)


def test_explicit_null_budget_counts_as_unset():
    assert processor_kwargs_from_config(_config(max_pixels=None)) == {}
