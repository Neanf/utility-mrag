"""Tests for handing the Qwen3-VL processor a pixel budget.

Luo et al. never state a resolution cap, and the released wrapper builds the
processor with no keyword arguments at all, so every image reaches the vision
tower at its native size. That is not survivable on the released MRAG-Bench
pool: its largest image is 50.3 MP, i.e. 196,566 patches, whose attention
matrix needs ~1.2 TB, and even the p99 image (16.7 MP) exceeds an 80 GB card.
A cap is therefore a precondition for running at all, not a speed knob.

Two behaviours matter: with nothing configured the wrapper must keep the
released behaviour untouched, and a configured budget must become a processor
keyword argument. Recorded as F10 in our baseline-fidelity log.

Only ``processor_kwargs_from_config`` is exercised here, not ``load()``.
``load()`` does ``from transformers import AutoProcessor`` inside the function
body, and ``transformers`` is a ``_LazyModule``: patching the attribute is
visible to ``getattr`` but *not* to that ``from``-import, so a unit test of the
call site would silently assert against a stand-in that never runs while the
real 8B weights load from disk. No test in this repo loads a model, and this
one does not either. The call site is instead verified end to end by probe P4,
which scores the same questions at three different caps -- if the budget never
reached the processor, the three Top-3 sets would come back identical.
"""

from __future__ import annotations

from utility_mrag.models.base import ModelConfig
from utility_mrag.models.qwen_vl import processor_kwargs_from_config

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


def test_configured_budget_becomes_a_processor_kwarg():
    assert processor_kwargs_from_config(_config(max_pixels=ONE_MEGAPIXEL)) == {
        "max_pixels": ONE_MEGAPIXEL
    }


def test_budget_written_as_a_yaml_string_becomes_an_int():
    kwargs = processor_kwargs_from_config(_config(max_pixels=str(ONE_MEGAPIXEL)))

    assert kwargs == {"max_pixels": ONE_MEGAPIXEL}
    assert isinstance(kwargs["max_pixels"], int)


def test_explicit_null_budget_counts_as_unset():
    assert processor_kwargs_from_config(_config(max_pixels=None)) == {}
