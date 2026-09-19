"""Tests for prompt rendering against the templates printed in the paper.

Luo et al. (ACL 2026) Appendix D.6 prints the MRAG-Bench answer-generation
prompt verbatim. The released template differs from it in three ways: it drops
the "Instruction:" prefix, renders choices as "A: x" instead of "(A) x", and
omits the trailing "Answer:" cue.

The inline "{Image_placeholder}" of the printed template is not reproduced:
images are passed to the processor as separate content items, which is the
handling the paper describes for models that reject inline placeholders.
"""

from __future__ import annotations

from utility_mrag.scoring.prompt_templates import format_generation_prompt

CHOICES = {"A": "Lhasa Apso", "B": "Maltese", "C": "Havanese", "D": "Shih-Tzu"}


def test_mrag_bench_generation_prompt_matches_the_paper():
    prompt = format_generation_prompt(
        dataset="mrag_bench", question="Which animal is this?", choices=CHOICES
    )

    assert prompt == (
        "Instruction: You will be given one question concerning several images. "
        "The first image is the input image; the remaining images are retrieved "
        "examples to help you. Answer with the option's letter from the given "
        "choices directly.\n"
        "\n"
        "Question: Which animal is this?\n"
        "Choices:\n"
        "(A) Lhasa Apso\n"
        "(B) Maltese\n"
        "(C) Havanese\n"
        "(D) Shih-Tzu\n"
        "Answer:"
    )


def test_choices_given_as_a_list_render_with_the_same_letters():
    prompt = format_generation_prompt(
        dataset="mrag_bench",
        question="Which animal is this?",
        choices=["Lhasa Apso", "Maltese", "Havanese", "Shih-Tzu"],
    )

    assert "(A) Lhasa Apso\n(B) Maltese\n(C) Havanese\n(D) Shih-Tzu\nAnswer:" in prompt
