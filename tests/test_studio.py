"""The decode studio: the per-step trace, its payload, and the page it writes."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from zte.evaluation.interactive.studio import _quantise, _trim, studio_html
from zte.models.decoder import FrozenLM

# --------------------------------------------------------------------------- #
# The trace, and the guarantee that it changes nothing
# --------------------------------------------------------------------------- #


def test_tracing_a_decode_does_not_change_the_decode() -> None:
    """The studio must be unable to move the number it visualises.

    Note:
        The trace costs one extra head application per step to measure the evidence path's effect. If that ever
        reached the emitted tokens, a page built to inspect a decode would be showing a different decode from the
        one the evaluation scored -- and nothing downstream would notice.
    """
    lm = FrozenLM('tiny')
    torch.manual_seed(0)
    prefix = torch.randn(2, 4, lm.hidden_dim)

    plain = lm.generate_from_prefix(prefix, max_new_tokens=12)
    sink: list[list[dict[str, Any]]] = []
    traced = lm.generate_from_prefix(prefix, max_new_tokens=12, trace=sink)

    assert plain == traced
    assert len(sink) == 2


def test_the_trace_records_what_the_decoder_chose_and_what_it_nearly_chose() -> None:
    """A step with only the emitted token is a transcript; the alternatives are what make it a diagnostic."""
    lm = FrozenLM('tiny')
    sink: list[list[dict[str, Any]]] = []
    lm.generate_from_prefix(torch.randn(1, 4, lm.hidden_dim), max_new_tokens=6, trace=sink)

    steps = sink[0]
    assert steps, 'a decode that emitted tokens must have traced them'
    first = steps[0]
    assert {'step', 'piece', 'probability', 'entropy', 'evidence_kl', 'alternatives'} <= set(first)
    assert 0.0 <= first['probability'] <= 1.0
    assert first['entropy'] >= 0.0

    probabilities = [a['probability'] for a in first['alternatives']]
    assert probabilities == sorted(probabilities, reverse=True), 'alternatives must be ranked'
    assert first['probability'] == pytest.approx(probabilities[0])


def test_a_decode_with_no_evidence_path_reports_no_evidence() -> None:
    """`evidence_kl` is the nudge's own effect, so with no nudge it must be exactly zero rather than noise."""
    lm = FrozenLM('tiny')
    sink: list[list[dict[str, Any]]] = []
    lm.generate_from_prefix(torch.randn(1, 4, lm.hidden_dim), max_new_tokens=5, trace=sink)

    assert all(step['evidence_kl'] == 0.0 and step['evidence_norm'] == 0.0 for step in sink[0])


def test_an_evidence_nudge_shows_up_as_a_nonzero_shift() -> None:
    """And with a nudge it must be nonzero, or the panel would read as 'the brain did nothing' for every run."""
    lm = FrozenLM('tiny')
    sink: list[list[dict[str, Any]]] = []
    lm.generate_from_prefix(
        torch.randn(1, 4, lm.hidden_dim),
        max_new_tokens=5,
        evidence=lambda steps: torch.full((1, 1, lm.hidden_dim), 0.4),
        trace=sink,
    )

    assert any(step['evidence_kl'] > 0.0 for step in sink[0])
    assert all(step['evidence_norm'] > 0.0 for step in sink[0])


# --------------------------------------------------------------------------- #
# The payload
# --------------------------------------------------------------------------- #


def test_the_pointer_is_cut_back_to_the_readings_own_words() -> None:
    """It arrives padded to the batch's longest reading, and those columns are always empty."""
    padded = [[0.5, 0.5, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0]]

    assert _trim(padded, 2) == [[0.5, 0.5], [0.1, 0.9]]
    assert _trim(None, 2) is None
    assert _trim(padded, 0) == padded


def test_band_power_is_quantised_relative_to_the_reading_it_came_from() -> None:
    """The colour scale compares electrodes and words inside one reading; across readings it means nothing."""
    block = np.array([[[1.0, 10.0, 100.0]], [[1.0, 1.0, 1.0]]])
    out = _quantise(block)

    flat = [v for word in out for band in word for v in band]
    assert min(flat) == 0
    assert max(flat) == 255
    assert all(isinstance(v, int) for v in flat)


def test_a_flat_reading_quantises_without_dividing_by_zero() -> None:
    """Every electrode equal is a real (if uninformative) reading, not a crash."""
    assert _quantise(np.ones((2, 1, 3))) == [[[0, 0, 0]], [[0, 0, 0]]]
    assert _quantise(np.zeros((0, 1, 3))) == []


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #


def _payload(html: str) -> dict[str, Any]:
    """Pulls the JSON island back out of a written page."""
    match = re.search(r'<script id="data" type="application/json">\s*(\{.*?\})\s*</script>', html, re.S)
    assert match is not None
    return dict(json.loads(match.group(1)))


def test_the_studio_page_is_self_contained(tmp_path: Path) -> None:
    """It has to open from a Drive mirror on a machine with no network, so nothing may be fetched at view time."""
    page = studio_html({'run_name': 'r', 'applicable': True, 'readings': [], 'bands': []}, tmp_path / 'STUDIO.html')
    html = page.read_text(encoding='utf-8')

    assert '<canvas' in html
    remote = re.findall(r'<(?:script|link|img|iframe)\b[^>]*\b(?:src|href)\s*=\s*["\'](https?:|//)', html, re.I)
    assert not remote, f'the page fetches {len(remote)} external resource(s)'


def test_the_page_says_it_is_not_a_result(tmp_path: Path) -> None:
    """A screenshot of one good decode is the most misusable artifact this project can produce."""
    page = studio_html({'run_name': 'r', 'applicable': True, 'readings': [], 'bands': []}, tmp_path / 'STUDIO.html')
    html = page.read_text(encoding='utf-8').lower()

    assert 'inspection tool, not a result' in html
    assert 'permutation null' in html


def test_an_inapplicable_run_writes_a_page_that_says_why(tmp_path: Path) -> None:
    """Silence would look like a broken build; the reason is the artifact."""
    page = studio_html({'run_name': 'r', 'applicable': False, 'reason': 'no decoder'}, tmp_path / 'STUDIO.html')

    assert _payload(page.read_text(encoding='utf-8'))['reason'] == 'no decoder'


def test_a_missing_html_suffix_is_corrected(tmp_path: Path) -> None:
    """The page is opened by double-clicking it, so it has to be named like a page."""
    assert studio_html({'run_name': 'r', 'applicable': False}, tmp_path / 'studio').suffix == '.html'
