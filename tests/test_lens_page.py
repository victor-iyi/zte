"""Tests for the lens page builder."""

import json
import math
from pathlib import Path
from typing import Any, Final

import pytest

from zte.lens.page import build_lens_page

# Every panel heading the template must carry, so a missing section is a broken page, not a silent gap.
PANEL_TITLES: Final[tuple[str, ...]] = (
    'The reading',
    'The scalp',
    'The neighbourhood',
    'The decode trace',
)

# The exact disclaimer the CLI writes; the page must refuse to render without one.
DISCLAIMER: Final[str] = 'inspection, not a result -- no number here is a headline'


def _encode_lens() -> dict[str, Any]:
    """A minimal encode-mode capture: one held-out reading, a two-channel scalp and three neighbours."""
    return {
        'mode': 'encode',
        'reading': {
            'subject': 'ZAB',
            'task': 'NR',
            'text': 'The quick brown fox jumps.',
            'words': ['The', 'quick', 'brown', 'fox', 'jumps.'],
            'n_words': 5,
            'is_holdout': True,
        },
        'embedding': {'dim': 768, 'norm': 12.41},
        'word_saliency': {'scores': [0.01, 0.21, 0.05, 0.4, -0.02], 'method': 'pad-mask occlusion'},
        'channel_saliency': {
            'labels': ['E1', 'E2'],
            'regions': ['frontal', 'occipital'],
            'xy': [[0.0, 0.9], [0.0, -0.9]],
            'xyz': [[0.0, 0.9, 0.1], [0.0, -0.9, 0.1]],
            'scores': [0.3, -0.1],
            'method': 'region occlusion',
        },
        'neighbors': [
            {'text': 'The quick brown fox jumps.', 'cosine': 0.91, 'subject': 'ZDM', 'is_true_sentence': True},
            {'text': 'He was born in Hawaii.', 'cosine': 0.62, 'subject': 'ZAB', 'is_true_sentence': False},
            {'text': 'The film won three awards.', 'cosine': 0.44, 'subject': 'ZKW', 'is_true_sentence': False},
        ],
        'decode': None,
        'disclaimer': DISCLAIMER,
        'provenance': {
            'ckpt': 'res/experiments/run/checkpoints/best.pt',
            'ckpt_sha256': 'a' * 64,
            'run_name': 'exp16_lens_run',
            'git_commit': 'abc123def4567890',
            'train_holdout': 'ZAB',
        },
    }


def _decode_lens() -> dict[str, Any]:
    """An encode capture upgraded to decode mode: a generation trace with evidence and the null control."""
    data = _encode_lens()
    data['mode'] = 'decode'
    data['decode'] = {
        'generated': 'a fox jumped over',
        'tokens': ['a', 'fox', 'jumped', 'over'],
        'slot_influence': [0.4, 0.1, 0.02, 0.3],
        'word_evidence': [[0, 0, 0.2], [1, 3, 0.8], [2, 4, 0.5]],
        'null_prefix_generated': 'the the the the',
    }

    return data


def _write_lens(path: Path, data: dict[str, Any]) -> Path:
    """Writes a fabricated lens.json the way `zte-lens` lays one out."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')

    return path


def _payload(html: str) -> str:
    """Extracts the inlined JSON island, so assertions never match plotly.js itself."""
    return html.split('<script id="lens-data" type="application/json">', 1)[1].split('</script>', 1)[0]


def test_build_lens_page_writes_a_self_contained_page(tmp_path: Path) -> None:
    """An encode capture lands as one HTML file with every panel title, the payload and no unreplaced tokens."""
    src = _write_lens(tmp_path / 'lens.json', _encode_lens())

    out = build_lens_page(src, tmp_path / 'LENS.html')

    assert out.is_file() and out.suffix == '.html'
    html = out.read_text(encoding='utf-8')
    for title in PANEL_TITLES:
        assert title in html

    payload = _payload(html)
    assert 'quick brown fox' in payload
    assert 'abc123def4567890' in payload
    assert 'Plotly' in html

    # Builder tokens must all have been consumed; a leftover token is a half-rendered page.
    assert '__LENS_' not in html
    assert '/*__CSS__*/' not in html
    assert '/*__JS__*/' not in html


def test_the_disclaimer_is_mandatory_and_rendered(tmp_path: Path) -> None:
    """The disclaimer travels into the page verbatim, and a capture without one is refused outright."""
    src = _write_lens(tmp_path / 'lens.json', _encode_lens())
    html = build_lens_page(src, tmp_path / 'LENS.html').read_text(encoding='utf-8')

    assert DISCLAIMER in _payload(html)
    assert 'no number here is a headline' in html

    stripped = _encode_lens()
    stripped['disclaimer'] = ''
    src = _write_lens(tmp_path / 'bare.json', stripped)
    with pytest.raises(ValueError, match='disclaimer'):
        build_lens_page(src, tmp_path / 'BARE.html')

    del stripped['disclaimer']
    src = _write_lens(tmp_path / 'missing.json', stripped)
    with pytest.raises(ValueError, match='missing required keys'):
        build_lens_page(src, tmp_path / 'MISSING.html')


def test_an_encode_capture_hides_the_decode_panel(tmp_path: Path) -> None:
    """With no decode trace the panel ships hidden, so an encode page never shows an empty generation."""
    src = _write_lens(tmp_path / 'lens.json', _encode_lens())
    html = build_lens_page(src, tmp_path / 'LENS.html').read_text(encoding='utf-8')

    assert 'id="decodepanel" hidden' in html


def test_a_decode_capture_renders_the_trace_panel(tmp_path: Path) -> None:
    """A decode capture unhides the trace panel and carries the generation and its null control."""
    src = _write_lens(tmp_path / 'lens.json', _decode_lens())
    html = build_lens_page(src, tmp_path / 'LENS.html').read_text(encoding='utf-8')

    assert 'id="decodepanel">' in html
    payload = _payload(html)
    assert 'a fox jumped over' in payload
    assert 'the the the the' in payload
    assert 'no brain attached' in html


def test_null_channel_saliency_shows_the_honest_note(tmp_path: Path) -> None:
    """With no montage the scalp plot ships hidden and the honest note ships visible -- and vice versa."""
    data = _encode_lens()
    data['channel_saliency'] = None
    src = _write_lens(tmp_path / 'lens.json', data)
    html = build_lens_page(src, tmp_path / 'LENS.html').read_text(encoding='utf-8')

    assert 'id="scalp" hidden' in html
    assert 'id="scalpnote">' in html

    src = _write_lens(tmp_path / 'full.json', _encode_lens())
    html = build_lens_page(src, tmp_path / 'FULL.html').read_text(encoding='utf-8')

    assert 'id="scalp">' in html
    assert 'id="scalpnote" hidden' in html


def test_non_finite_values_never_reach_the_payload(tmp_path: Path) -> None:
    """NaN and infinity in a capture become JSON null, never a literal the browser cannot parse."""
    data = _decode_lens()
    data['word_saliency']['scores'][1] = math.nan
    data['channel_saliency']['scores'][0] = math.inf
    data['decode']['slot_influence'][2] = math.nan
    src = _write_lens(tmp_path / 'lens.json', data)

    payload = _payload(build_lens_page(src, tmp_path / 'LENS.html').read_text(encoding='utf-8'))

    assert 'NaN' not in payload
    assert 'Infinity' not in payload
    json.loads(payload)


def test_a_training_brain_capture_keeps_its_flag(tmp_path: Path) -> None:
    """A non-holdout reading keeps is_holdout false in the payload, so the warning badge has its truth to render."""
    data = _encode_lens()
    data['reading']['is_holdout'] = False
    src = _write_lens(tmp_path / 'lens.json', data)

    html = build_lens_page(src, tmp_path / 'LENS.html').read_text(encoding='utf-8')

    assert '"is_holdout":false' in _payload(html)
    assert 'TRAINING brain' in html


def test_malformed_lens_json_raises(tmp_path: Path) -> None:
    """A missing file, invalid JSON, a wrong shape or an unknown mode fails loudly with the reason."""
    with pytest.raises(FileNotFoundError, match='lens'):
        build_lens_page(tmp_path / 'absent.json', tmp_path / 'LENS.html')

    bad = tmp_path / 'lens.json'
    bad.write_text('not json {', encoding='utf-8')
    with pytest.raises(ValueError, match='not valid JSON'):
        build_lens_page(bad, tmp_path / 'LENS.html')

    bad.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='JSON object'):
        build_lens_page(bad, tmp_path / 'LENS.html')

    bad.write_text(json.dumps({'mode': 'encode'}), encoding='utf-8')
    with pytest.raises(ValueError, match='missing required keys'):
        build_lens_page(bad, tmp_path / 'LENS.html')

    wrong = _encode_lens()
    wrong['mode'] = 'dream'
    bad.write_text(json.dumps(wrong), encoding='utf-8')
    with pytest.raises(ValueError, match='unknown mode'):
        build_lens_page(bad, tmp_path / 'LENS.html')


def test_every_lens_panel_carries_a_direction_cue(tmp_path: Path) -> None:
    """Each panel states which way is better -- warm/cool, cosine, evidence weight and spoke length are all named."""
    src = _write_lens(tmp_path / 'lens.json', _decode_lens())

    html = build_lens_page(src, tmp_path / 'LENS.html').read_text(encoding='utf-8')

    assert 'warm = leaned on &middot; cool = pushed away' in html
    assert 'bigger + warmer = more influence' in html
    assert 'cosine, higher = closer' in html
    assert 'darker cell = more evidence weight' in html
    assert 'longer spoke = more influence' in html


def test_the_scalp_gap_note_names_the_gap_and_its_fix(tmp_path: Path) -> None:
    """The montage-less scalp note says exactly what is missing and what run would fill the panel."""
    data = _encode_lens()
    data['channel_saliency'] = None
    src = _write_lens(tmp_path / 'lens.json', data)

    html = build_lens_page(src, tmp_path / 'LENS.html').read_text(encoding='utf-8')

    assert 'No montage geometry travelled with this checkpoint' in html
    assert 'zte-lens' in html
    assert 'class="emptyicon"' in html
