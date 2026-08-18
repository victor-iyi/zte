"""Tests for the Parallax Chamber page builder."""

import json
import math
from pathlib import Path
from typing import Any, Final

import pytest

from zte.parallax.chamber import build_chamber

# Every panel heading the template must carry, so a missing section is a broken page, not a silent gap.
PANEL_TITLES: Final[tuple[str, ...]] = (
    'The parallax view',
    'Transfer flow',
    'Capacity dials',
    'Percentile rain',
    'Menu decomposition',
    'CKA triad',
)

# One explicit direction cue per panel; a chart without a better/worse reading is not done.
DIRECTION_CUES: Final[tuple[str, ...]] = (
    'hover rank pctl: higher is better &middot; axes arbitrary',
    'wider link = more lift &middot; chance = 0.5',
    'larger K is better &middot; chance = 1/K',
    'higher is better &middot; chance = 0.5',
    'right is better &middot; chance = 0.5',
    'thicker edge = closer geometry &middot; CKA 1 = identical',
)


def _point(text: str, cluster: int, n_words: int, dx: float) -> dict[str, Any]:
    """One fabricated sentence point seen by both the NR and SR models."""
    return {
        'text': text,
        'cluster': cluster,
        'n_words': n_words,
        'views': {'NR': [dx, 0.1, 0.2], 'SR': [dx + 0.05, 0.12, 0.18]},
        'rank_percentile': {'NR': 0.62, 'SR': 0.48},
    }


def _cell(rank_percentile: float) -> dict[str, Any]:
    """One fabricated transfer-cell summary."""
    return {
        'rank_percentile': rank_percentile,
        'ci': [rank_percentile - 0.05, rank_percentile + 0.05],
        'top1': 0.02,
        'chance': 1.0 / 300.0,
        'n_seeds': 2,
    }


def _chamber_data() -> dict[str, Any]:
    """A minimal two-task report: six points, a full 2x2 transfer block, capacity and one CKA pair."""
    return {
        'holdout': 'ZAB',
        'tasks': ['NR', 'SR'],
        'points': {
            'NR': [
                _point('The quick brown fox jumps over the lazy dog.', 0, 9, 0.0),
                _point('He was born in Hawaii.', 1, 5, 0.3),
                _point('She studied astrophysics at night.', 2, 5, 0.6),
            ],
            'SR': [
                _point('The film won three awards.', 0, 5, 0.1),
                _point('A senator from Ohio retired.', 1, 5, 0.4),
                _point('The novel sold a million copies.', 2, 6, 0.7),
            ],
        },
        'transfer': {
            'NR': {'NR': _cell(0.71), 'SR': _cell(0.52)},
            'SR': {'NR': _cell(0.49), 'SR': _cell(0.68)},
        },
        'capacity': {
            'NR': {'k_at_target': 8, 'k2_accuracy': 0.91},
            'SR': {'k_at_target': None, 'k2_accuracy': 0.55},
        },
        'cka': {'NR|SR': 0.63},
    }


def _write_report(
    report_dir: Path,
    data: dict[str, Any],
    *,
    with_provenance: bool = True,
    decomposition: dict[str, Any] | None = None,
) -> None:
    """Writes a fabricated report directory the way `zte-parallax report` lays one out."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'CHAMBER_DATA.json').write_text(json.dumps(data), encoding='utf-8')
    if with_provenance:
        parallax: dict[str, Any] = {
            'study': 'parallax',
            'holdout': 'ZAB',
            'seeds': [0, 1],
            'provenance': {'git_commit': 'abc123def4567890', 'run_name': 'parallax_nr_loZAB_s0'},
        }
        if decomposition is not None:
            parallax['menu_decomposition'] = decomposition
        (report_dir / 'PARALLAX.json').write_text(json.dumps(parallax), encoding='utf-8')


def _payload(html: str) -> str:
    """Extracts the inlined JSON island, so assertions never match plotly.js itself."""
    return html.split('<script id="chamber-data" type="application/json">', 1)[1].split('</script>', 1)[0]


def test_build_chamber_writes_a_self_contained_page(tmp_path: Path) -> None:
    """The chamber lands as one HTML file carrying every panel, the payload and the provenance."""
    report = tmp_path / 'report'
    _write_report(report, _chamber_data())

    out = build_chamber(report, tmp_path / 'chamber.html')

    assert out.is_file() and out.suffix == '.html'
    html = out.read_text(encoding='utf-8')
    for title in PANEL_TITLES:
        assert title in html

    payload = _payload(html)
    assert 'quick brown fox' in payload
    assert 'abc123def4567890' in payload
    assert 'Plotly' in html


def test_non_finite_values_never_reach_the_payload(tmp_path: Path) -> None:
    """NaN and infinity in the report become JSON null, never a literal the browser cannot parse."""
    data = _chamber_data()
    data['transfer']['SR']['NR']['rank_percentile'] = math.nan
    data['points']['NR'][0]['rank_percentile']['SR'] = math.inf
    report = tmp_path / 'report'
    _write_report(report, data)

    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')
    payload = _payload(html)

    assert 'NaN' not in payload
    assert 'Infinity' not in payload
    json.loads(payload)


def test_a_missing_task_degrades_instead_of_crashing(tmp_path: Path) -> None:
    """A report with TSR listed but never trained still renders: panels drop the task, nothing raises."""
    data = _chamber_data()
    data['tasks'] = ['NR', 'SR', 'TSR']
    report = tmp_path / 'report'
    _write_report(report, data, with_provenance=False)

    out = build_chamber(report, tmp_path / 'chamber.html')

    assert out.is_file()
    assert 'quick brown fox' in _payload(out.read_text(encoding='utf-8'))


def test_a_report_dir_without_chamber_data_raises(tmp_path: Path) -> None:
    """An empty report directory is a user error and says exactly what is missing."""
    empty = tmp_path / 'empty'
    empty.mkdir()

    with pytest.raises(FileNotFoundError, match='CHAMBER_DATA.json'):
        build_chamber(empty, tmp_path / 'chamber.html')


def test_malformed_chamber_data_raises(tmp_path: Path) -> None:
    """Invalid JSON or a wrong top-level shape fails loudly rather than rendering an empty page."""
    report = tmp_path / 'report'
    report.mkdir()

    (report / 'CHAMBER_DATA.json').write_text('not json {', encoding='utf-8')
    with pytest.raises(ValueError, match='not valid JSON'):
        build_chamber(report, tmp_path / 'chamber.html')

    (report / 'CHAMBER_DATA.json').write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='JSON object'):
        build_chamber(report, tmp_path / 'chamber.html')

    (report / 'CHAMBER_DATA.json').write_text(json.dumps({'tasks': ['NR']}), encoding='utf-8')
    with pytest.raises(ValueError, match='missing required keys'):
        build_chamber(report, tmp_path / 'chamber.html')


def test_every_panel_carries_a_direction_cue(tmp_path: Path) -> None:
    """Each panel header states which way is better and where chance sits, so no chart ships unexplained."""
    report = tmp_path / 'report'
    _write_report(report, _chamber_data())

    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')

    for cue in DIRECTION_CUES:
        assert cue in html

    # The chance and target reference lines are labelled in text, never an unlabelled dash.
    assert 'chance = 0.5' in html
    assert 'chance 0.5' in html
    assert 'target 0.8' in html


def test_a_missing_task_names_itself_in_every_gap_note(tmp_path: Path) -> None:
    """A task declared but never measured renders a gap note naming it and the notebook cell that fills it."""
    data = _chamber_data()
    data['tasks'] = ['NR', 'SR', 'TSR']
    report = tmp_path / 'report'
    _write_report(report, data)

    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')

    assert 'TSR has no sentence points in this report' in html
    assert 'TSR has no transfer cells in this report' in html
    assert 'TSR has no menu-capacity audit in this report' in html
    assert 'TSR has no percentile drops in this report' in html
    assert 'TSR is missing from the CKA triad' in html
    assert 'notebooks/zte_parallax.ipynb' in html


def test_a_complete_report_carries_no_gap_notes(tmp_path: Path) -> None:
    """When every declared task is measured, no gap note is rendered anywhere on the page."""
    report = tmp_path / 'report'
    _write_report(report, _chamber_data())

    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')

    assert '<div class="missnote">' not in html


def test_an_uncertified_dial_still_shows_the_two_way_number(tmp_path: Path) -> None:
    """A 'none certified' arm keeps its measured K=2 accuracy on the dial against labelled chance/target ticks."""
    report = tmp_path / 'report'
    _write_report(report, _chamber_data())

    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')

    # SR is the uncertified arm in the fixture; its 2-way accuracy must reach the page regardless.
    assert '"k_at_target":null' in _payload(html)
    assert '"k2_accuracy":0.55' in _payload(html)
    assert 'none certified' in html
    assert 'chance 0.5' in html
    assert 'target 0.8' in html


def test_enrolled_capacity_keys_travel_and_are_named(tmp_path: Path) -> None:
    """Enrolled needles ride along when the audit carries them, and the legend names both scoring rules."""
    data = _chamber_data()
    data['capacity']['NR']['enrolled_k_at_target'] = 16
    data['capacity']['NR']['enrolled_k2_accuracy'] = 0.97
    report = tmp_path / 'report'
    _write_report(report, data)

    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')

    assert '"enrolled_k_at_target":16' in _payload(html)
    assert '"enrolled_k2_accuracy":0.97' in _payload(html)
    assert 'prototype = one reference per sentence; enrolled = best of the enrolled readings' in html
    assert 'enrolled_k_at_target' in html


def test_a_gamed_open_menu_travels_with_its_badge(tmp_path: Path) -> None:
    """A gamed capacity block renders the server-side disqualification note; a healthy page carries no trace of it.

    The note markup is built in Python from the capacity blocks alone, so its presence is a rendering decision,
    not a pass-through -- a page built from healthy data must not contain the note class at all.
    """
    healthy = tmp_path / 'healthy'
    _write_report(healthy, _chamber_data())
    healthy_html = build_chamber(healthy, tmp_path / 'healthy.html').read_text(encoding='utf-8')
    assert 'srv-gamed-note' not in healthy_html

    data = _chamber_data()
    data['capacity']['NR']['open'] = {'k2_accuracy': 0.707, 'gamed': True}
    report = tmp_path / 'report'
    _write_report(report, data)
    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')

    assert '"gamed":true' in _payload(html)
    assert 'srv-gamed-note' in html
    assert 'NR: length-gamed' in html and 'no capacity may be read' in html
    assert html.count('srv-gamed-note') == 1, 'only the gamed arm is flagged'


def test_menu_decomposition_travels_and_unhides_its_panel(tmp_path: Path) -> None:
    """A decomposition in PARALLAX.json reaches the payload and the panel ships visible, its empty state hidden."""
    decomposition = {
        'NR': {'prototype_tol0': 0.522, 'prototype_tol1': 0.526, 'best_reading_tol0': 0.707, 'best_reading_tol1': 0.71},
        'SR': {'prototype_tol0': 0.51, 'prototype_tol1': 0.52, 'best_reading_tol0': 0.68, 'best_reading_tol1': 0.69},
    }
    report = tmp_path / 'report'
    _write_report(report, _chamber_data(), decomposition=decomposition)

    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')

    payload = _payload(html)
    assert '"menu_decomposition"' in payload
    assert '"prototype_tol0":0.522' in payload
    assert '"best_reading_tol0":0.707' in payload
    assert 'id="decomp">' in html
    assert 'id="decompnote" hidden' in html


def test_a_report_without_decomposition_shows_its_empty_state(tmp_path: Path) -> None:
    """With no decomposition the panel ships its designed empty-state card, never bare axes."""
    report = tmp_path / 'report'
    _write_report(report, _chamber_data())

    html = build_chamber(report, tmp_path / 'chamber.html').read_text(encoding='utf-8')

    assert 'id="decomp" hidden' in html
    assert 'id="decompnote">' in html
    assert 'No menu decomposition travelled with this report' in html
