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
    'CKA triad',
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


def _write_report(report_dir: Path, data: dict[str, Any], *, with_provenance: bool = True) -> None:
    """Writes a fabricated report directory the way `zte-parallax report` lays one out."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / 'CHAMBER_DATA.json').write_text(json.dumps(data), encoding='utf-8')
    if with_provenance:
        parallax = {
            'study': 'parallax',
            'holdout': 'ZAB',
            'seeds': [0, 1],
            'provenance': {'git_commit': 'abc123def4567890', 'run_name': 'parallax_nr_loZAB_s0'},
        }
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
