"""The experiments INDEX merges with its mirrored Drive copy instead of clobbering it."""

import shutil
from pathlib import Path
from typing import Any

from zte.cli.run import _catalogue


def _manifest(held_out_top1: float) -> dict[str, Any]:
    """Builds the minimal manifest a catalogue row reads."""
    return {
        'dataset': {'n_words': 100},
        'evaluation': {
            'held_out_retrieval_top1': held_out_top1,
            'sentence_retrieval_top1': 0.01,
            'subject_transfer_top1': 0.02,
            'effective_rank_ratio': 0.4,
        },
    }


def _rows(index: Path) -> list[str]:
    """Data rows of a written catalogue, header and separator excluded."""
    lines = index.read_text(encoding='utf-8').splitlines()

    return [ln for ln in lines if ln.startswith('|') and not ln.startswith('| run |') and not ln.startswith('| ---')]


def test_two_sessions_with_disjoint_runs_both_survive_on_the_remote(tmp_path: Path) -> None:
    """A fresh session's catalogue write keeps the rows an earlier session already mirrored to Drive."""
    drive = tmp_path / 'drive'
    drive.mkdir()
    remote = drive / 'INDEX.md'

    # Session A on one VM writes its run and mirrors the index, exactly as the catalogue mirror does.
    session_a = tmp_path / 'vm_a'
    session_a.mkdir()
    _catalogue(session_a, 'run_a', _manifest(0.1), remote_index=remote)
    shutil.copy2(session_a / 'INDEX.md', remote)

    # Session B is a fresh VM: its local index does not exist, only the remote carries run_a.
    session_b = tmp_path / 'vm_b'
    session_b.mkdir()
    _catalogue(session_b, 'run_b', _manifest(0.2), remote_index=remote)
    shutil.copy2(session_b / 'INDEX.md', remote)

    text = remote.read_text(encoding='utf-8')
    assert '| run_a |' in text
    assert '| run_b |' in text


def test_rerun_updates_its_row_in_place(tmp_path: Path) -> None:
    """Re-cataloguing an existing run replaces its row without duplicating it or reordering the table."""
    out = tmp_path / 'vm'
    out.mkdir()
    _catalogue(out, 'run_a', _manifest(0.1))
    _catalogue(out, 'run_b', _manifest(0.2))
    _catalogue(out, 'run_a', _manifest(0.3))

    rows = _rows(out / 'INDEX.md')
    a_rows = [r for r in rows if r.startswith('| run_a |')]
    assert len(a_rows) == 1
    assert '| 0.3 |' in a_rows[0]
    assert rows.index(a_rows[0]) < rows.index(next(r for r in rows if r.startswith('| run_b |')))


def test_local_row_wins_over_the_remote_for_the_same_run(tmp_path: Path) -> None:
    """When both indexes carry the same run, the session doing the writing owns the row."""
    drive = tmp_path / 'drive'
    drive.mkdir()
    _catalogue(drive, 'run_a', _manifest(0.1))

    out = tmp_path / 'vm'
    out.mkdir()
    _catalogue(out, 'run_a', _manifest(0.9), remote_index=drive / 'INDEX.md')

    rows = _rows(out / 'INDEX.md')
    assert len(rows) == 1
    assert '| 0.9 |' in rows[0]


def test_unreachable_remote_degrades_to_local_only(tmp_path: Path) -> None:
    """A missing or unreadable remote index never blocks the local catalogue write."""
    out = tmp_path / 'vm'
    out.mkdir()
    _catalogue(out, 'run_a', _manifest(0.1), remote_index=tmp_path / 'missing' / 'INDEX.md')

    assert '| run_a |' in (out / 'INDEX.md').read_text(encoding='utf-8')


def test_pre_held_out_remote_layout_is_not_merged(tmp_path: Path) -> None:
    """Rows from an index predating the held-out column cannot be reconciled and are dropped, not mangled."""
    drive = tmp_path / 'drive'
    drive.mkdir()
    remote = drive / 'INDEX.md'
    remote.write_text(
        '# ZTE experiment catalogue\n\n| run | words | retrieval |\n| --- | --- | --- |\n| old_run | 5 | 0.1 |\n',
        encoding='utf-8',
    )

    out = tmp_path / 'vm'
    out.mkdir()
    _catalogue(out, 'run_a', _manifest(0.1), remote_index=remote)

    text = (out / 'INDEX.md').read_text(encoding='utf-8')
    assert '| old_run |' not in text
    assert '| run_a |' in text


def test_header_is_normalised_to_a_single_table(tmp_path: Path) -> None:
    """Merging never duplicates the header or the separator row."""
    drive = tmp_path / 'drive'
    drive.mkdir()
    _catalogue(drive, 'run_a', _manifest(0.1))

    out = tmp_path / 'vm'
    out.mkdir()
    _catalogue(out, 'run_b', _manifest(0.2), remote_index=drive / 'INDEX.md')

    lines = (out / 'INDEX.md').read_text(encoding='utf-8').splitlines()
    assert sum(1 for ln in lines if ln.startswith('| run |')) == 1
    assert sum(1 for ln in lines if ln.startswith('| --- |')) == 1
