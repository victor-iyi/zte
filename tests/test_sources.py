"""Tests for dataset-source resolution — in particular, only extracting .mat-bearing zips."""

from __future__ import annotations

import zipfile
from pathlib import Path

from zte.data.io.sources import _zip_has_mat, resolve_source


def _zip(path: Path, members: dict[str, bytes]) -> Path:
    """Writes a zip with the given `{name: bytes}` members."""
    with zipfile.ZipFile(path, 'w') as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_zip_has_mat_reads_index_only(tmp_path: Path) -> None:
    assert _zip_has_mat(_zip(tmp_path / 'a.zip', {'resultsZAB_SR.mat': b'x'}))
    assert not _zip_has_mat(_zip(tmp_path / 'b.zip', {'run.py': b'print(1)'}))
    assert not _zip_has_mat(tmp_path / 'missing.zip')  # unreadable -> False, not a crash


def test_resolve_source_extracts_only_data_zips(tmp_path: Path) -> None:
    src = tmp_path / 'ZuCo Dataset'
    src.mkdir()
    _zip(src / 'task1_SR.zip', {'resultsZAB_SR.mat': b'MATLAB5.0 fake'})
    _zip(src / 'scripts.zip', {'run.py': b'print(1)'})  # unrelated — must be skipped
    _zip(src / 'resources.zip', {'notes.txt': b'x'})  # unrelated — must be skipped
    ext = tmp_path / 'extracted'

    resolve_source(src, extract_dir=ext)
    files = {p.name for p in ext.rglob('*') if p.is_file() and not p.name.startswith('.')}
    assert 'resultsZAB_SR.mat' in files
    assert 'run.py' not in files and 'notes.txt' not in files


def _zuco_folder(root: Path) -> Path:
    """A folder of ZuCo task zips (SR/NR/TSR, two subjects) plus an unrelated scripts.zip."""
    src = root / 'ZuCo Dataset'
    src.mkdir()
    _zip(src / 'task1 - SR.zip', {'resultsZAB_SR.mat': b'x', 'resultsZDM_SR.mat': b'x'})
    _zip(src / 'task2 - NR.zip', {'resultsZAB_NR.mat': b'x', 'resultsZDM_NR.mat': b'x'})
    _zip(src / 'task3 - TSR.zip', {'resultsZAB_TSR.mat': b'x'})
    _zip(src / 'scripts.zip', {'run.py': b'x'})
    return src


def test_task_selective_extraction_skips_unneeded_tasks(tmp_path: Path) -> None:
    src = _zuco_folder(tmp_path)
    ext = tmp_path / 'zuco_extracted'
    resolve_source(src, extract_dir=ext, tasks=['SR', 'NR'])
    mats = {p.name for p in ext.rglob('*.mat')}
    assert mats == {
        'resultsZAB_SR.mat',
        'resultsZDM_SR.mat',
        'resultsZAB_NR.mat',
        'resultsZDM_NR.mat',
    }
    assert not any('TSR' in m for m in mats)  # task3 never unpacked
    assert not (ext / 'run.py').exists()  # scripts.zip never unpacked


def test_subject_and_task_filters_compose(tmp_path: Path) -> None:
    src = _zuco_folder(tmp_path)
    ext = tmp_path / 'zuco_extracted'
    resolve_source(src, extract_dir=ext, tasks=['SR'], subjects=['ZAB'])
    assert {p.name for p in ext.rglob('*.mat')} == {'resultsZAB_SR.mat'}


def test_extraction_is_idempotent_unless_overwrite(tmp_path: Path) -> None:
    src = _zuco_folder(tmp_path)
    ext = tmp_path / 'zuco_extracted'
    resolve_source(src, extract_dir=ext, tasks=['SR'])
    target = next(ext.rglob('resultsZAB_SR.mat'))
    target.write_bytes(b'EDITED')  # simulate downstream edit
    resolve_source(src, extract_dir=ext, tasks=['SR'])  # no overwrite -> left as-is
    assert target.read_bytes() == b'EDITED'
    resolve_source(src, extract_dir=ext, tasks=['SR'], overwrite=True)  # forced re-extract
    assert target.read_bytes() == b'x'


def test_resolve_source_prefers_extracted_mat_over_stray_zip(tmp_path: Path) -> None:
    src = tmp_path / 'data'
    src.mkdir()
    (src / 'resultsZAB_SR.mat').write_bytes(b'MATLAB5.0 fake')
    _zip(src / 'scripts.zip', {'run.py': b'print(1)'})
    out = resolve_source(src, extract_dir=tmp_path / 'extracted')
    assert out == src  # used in place; the stray zip is never unpacked
    assert not (tmp_path / 'extracted').exists()
