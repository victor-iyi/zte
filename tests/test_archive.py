"""Tests for the run-archive utilities (zip / list / unpack / delete) and env bootstrap."""

from __future__ import annotations

import json
from pathlib import Path

from zte.utils import (
    accelerator_info,
    clean_outputs,
    delete_run,
    ensure_dirs,
    list_runs,
    unpack,
    zip_experiments,
    zip_run,
)
from zte.utils.archive import is_synthetic_run


def _make_run(root: Path, name: str, synthetic: bool | None = None) -> Path:
    """Fabricates a minimal run directory resembling a real one."""
    run = root / name
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'evaluation').mkdir()
    (run / 'cache' / 'x').mkdir(parents=True)  # heavy dir, should be excluded by default
    (run / 'checkpoints' / 'best.pt').write_bytes(b'0' * 2048)
    (run / 'checkpoints' / 'last.pt').write_bytes(b'0' * 2048)
    (run / 'checkpoints' / 'ckpt_epoch0001.pt').write_bytes(b'0' * 2048)
    (run / 'checkpoints' / 'tb').mkdir()
    (run / 'checkpoints' / 'tb' / 'events').write_bytes(b'1' * 4096)  # tb, excluded by default
    (run / 'evaluation' / 'metrics.json').write_text(json.dumps({'ok': True}))
    (run / 'config.yaml').write_text('run_name: ' + name)
    (run / 'cache' / 'x' / 'big.npz').write_bytes(b'2' * 8192)
    if synthetic is not None:
        (run / 'manifest.json').write_text(json.dumps({'run_name': name, 'synthetic': synthetic}))
    return run


def test_skip_synthetic_excludes_smoke_runs(tmp_path: Path) -> None:
    """`skip_synthetic` drops --synthetic runs but keeps real (and flag-less) ones."""
    exp = tmp_path / 'experiments'
    exp.mkdir()
    _make_run(exp, 'real', synthetic=False)
    _make_run(exp, 'smoke', synthetic=True)
    _make_run(exp, 'legacy')  # no manifest -> treated as real

    assert is_synthetic_run(exp / 'smoke') is True
    assert is_synthetic_run(exp / 'real') is False
    assert is_synthetic_run(exp / 'legacy') is False

    archive = zip_experiments(exp, out=tmp_path / 'real.zip', best_only=True, skip_synthetic=True)
    tops = set(unpack(archive, tmp_path / 'out'))
    assert tops == {'real', 'legacy'}  # smoke excluded


def test_best_only_keeps_just_best_checkpoint(tmp_path: Path) -> None:
    exp = tmp_path / 'experiments'
    exp.mkdir()
    _make_run(exp, 'runA')
    archive = zip_run(exp / 'runA', out=tmp_path / 'runA.zip', best_only=True)
    dest = tmp_path / 'out'
    unpack(archive, dest)
    ckpts = dest / 'runA' / 'checkpoints'
    assert (ckpts / 'best.pt').exists()
    assert not (ckpts / 'last.pt').exists()
    assert not (ckpts / 'ckpt_epoch0001.pt').exists()


def test_move_removes_local_run_after_zip(tmp_path: Path) -> None:
    exp = tmp_path / 'experiments'
    exp.mkdir()
    run = _make_run(exp, 'runA')
    zip_run(run, out=tmp_path / 'runA.zip', move=True)
    assert not run.exists()  # local run freed after archiving


def test_zip_excludes_heavy_dirs_and_unpacks_inference_ready(tmp_path: Path) -> None:
    exp = tmp_path / 'experiments'
    exp.mkdir()
    _make_run(exp, 'runA')

    rows = list_runs(exp)
    assert len(rows) == 1 and rows[0]['name'] == 'runA' and rows[0]['has_checkpoint']

    archive = zip_run(exp / 'runA', out=tmp_path / 'runA.zip')
    assert archive.exists()

    dest = tmp_path / 'unpacked'
    tops = unpack(archive, dest)
    assert tops == ['runA']
    # inference needs the checkpoint + config; heavy cache / tb are excluded.
    assert (dest / 'runA' / 'checkpoints' / 'best.pt').exists()
    assert (dest / 'runA' / 'config.yaml').exists()
    assert not (dest / 'runA' / 'cache').exists()
    assert not (dest / 'runA' / 'checkpoints' / 'tb').exists()


def test_zip_experiments_bundles_multiple_runs(tmp_path: Path) -> None:
    exp = tmp_path / 'experiments'
    exp.mkdir()
    _make_run(exp, 'runA')
    _make_run(exp, 'runB')
    archive = zip_experiments(exp, out=tmp_path / 'all.zip')
    tops = unpack(archive, tmp_path / 'out')
    assert set(tops) == {'runA', 'runB'}


def test_delete_run_is_guarded(tmp_path: Path) -> None:
    exp = tmp_path / 'experiments'
    exp.mkdir()
    run = _make_run(exp, 'runA')
    assert delete_run(run) is False  # dry run by default
    assert run.exists()
    assert delete_run(run, yes=True) is True
    assert not run.exists()


def test_clean_outputs_is_guarded_and_selective(tmp_path: Path) -> None:
    (tmp_path / 'res' / 'experiments' / 'r1').mkdir(parents=True)
    (tmp_path / 'res' / 'cache').mkdir(parents=True)
    (tmp_path / 'res' / 'data').mkdir(parents=True)
    # dry run removes nothing
    assert clean_outputs(['experiments', 'cache'], root=tmp_path) == []
    assert (tmp_path / 'res' / 'experiments').exists()
    # confirmed run removes just the named targets, leaving data intact
    removed = clean_outputs(['experiments', 'cache'], root=tmp_path, yes=True)
    assert {p.name for p in removed} == {'experiments', 'cache'}
    assert not (tmp_path / 'res' / 'experiments').exists()
    assert (tmp_path / 'res' / 'data').exists()
    # 'all' wipes the whole res/ tree
    clean_outputs(['all'], root=tmp_path, yes=True)
    assert not (tmp_path / 'res').exists()


def test_ensure_dirs_and_accelerator_info(tmp_path: Path) -> None:
    made = ensure_dirs(tmp_path)
    assert all(p.is_dir() for p in made)
    info = accelerator_info()
    assert info['kind'] in {'cuda', 'mps', 'xla', 'cpu'} and 'torch_version' in info
