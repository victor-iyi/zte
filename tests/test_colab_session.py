"""Tests for the dated Drive session layout and for finding runs and checkpoints across sessions."""

import datetime
import json
from pathlib import Path

from zte.data.cache import REMOTE_ENV_VAR
from zte.utils.session import DriveSession, discover_runs, every_session, find_checkpoint


def _make_run(
    root: Path,
    name: str,
    *,
    synthetic: bool = False,
    evaluated: bool = True,
    checkpoints: bool = False,
    catalogued: bool = True,
) -> Path:
    """Fabricates a run directory in the order `zte-run` writes it: config, then evaluation, then manifest."""
    run = root / name
    run.mkdir(parents=True)
    (run / 'config.yaml').write_text(f'run_name: {name}\n', encoding='utf-8')
    if catalogued:
        (run / 'manifest.json').write_text(json.dumps({'run_name': name, 'synthetic': synthetic}), encoding='utf-8')
    if evaluated:
        (run / 'evaluation').mkdir()
        (run / 'evaluation' / 'metrics.json').write_text(json.dumps({'ok': True}), encoding='utf-8')
    if checkpoints:
        (run / 'checkpoints').mkdir()
        (run / 'checkpoints' / 'best.pt').write_bytes(b'0')
        (run / 'checkpoints' / 'last.pt').write_bytes(b'1')
    return run


def _session(root: Path, date: str, *, experiments: bool = True) -> Path:
    """Fabricates a dated session folder on a stand-in Drive root."""
    dated = root / date
    runs = dated / 'experiments'
    if experiments:
        runs.mkdir(parents=True)
    else:
        dated.mkdir(parents=True)
    return runs


def test_session_derives_every_path_from_the_drive_root(tmp_path: Path) -> None:
    """A session names its data, runs, analysis, archives and prepared store off the one Drive root."""
    session = DriveSession.create(tmp_path, run_date='2026-08-13')

    assert session.data_dir == tmp_path / 'ZuCo Dataset'
    assert session.session_dir == tmp_path / '2026-08-13'
    assert session.drive_runs == tmp_path / '2026-08-13' / 'experiments'
    assert session.drive_analysis == tmp_path / '2026-08-13' / 'analysis'
    assert session.drive_archives == tmp_path / '2026-08-13' / 'archives'
    assert session.prepared_drive == tmp_path / 'prepared'


def test_creating_a_session_makes_its_directories(tmp_path: Path) -> None:
    """The runs, analysis and archives folders exist afterwards, and `created` says so."""
    session = DriveSession.create(tmp_path / 'ZTE', run_date='2026-08-13')

    assert session.drive_runs.is_dir()
    assert session.drive_analysis.is_dir()
    assert session.drive_archives.is_dir()
    assert session.created is True
    assert session.drive_mounted is True
    assert session.data_dir_present is False


def test_make_dirs_off_leaves_the_drive_untouched(tmp_path: Path) -> None:
    """`make_dirs=False` resolves the paths without writing anything, and `created` reports that honestly."""
    session = DriveSession.create(tmp_path, run_date='2026-08-13', make_dirs=False)

    assert not session.session_dir.exists()
    assert session.created is False


def test_only_an_explicit_date_counts_as_resuming(tmp_path: Path) -> None:
    """Today's session is new; naming an earlier folder reopens it and is flagged as resumed."""
    fresh = DriveSession.create(tmp_path, make_dirs=False)
    resumed = DriveSession.create(tmp_path, run_date='2026-08-13', make_dirs=False)

    assert fresh.run_date == datetime.date.today().isoformat()
    assert fresh.resumed is False
    assert resumed.run_date == '2026-08-13'
    assert resumed.resumed is True


def test_write_mode_decides_where_runs_are_written(tmp_path: Path) -> None:
    """`drive` writes into the session folder; `local+mirror` writes locally, and both back up to Drive."""
    mirrored = DriveSession.create(tmp_path, run_date='2026-08-13', make_dirs=False)
    direct = DriveSession.create(tmp_path, run_date='2026-08-13', write_mode='drive', make_dirs=False)

    assert mirrored.out_root == mirrored.local_runs
    assert mirrored.out_root != mirrored.drive_runs
    assert direct.out_root == direct.drive_runs
    assert mirrored.drive_backup == direct.drive_backup == direct.drive_runs


def test_as_dict_is_json_ready(tmp_path: Path) -> None:
    """Every value survives `json.dumps`, because the notebook kernel parses this with the standard library."""
    payload = DriveSession.create(tmp_path, run_date='2026-08-13', write_mode='drive').as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload['out_root'] == payload['drive_runs'] == payload['drive_backup']
    assert payload['prepared_drive'] == str(tmp_path / 'prepared')
    assert payload['data_dir_present'] is False
    assert payload['created'] is True


def test_env_points_the_persistent_cache_at_drive(tmp_path: Path) -> None:
    """The exported environment names the session's directories and the store the bundle cache actually reads."""
    env = DriveSession.create(tmp_path, run_date='2026-08-13', make_dirs=False).env()

    assert set(env) == {
        'ZTE_DRIVE',
        'DATA_DIR',
        'DRIVE_DIR',
        'DRIVE_RUNS',
        'DRIVE_ANALYSIS',
        'DRIVE_BACKUP',
        'RUN_DATE',
        'OUT_ROOT',
        'DATA_CACHE',
        'ZTE_CACHE_REMOTE',
    }
    assert env[REMOTE_ENV_VAR] == str(tmp_path / 'prepared')
    assert env['DATA_CACHE'] != env[REMOTE_ENV_VAR]
    assert env['RUN_DATE'] == '2026-08-13'
    assert all(isinstance(value, str) for value in env.values())


def test_every_session_lists_dated_folders_newest_first(tmp_path: Path) -> None:
    """Sessions come back newest first, and a date folder with no runs directory is not offered."""
    _session(tmp_path, '2026-08-13')
    _session(tmp_path, '2026-08-16')
    _session(tmp_path, '2026-08-14', experiments=False)
    (tmp_path / 'prepared').mkdir()

    assert every_session(tmp_path) == [
        tmp_path / '2026-08-16' / 'experiments',
        tmp_path / '2026-08-13' / 'experiments',
    ]


def test_every_session_tolerates_an_unmounted_drive(tmp_path: Path) -> None:
    """A Drive root that is not there yields no sessions rather than raising."""
    assert every_session(tmp_path / 'not-mounted') == []


def test_discover_runs_keeps_the_first_root_of_a_repeated_name(tmp_path: Path) -> None:
    """A run present on Drive and locally is reported once, from the root named first."""
    drive = _session(tmp_path / 'ZTE', '2026-08-16')
    local = tmp_path / 'res' / 'experiments'
    local.mkdir(parents=True)
    _make_run(drive, 'exp16_lo_ZAB_s42')
    _make_run(local, 'exp16_lo_ZAB_s42')
    _make_run(local, 'local_only')

    found = discover_runs([drive, local])

    assert [r.name for r in found] == ['exp16_lo_ZAB_s42', 'local_only']
    assert found[0].path == drive / 'exp16_lo_ZAB_s42'
    assert (found[0].source, found[0].session) == ('drive', '2026-08-16')
    assert (found[1].source, found[1].session) == ('local', None)


def test_discover_runs_includes_a_run_that_died_before_evaluation(tmp_path: Path) -> None:
    """A trained-but-unevaluated run is still found; `evaluated` is what records the difference."""
    local = tmp_path / 'experiments'
    local.mkdir()
    _make_run(local, 'crashed', evaluated=False)
    _make_run(local, 'finished')

    by_name = {r.name: r for r in discover_runs([local])}

    assert set(by_name) == {'crashed', 'finished'}
    assert by_name['crashed'].evaluated is False
    assert by_name['finished'].evaluated is True


def test_discover_runs_finds_a_run_the_vm_killed_mid_training(tmp_path: Path) -> None:
    """`zte-run` writes `config.yaml` when it makes the directory and `manifest.json` only once training is over.

    A reclaimed Colab VM leaves exactly the first of those, and that run is the one the user came back to resume.
    """
    local = tmp_path / 'experiments'
    local.mkdir()
    killed = local / 'exp16_lo_ZAB_s42'
    killed.mkdir()
    (killed / 'config.yaml').write_text('run_name: exp16_lo_ZAB_s42\n', encoding='utf-8')

    found = discover_runs([local])

    assert [r.name for r in found] == ['exp16_lo_ZAB_s42']
    assert found[0].evaluated is False
    assert found[0].synthetic is False


def test_discover_runs_flags_synthetic_smoke_runs(tmp_path: Path) -> None:
    """A run whose manifest records `--synthetic` is marked, so it can never be quoted as a result."""
    local = tmp_path / 'experiments'
    local.mkdir()
    _make_run(local, 'smoke', synthetic=True)
    _make_run(local, 'real')

    by_name = {r.name: r for r in discover_runs([local])}

    assert by_name['smoke'].synthetic is True
    assert by_name['real'].synthetic is False


def test_discover_runs_skips_a_missing_root(tmp_path: Path) -> None:
    """A root that does not exist on this machine is passed over, not fatal."""
    local = tmp_path / 'experiments'
    local.mkdir()
    _make_run(local, 'only')

    assert [r.name for r in discover_runs([tmp_path / 'gone', local])] == ['only']


def test_find_checkpoint_prefers_the_earlier_root(tmp_path: Path) -> None:
    """The first root holding the checkpoint wins, so Drive can be searched before the local disk."""
    drive = _session(tmp_path / 'ZTE', '2026-08-16')
    local = tmp_path / 'experiments'
    local.mkdir()
    _make_run(drive, 'exp16', checkpoints=True)
    _make_run(local, 'exp16', checkpoints=True)

    assert find_checkpoint('exp16', [drive, local]) == drive / 'exp16' / 'checkpoints' / 'best.pt'
    assert find_checkpoint('exp16', [local, drive]) == local / 'exp16' / 'checkpoints' / 'best.pt'
    assert find_checkpoint('exp16', [drive, local], which='last') == drive / 'exp16' / 'checkpoints' / 'last.pt'


def test_find_checkpoint_never_falls_back_from_best_to_last(tmp_path: Path) -> None:
    """A run with only `last.pt` reports no best checkpoint rather than handing back a different model."""
    local = tmp_path / 'experiments'
    local.mkdir()
    run = _make_run(local, 'exp16', checkpoints=True)
    (run / 'checkpoints' / 'best.pt').unlink()

    assert find_checkpoint('exp16', [local]) is None
    assert find_checkpoint('exp16', [local], which='last') == run / 'checkpoints' / 'last.pt'


def test_find_checkpoint_returns_none_for_an_untrained_run(tmp_path: Path) -> None:
    """A run nobody has trained resolves to `None` rather than raising."""
    local = tmp_path / 'experiments'
    local.mkdir()

    assert find_checkpoint('never_trained', [local, tmp_path / 'gone']) is None
