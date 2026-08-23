"""Work whose artifacts already match their inputs is skipped, and work whose inputs moved is redone."""

import argparse
import json
import sys
from pathlib import Path

import pytest
import torch

from zte.cli import audit as audit_mod
from zte.cli import decode as decode_mod
from zte.cli import rebaseline as rebaseline_mod
from zte.cli.support.done import is_done, mark_done, signature, stamp_for
from zte.cli.support.sources import dataset_key
from zte.config import DatasetConfig, ZTEConfig
from zte.data.synthetic import generate_synthetic_zuco


def _artifacts(directory: Path, *names: str) -> tuple[Path, ...]:
    """Writes placeholder artifacts and returns their paths, first one first."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in names)
    for path in paths:
        path.write_text('{}', encoding='utf-8')

    return paths


def _namespace(**options: object) -> argparse.Namespace:
    return argparse.Namespace(**options)


def test_identical_inputs_are_not_rebuilt(tmp_path: Path) -> None:
    """An artifact recorded against these exact inputs reports done."""
    artifacts = _artifacts(tmp_path, 'rebaseline.json', 'rebaseline.md')
    sig = signature(_namespace(n_boot=2000, seed=0), tool='rebaseline', extra={'ckpt_sha256': 'abc'})
    mark_done(artifacts, sig)

    assert is_done(artifacts, sig)


def test_a_changed_option_is_rebuilt(tmp_path: Path) -> None:
    """A knob that moved invalidates the artifact it produced, even though the files are all there."""
    artifacts = _artifacts(tmp_path, 'rebaseline.json', 'rebaseline.md')
    mark_done(artifacts, signature(_namespace(n_boot=2000, seed=0), tool='rebaseline'))

    assert not is_done(artifacts, signature(_namespace(n_boot=500, seed=0), tool='rebaseline'))


def test_a_retrained_checkpoint_is_rebuilt(tmp_path: Path) -> None:
    """The same options over different weights is different work: the digest is part of the record."""
    artifacts = _artifacts(tmp_path, 'generation.json', 'generation.jsonl')
    mark_done(artifacts, signature(_namespace(split='test'), tool='decode', extra={'ckpt_sha256': 'abc'}))

    later = signature(_namespace(split='test'), tool='decode', extra={'ckpt_sha256': 'def'})

    assert not is_done(artifacts, later)


def test_an_option_added_later_is_rebuilt(tmp_path: Path) -> None:
    """A record predating a new knob cannot vouch for it, so the artifact is redone rather than served."""
    artifacts = _artifacts(tmp_path, 'transfer.json')
    mark_done(artifacts, signature(_namespace(seed=0), tool='parallax-transfer'))

    assert not is_done(artifacts, signature(_namespace(seed=0, piece_oracle=True), tool='parallax-transfer'))


def test_a_missing_artifact_is_rebuilt(tmp_path: Path) -> None:
    """A half-written output -- one file of two -- is not done."""
    artifacts = _artifacts(tmp_path, 'rebaseline.json', 'rebaseline.md')
    sig = signature(_namespace(seed=0), tool='rebaseline')
    mark_done(artifacts, sig)
    artifacts[1].unlink()

    assert not is_done(artifacts, sig)


def test_an_unrecorded_artifact_is_rebuilt(tmp_path: Path) -> None:
    """Files from a session that kept no record are rebuilt: unprovable is not the same as fresh."""
    artifacts = _artifacts(tmp_path, 'rebaseline.json', 'rebaseline.md')

    assert not is_done(artifacts, signature(_namespace(seed=0), tool='rebaseline'))


def test_an_unreadable_record_is_rebuilt(tmp_path: Path) -> None:
    """A truncated record -- a session killed mid-write -- fails closed."""
    artifacts = _artifacts(tmp_path, 'rebaseline.json', 'rebaseline.md')
    sig = signature(_namespace(seed=0), tool='rebaseline')
    mark_done(artifacts, sig)
    stamp_for(artifacts[0]).write_text('{"signature":', encoding='utf-8')

    assert not is_done(artifacts, sig)


def test_an_artifact_rewritten_underneath_is_rebuilt(tmp_path: Path) -> None:
    """`zte-run` re-evaluating into the same directory invalidates a decode record that still matches its inputs."""
    artifacts = _artifacts(tmp_path, 'generation.json', 'generation.jsonl')
    sig = signature(_namespace(split='test'), tool='decode')
    mark_done(artifacts, sig)
    artifacts[0].write_text('{"generation": "written by another command"}', encoding='utf-8')

    assert not is_done(artifacts, sig)


def test_force_rebuilds_a_matching_artifact(tmp_path: Path) -> None:
    """`--force` is the override: it redoes work that matches its record perfectly."""
    artifacts = _artifacts(tmp_path, 'rebaseline.json', 'rebaseline.md')
    sig = signature(_namespace(seed=0), tool='rebaseline')
    mark_done(artifacts, sig)

    assert not is_done(artifacts, sig, force=True)


def test_two_commands_in_one_directory_keep_separate_records(tmp_path: Path) -> None:
    """A record is named after its artifact, so one command's output never vouches for another's."""
    (audited,) = _artifacts(tmp_path, 'rebaseline.json')
    (decoded,) = _artifacts(tmp_path, 'generation.json')
    mark_done([audited], signature(_namespace(seed=0), tool='rebaseline'))

    assert stamp_for(audited) != stamp_for(decoded)
    assert not is_done([decoded], signature(_namespace(seed=0), tool='decode'))


def test_a_moved_data_root_does_not_rebuild(tmp_path: Path) -> None:
    """The same recording keys the same from any machine: a Drive path is not part of the identity."""
    colab = DatasetConfig(root='/content/drive/MyDrive/ZuCo Dataset', cache_dir=str(tmp_path / 'a'))
    local = DatasetConfig(root='res/data/zuco_extracted', cache_dir=str(tmp_path / 'b'))

    assert dataset_key(colab) == dataset_key(local)


def test_a_changed_representation_changes_the_data_identity(tmp_path: Path) -> None:
    """Different processing is different data, and the key says so."""
    band = DatasetConfig(root='res/data/zuco_extracted', cache_dir=str(tmp_path))
    raw = DatasetConfig(root='res/data/zuco_extracted', cache_dir=str(tmp_path), representation='raw')

    assert dataset_key(band) != dataset_key(raw)


# -- end to end, through `zte-audit` ----------------------------------------- #


@pytest.fixture(scope='module')
def audit_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A YAML over a tiny synthetic tree, so `zte-audit` runs offline against a stable corpus."""
    root = tmp_path_factory.mktemp('zuco_done')
    generate_synthetic_zuco(root, subjects=('ZAB', 'ZDM'), tasks=('SR', 'NR'), n_sentences=6, show_progress=False)
    config = ZTEConfig(dataset=DatasetConfig(root=str(root), cache_dir=str(root / 'cache')))
    path = root / 'audit.yaml'
    config.to_yaml(path)

    return path


@pytest.fixture
def audit_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Counts how often the audit actually recomputes its report."""
    calls: list[int] = []
    real = audit_mod.confound_report

    def counted(words: object) -> dict[str, object]:
        calls.append(1)
        return real(words)

    monkeypatch.setattr(audit_mod, 'confound_report', counted)

    return calls


def _run_audit(monkeypatch: pytest.MonkeyPatch, config: Path, out: Path, *extra: str) -> None:
    monkeypatch.setattr(sys, 'argv', ['zte-audit', '--config', str(config), '--out', str(out), *extra])
    audit_mod.main()


def test_audit_reruns_are_free(
    audit_config: Path, audit_calls: list[int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second identical run writes nothing new -- the notebook cell can be re-run at no cost."""
    out = tmp_path / 'confound_audit.md'
    _run_audit(monkeypatch, audit_config, out)
    _run_audit(monkeypatch, audit_config, out)

    assert audit_calls == [1]
    assert out.is_file()
    assert json.loads(stamp_for(out).read_text(encoding='utf-8'))['signature']['tool'] == 'audit'


def test_audit_redoes_the_work_when_an_option_changes(
    audit_config: Path, audit_calls: list[int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different tokeniser is a different report, and the guard does not stand in its way."""
    out = tmp_path / 'confound_audit.md'
    _run_audit(monkeypatch, audit_config, out, '--tokenizer', 'gpt2')
    _run_audit(monkeypatch, audit_config, out, '--tokenizer', 'distilgpt2')

    assert len(audit_calls) == 2


def test_audit_force_redoes_the_work(
    audit_config: Path, audit_calls: list[int], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--force` overrides a perfectly matching record."""
    out = tmp_path / 'confound_audit.md'
    _run_audit(monkeypatch, audit_config, out)
    _run_audit(monkeypatch, audit_config, out, '--force')

    assert len(audit_calls) == 2


# -- the guard sits in front of the expensive half --------------------------- #


def _fake_ckpt(tmp_path: Path) -> Path:
    """A config-only checkpoint payload: enough for a command to read its run config and hash the file."""
    path = tmp_path / 'best.pt'
    torch.save({'config': ZTEConfig(run_name='guarded').to_dict(), 'model': {}, 'epoch': 3, 'step': 30}, path)

    return path


def test_rebaseline_skips_before_it_builds_the_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The second audit of the same checkpoint never stages the bundle -- that is where the minutes are."""
    builds: list[int] = []

    def fake_dataset(args: argparse.Namespace, dataset: object) -> str:
        builds.append(1)
        return 'built'

    def fake_rebaseline(ckpt: object, dataset: object, *, out_dir: Path, **kwargs: object) -> dict[str, object]:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'rebaseline.json').write_text('{}', encoding='utf-8')
        (out_dir / 'rebaseline.md').write_text('# audit', encoding='utf-8')
        return {}

    monkeypatch.setattr(rebaseline_mod, 'dataset_for_config', fake_dataset)
    monkeypatch.setattr(rebaseline_mod, 'run_rebaseline', fake_rebaseline)
    argv = ['zte-rebaseline', '--ckpt', str(_fake_ckpt(tmp_path)), '--synthetic', '--out', str(tmp_path / 'rb')]

    monkeypatch.setattr(sys, 'argv', argv)
    rebaseline_mod.main()
    monkeypatch.setattr(sys, 'argv', argv)
    rebaseline_mod.main()

    assert builds == [1]

    monkeypatch.setattr(sys, 'argv', [*argv, '--force'])
    rebaseline_mod.main()

    assert builds == [1, 1]


def test_decode_skips_before_it_loads_the_language_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A decode already written from these weights is not generated again."""
    decodes: list[int] = []

    def fake_dataset(args: argparse.Namespace, dataset: object) -> str:
        return 'built'

    def fake_decode(decoder: object, dataset: object, indices: object, **kwargs: object) -> dict[str, object]:
        decodes.append(1)
        out_dir = Path(str(kwargs['out_dir']))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'generation.json').write_text('{}', encoding='utf-8')
        (out_dir / 'generation.jsonl').write_text('', encoding='utf-8')
        return {}

    monkeypatch.setattr(decode_mod, 'dataset_for_config', fake_dataset)
    monkeypatch.setattr(decode_mod, 'decode_evaluation', fake_decode)
    monkeypatch.setattr(decode_mod, 'split_indices', lambda dataset, config, split: [0])
    monkeypatch.setattr(
        'zte.inference.decode.ZTEDecoder.from_checkpoint',
        classmethod(lambda cls, ckpt, dataset=None, device=None: 'stub-decoder'),
    )
    argv = ['zte-decode', '--ckpt', str(_fake_ckpt(tmp_path)), '--synthetic', '--out', str(tmp_path / 'dec')]

    monkeypatch.setattr(sys, 'argv', argv)
    decode_mod.main()
    monkeypatch.setattr(sys, 'argv', argv)
    decode_mod.main()

    assert decodes == [1]


def test_decode_redoes_the_work_for_a_different_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same checkpoint, different cell: the guard must not serve the test split as the validation one."""
    decodes: list[str] = []

    def fake_decode(decoder: object, dataset: object, indices: object, **kwargs: object) -> dict[str, object]:
        decodes.append(str(kwargs['split']))
        out_dir = Path(str(kwargs['out_dir']))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'generation.json').write_text('{}', encoding='utf-8')
        (out_dir / 'generation.jsonl').write_text('', encoding='utf-8')
        return {}

    monkeypatch.setattr(decode_mod, 'dataset_for_config', lambda args, dataset: 'built')
    monkeypatch.setattr(decode_mod, 'decode_evaluation', fake_decode)
    monkeypatch.setattr(decode_mod, 'split_indices', lambda dataset, config, split: [0])
    monkeypatch.setattr(
        'zte.inference.decode.ZTEDecoder.from_checkpoint',
        classmethod(lambda cls, ckpt, dataset=None, device=None: 'stub-decoder'),
    )
    argv = ['zte-decode', '--ckpt', str(_fake_ckpt(tmp_path)), '--synthetic', '--out', str(tmp_path / 'dec')]

    monkeypatch.setattr(sys, 'argv', [*argv, '--split', 'test'])
    decode_mod.main()
    monkeypatch.setattr(sys, 'argv', [*argv, '--split', 'val'])
    decode_mod.main()

    assert decodes == ['test', 'val']
