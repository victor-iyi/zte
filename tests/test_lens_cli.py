"""Tests for the `zte-lens` command-line surface: flags, the holdout default, wiring, and the decode refusal."""

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from zte.cli.lens import parse_arguments, render_page, resolve_subject, run_lens, write_lens_json
from zte.cli.support.sources import DEFAULT_EXTRACT_DIR
from zte.config import ZTEConfig
from zte.data.dataset import ZuCoDataset
from zte.lens.saliency import DISCLAIMER

# --------------------------------------------------------------------------- #
# fixtures and fabrication helpers
# --------------------------------------------------------------------------- #


def _fake_ckpt(tmp_path: Path, config: ZTEConfig, extra: dict[str, Any] | None = None) -> Path:
    """Writes a config-only checkpoint payload, the shape `CheckpointManager.build_state` produces."""
    payload: dict[str, Any] = {'config': config.to_dict(), 'model': {}, 'epoch': 0, 'step': 0}
    if extra:
        payload['extra'] = extra
    path = tmp_path / 'best.pt'
    torch.save(payload, path)

    return path


def _lens_config(holdout: str | None = 'ZAB') -> ZTEConfig:
    """A run config naming a LOSO holdout, so the default subject resolves without a flag."""
    config = ZTEConfig(run_name='lens_test')
    config.train.loso_holdout_subject = holdout

    return config


def _stub_core(monkeypatch: pytest.MonkeyPatch, calls: dict[str, Any]) -> None:
    """Replaces the core report builders on the real modules, recording what the CLI hands them."""

    def select_reading(dataset: Any, subject: str, index: int = 0, contains: str | None = None) -> Any:
        calls['selection'] = (subject, index, contains)
        return types.SimpleNamespace(subject=subject, position=index)

    def lens_report(embedder: Any, dataset: Any, reading: Any, **kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        calls['embedder'] = embedder
        calls['dataset'] = dataset
        calls['reading'] = reading
        return {'mode': 'decode' if kwargs.get('decoder') is not None else 'encode', 'disclaimer': DISCLAIMER}

    monkeypatch.setattr('zte.lens.saliency.select_reading', select_reading)
    monkeypatch.setattr('zte.lens.saliency.lens_report', lens_report)
    monkeypatch.setattr(
        'zte.inference.embed.ZTEEmbedder.from_checkpoint',
        classmethod(lambda cls, ckpt, dataset=None, device=None: 'stub-embedder'),
    )


@pytest.fixture()
def bundle_dir(small_dataset: ZuCoDataset, tmp_path: Path) -> Path:
    """Saves the small synthetic dataset as a bundle, the cheapest source `run_lens` can build from."""
    out = tmp_path / 'bundle'
    small_dataset.save(out)

    return out


# --------------------------------------------------------------------------- #
# the CLI surface, flag by flag
# --------------------------------------------------------------------------- #


def test_encode_parser_defaults_match_the_contract() -> None:
    """`encode` parses with exactly the contract defaults when only the required flags are given."""
    args = parse_arguments(['encode', '--ckpt', 'best.pt', '--out', 'lens', '--synthetic'])

    assert args.command == 'encode'
    assert args.ckpt == 'best.pt' and args.out == Path('lens')
    assert args.subject is None and args.index == 0 and args.contains is None
    assert args.top_k == 10 and args.device == 'auto' and args.html is False
    assert args.synthetic is True and args.root is None and args.bundle is None
    assert args.extract_dir == DEFAULT_EXTRACT_DIR
    assert args.log_level == 'INFO'


def test_decode_parser_defaults_match_the_contract() -> None:
    """`decode` swaps `--top-k` for `--max-new-tokens 48` and keeps everything else."""
    args = parse_arguments(['decode', '--ckpt', 'best.pt', '--out', 'lens', '--synthetic'])

    assert args.command == 'decode'
    assert args.max_new_tokens == 48 and args.device == 'auto' and args.html is False
    assert args.subject is None and args.index == 0 and args.contains is None
    assert not hasattr(args, 'top_k')


def test_every_documented_flag_parses() -> None:
    """The full documented surface parses and lands on the expected destinations."""
    args = parse_arguments(
        [
            'encode',
            '--ckpt',
            'b.pt',
            '--out',
            'o',
            '--bundle',
            'bun',
            '--subject',
            'ZDM',
            '--index',
            '3',
            '--contains',
            'movie',
            '--top-k',
            '5',
            '--device',
            'cpu',
            '--html',
        ]
    )

    assert args.bundle == Path('bun') and args.subject == 'ZDM' and args.index == 3
    assert args.contains == 'movie' and args.top_k == 5 and args.device == 'cpu' and args.html is True

    decode = parse_arguments(
        ['decode', '--ckpt', 'b.pt', '--out', 'o', '--root', 'data', '--max-new-tokens', '12', '--html']
    )
    assert decode.root == Path('data') and decode.max_new_tokens == 12 and decode.html is True


def test_the_parser_refuses_a_missing_subcommand_and_two_sources() -> None:
    """No subcommand and mutually exclusive data sources both fail at parse time, never later."""
    with pytest.raises(SystemExit):
        parse_arguments([])
    with pytest.raises(SystemExit):
        parse_arguments(['encode', '--ckpt', 'b.pt', '--out', 'o', '--root', 'r', '--synthetic'])
    with pytest.raises(SystemExit):
        parse_arguments(['encode', '--ckpt', 'b.pt', '--out', 'o'])


# --------------------------------------------------------------------------- #
# subject resolution
# --------------------------------------------------------------------------- #


def test_the_subject_defaults_to_the_checkpoints_holdout() -> None:
    """An explicit `--subject` wins; otherwise the run's own LOSO holdout is inspected."""
    assert resolve_subject(_lens_config('ZAB'), 'ZKW') == 'ZKW'
    assert resolve_subject(_lens_config('ZAB'), None) == 'ZAB'


def test_no_holdout_and_no_subject_is_a_clear_refusal() -> None:
    """A checkpoint naming no holdout has no default reading, and the error says which flag to pass."""
    with pytest.raises(SystemExit, match='no LOSO holdout'):
        resolve_subject(_lens_config(None), None)


# --------------------------------------------------------------------------- #
# the artifact writer and its honesty guard
# --------------------------------------------------------------------------- #


def test_the_writer_refuses_a_report_without_the_disclaimer(tmp_path: Path) -> None:
    """MUTATION stand-in: a lens report stripped of its disclaimer must never reach disk."""
    with pytest.raises(SystemExit, match='disclaimer'):
        write_lens_json({'mode': 'encode'}, tmp_path, 'run', 'ZAB', 0)
    with pytest.raises(SystemExit, match='disclaimer'):
        write_lens_json({'mode': 'encode', 'disclaimer': 'sounds fine'}, tmp_path, 'run', 'ZAB', 0)
    assert list(tmp_path.iterdir()) == []


def test_the_writer_lays_out_the_contract_path(tmp_path: Path) -> None:
    """The artifact lands at `<out>/<run_name>_<subject>_<index>/lens.json` and round-trips."""
    report = {'mode': 'encode', 'disclaimer': DISCLAIMER}
    path = write_lens_json(report, tmp_path, 'lens_test', 'ZAB', 2)

    assert path == tmp_path / 'lens_test_ZAB_2' / 'lens.json'
    assert json.loads(path.read_text(encoding='utf-8')) == report


# --------------------------------------------------------------------------- #
# end-to-end runs against the tiny synthetic bundle
# --------------------------------------------------------------------------- #


def test_encode_wires_the_core_api_exactly(bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`encode` builds the embedder from the checkpoint and hands the core the reading, never raw indices."""
    ckpt = _fake_ckpt(tmp_path, _lens_config('ZAB'))
    out = tmp_path / 'lens'
    args = parse_arguments(
        ['encode', '--ckpt', str(ckpt), '--out', str(out), '--bundle', str(bundle_dir), '--index', '1', '--top-k', '4']
    )
    calls: dict[str, Any] = {}
    _stub_core(monkeypatch, calls)

    json_path = run_lens(args, 'encode')

    assert json_path == out / 'lens_test_ZAB_1' / 'lens.json'
    assert json.loads(json_path.read_text(encoding='utf-8'))['disclaimer'] == DISCLAIMER
    assert calls['selection'] == ('ZAB', 1, None)
    assert calls['embedder'] == 'stub-embedder' and isinstance(calls['dataset'], ZuCoDataset)
    assert calls['reading'].subject == 'ZAB'
    assert calls['decoder'] is None and calls['ckpt_path'] == Path(ckpt)
    assert calls['top_k'] == 4 and calls['max_new_tokens'] is None


def test_decode_refuses_an_encoder_only_checkpoint(tmp_path: Path) -> None:
    """A checkpoint with no trained bridge is refused by name, before any dataset is built."""
    ckpt = _fake_ckpt(tmp_path, _lens_config('ZAB'))
    args = parse_arguments(['decode', '--ckpt', str(ckpt), '--out', str(tmp_path / 'lens'), '--synthetic'])

    with pytest.raises(SystemExit, match='decoder_state.*encoder-only'):
        run_lens(args, 'decode')


def test_decode_builds_the_decoder_and_passes_its_generation_cap(
    bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A decoder checkpoint passes the refusal gate; the trace decoder and `max_new_tokens` reach the core."""
    extra = {'decoder_state': {'bridge.to_bottleneck.weight': torch.zeros(2, 2)}}
    ckpt = _fake_ckpt(tmp_path, _lens_config('ZAB'), extra=extra)
    out = tmp_path / 'lens'
    args = parse_arguments(['decode', '--ckpt', str(ckpt), '--out', str(out), '--bundle', str(bundle_dir)])
    calls: dict[str, Any] = {}
    _stub_core(monkeypatch, calls)

    stub_decoder = types.SimpleNamespace(model='stub-model', config='stub-config', device='stub-device')
    monkeypatch.setattr(
        'zte.inference.decode.ZTEDecoder.from_checkpoint',
        classmethod(lambda cls, ckpt, dataset=None, device=None: stub_decoder),
    )

    class _EmbedderStub:
        def __init__(self, model: Any, config: Any, device: Any) -> None:
            calls['embedder_init'] = (model, config, device)

    monkeypatch.setattr('zte.inference.embed.ZTEEmbedder', _EmbedderStub)

    json_path = run_lens(args, 'decode')

    assert json_path == out / 'lens_test_ZAB_0' / 'lens.json'
    assert calls['decoder'] is stub_decoder and calls['max_new_tokens'] == 48 and calls['top_k'] == 10
    assert calls['embedder_init'] == ('stub-model', 'stub-config', 'stub-device')


def test_a_selection_error_becomes_a_clear_exit(
    bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core's selection refusal surfaces as a SystemExit carrying the core's own message."""
    ckpt = _fake_ckpt(tmp_path, _lens_config('ZAB'))
    args = parse_arguments(
        ['encode', '--ckpt', str(ckpt), '--out', str(tmp_path / 'o'), '--bundle', str(bundle_dir), '--index', '999']
    )
    monkeypatch.setattr(
        'zte.inference.embed.ZTEEmbedder.from_checkpoint',
        classmethod(lambda cls, ckpt, dataset=None, device=None: 'stub-embedder'),
    )

    with pytest.raises(SystemExit, match='out of range'):
        run_lens(args, 'encode')


def test_a_missing_report_builder_degrades_to_a_clear_exit(
    bundle_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without `zte.lens.saliency` the CLI exits with a message, not a traceback."""
    monkeypatch.setitem(sys.modules, 'zte.lens.saliency', None)
    ckpt = _fake_ckpt(tmp_path, _lens_config('ZAB'))
    args = parse_arguments(
        ['encode', '--ckpt', str(ckpt), '--out', str(tmp_path / 'lens'), '--bundle', str(bundle_dir)]
    )

    with pytest.raises(SystemExit, match='lens report builder is not available'):
        run_lens(args, 'encode')


def test_a_missing_page_renderer_degrades_to_a_clear_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`--html` without the page module exits clearly; the already-written lens.json is untouched."""
    monkeypatch.setitem(sys.modules, 'zte.lens.page', None)

    with pytest.raises(SystemExit, match='page renderer is not available'):
        render_page(tmp_path / 'lens.json')
