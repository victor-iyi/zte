"""`zte-visualize` -- build the offline interactive Thought-Space Explorer and Neuron Atlas HTML."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from zte.logging_utils import configure_logging, get_logger

_LOG = get_logger('cli.visualize')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-visualize` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Build the interactive ZTE Thought-Space Explorer (one offline HTML).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--run', type=str, help='A catalogued run dir (res/experiments/<name>).')
    src.add_argument(
        '--synthetic',
        action='store_true',
        help='Fabricate a tiny dataset and train two quick models.',
    )
    parser.add_argument('--out', type=Path, default=Path('res/explorer/thought_space_explorer.html'))
    parser.add_argument(
        '--kind',
        choices=['explorer', 'atlas', 'both'],
        default='explorer',
        help='Which interactive HTML(s) to emit: the Thought-Space Explorer, the Neuron Atlas, or both.',
    )
    parser.add_argument(
        '--atlas',
        action='store_true',
        help='Shorthand for --kind atlas (emit the Neuron Atlas instead of the explorer).',
    )
    parser.add_argument('--dims', type=int, choices=[2, 3], default=3)
    parser.add_argument('--max-points', type=int, default=6000)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser.parse_args()


def _add_categories(meta: pd.DataFrame, sentences: pd.DataFrame, root: str | None) -> pd.DataFrame:
    """Joins `category` / `length_band` onto word metadata from the sentence table.

    Args:
        meta (pd.DataFrame): Word metadata with `subject`, `task`, `sentence_idx`.
        sentences (pd.DataFrame): The dataset's sentence table.
        root (str | None): Optional dataset root for real category labels.
    """
    from zte.data.targets.categories import sentence_categories

    if not {'subject', 'task', 'sentence_idx'}.issubset(meta.columns):
        return meta
    cats = sentence_categories(sentences, root=root)
    keep = [c for c in ('subject', 'task', 'sentence_idx', 'category', 'length_band') if c in cats]
    merged = meta.merge(cats[keep], on=['subject', 'task', 'sentence_idx'], how='left')
    merged['category'] = merged.get('category', pd.Series(merged['task'])).fillna(merged['task'])
    merged['length_band'] = merged.get('length_band', pd.Series('', index=merged.index)).fillna('na')
    return merged


def _probe_word_len(emb: np.ndarray, meta: pd.DataFrame) -> float | None:
    """Cross-validated linear-probe score for word length (or `None` if infeasible)."""
    from zte.training.metrics import linear_probe

    if 'word_len' not in meta.columns or len(emb) < 12:
        return None
    return round(float(linear_probe(emb, meta['word_len'].to_numpy(), task='regression')['score']), 4)


def _emergence(emb: np.ndarray, meta: pd.DataFrame, raw_feats: np.ndarray | None) -> dict | None:
    """Full-embedding-space emergence report for the explorer's verdict banners.

    Computed here so the banners headline the same numbers `metrics.json` carries rather than the
    in-browser PCA-space estimate; `None` on failure, which falls the banners back to that estimate.
    """
    from zte.evaluation.analogy import analogy_report
    from zte.evaluation.emergence import emergence_report

    feats: np.ndarray | None = None
    if raw_feats is not None:
        arr = np.asarray(raw_feats)
        flat = arr.reshape(len(arr), -1) if arr.ndim > 1 else arr[:, None]
        if flat.ndim == 2 and len(flat) == len(emb):
            feats = flat  # aligned raw-feature control; optional, so omit when misaligned
    try:
        analogy = analogy_report(emb, meta, feats)
        return emergence_report(emb, meta, analogy=analogy)
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:  # pragma: no cover
        _LOG.warning('Emergence report failed (%r); banners will use the live estimate.', exc)
        return None


def _from_run(args: argparse.Namespace) -> dict:
    """Loads and re-embeds a catalogued run into explorer inputs (real ZTE embeddings)."""
    from zte.cli.evaluate import collect_embeddings
    from zte.data.dataset import ZuCoDataset
    from zte.device import resolve_device
    from zte.inference.embed import ZTEEmbedder

    run_dir = Path(args.run)
    ckpt = run_dir / 'checkpoints' / 'best.pt'
    if not ckpt.exists():
        ckpt = run_dir / 'checkpoints' / 'last.pt'
    dataset = ZuCoDataset.load(run_dir / 'bundle')
    embedder = ZTEEmbedder.from_checkpoint(ckpt, dataset, device=resolve_device(args.device))
    word_emb, word_meta, _rw, _se, _sid, _sm, word_bp = collect_embeddings(embedder, dataset)
    word_meta = _add_categories(word_meta, dataset.sentences, embedder.config.dataset.root)

    include_et = embedder.config.dataset.include_eye_tracking
    label = 'EEG + eye-tracking' if include_et else 'EEG-only'
    score = _probe_word_len(word_emb, word_meta)
    probe = {label: score} if score is not None else None
    _LOG.info('Re-embedded run %s: %d word embeddings (%s).', run_dir.name, len(word_emb), label)
    return {
        'emb': word_emb,
        'meta': word_meta,
        'eeg_only_emb': None,
        'band_power': word_bp,
        'probe_scores': probe,
        'emergence': _emergence(word_emb, word_meta, word_bp),
        'title': f'ZTE Thought-Space Explorer - {run_dir.name}',
        'atlas_title': f'ZTE Neuron Atlas - {run_dir.name}',
    }


def _quick_config(include_eye_tracking: bool, root: str) -> object:
    """Builds a tiny, fast ZTEConfig for the synthetic demo path."""
    from zte.config import ZTEConfig

    cfg = ZTEConfig()
    cfg.run_name = 'zte-visualize-synth'
    cfg.dataset.root = root
    cfg.dataset.tasks = ('SR', 'NR')
    cfg.dataset.representation = 'band_power'
    cfg.dataset.include_eye_tracking = include_eye_tracking
    cfg.dataset.cache_dir = str(Path(root).parent / f'cache_{"et" if include_eye_tracking else "eeg"}')
    cfg.model.frontend = 'band_power_mlp'
    cfg.model.embed_dim = 48
    cfg.model.hidden_dim = 64
    cfg.model.n_layers = 2
    cfg.model.n_heads = 4
    cfg.objective.name = 'skipgram'
    cfg.train.epochs = 2
    cfg.train.batch_size = 16
    cfg.train.device = 'cpu'
    cfg.train.precision = 'fp32'
    cfg.train.test_fraction = 0.0
    return cfg


def _embed_quick(cfg: object, ckpt_root: Path) -> tuple[np.ndarray, pd.DataFrame, np.ndarray | None, object]:
    """Builds a dataset, trains 2 epochs, and returns aligned word embeddings + meta.

    Returns:
        tuple: `(word_emb, word_meta, word_band_power, dataset)`, in present-word order -- identical for
            the ET and EEG-only configs, so the two embedding sets line up row-for-row for the toggle.
    """
    from zte.cli.evaluate import collect_embeddings
    from zte.data.dataset import ZuCoDataset
    from zte.inference.embed import ZTEEmbedder
    from zte.training.pipeline import run_training

    cfg.train.ckpt_dir = str(ckpt_root)  # type: ignore[attr-defined]
    dataset = ZuCoDataset(cfg.dataset).build()  # type: ignore[attr-defined]
    run_training(cfg, dataset)  # type: ignore[arg-type]
    embedder = ZTEEmbedder.from_checkpoint(ckpt_root / 'best.pt', dataset)
    word_emb, word_meta, _rw, _se, _sid, _sm, word_bp = collect_embeddings(embedder, dataset)
    return word_emb, word_meta, word_bp, dataset


def _from_synthetic(args: argparse.Namespace) -> dict:
    """Fabricates data and trains two quick models (ET + EEG-only) for the demo."""
    import tempfile

    from zte.data.synthetic import generate_synthetic_zuco

    root = 'res/data/synthetic_zuco_viz'
    generate_synthetic_zuco(root, subjects=('ZAB', 'ZDM', 'ZJN'), tasks=('SR', 'NR'), n_sentences=6)

    tmp = Path(tempfile.mkdtemp(prefix='zte_viz_'))
    _LOG.info('[1/2] Training EEG + eye-tracking model ...')
    et_emb, et_meta, et_bp, dataset = _embed_quick(_quick_config(True, root), tmp / 'et')
    _LOG.info('[2/2] Training EEG-only model ...')
    eeg_emb, _, _, _ = _embed_quick(_quick_config(False, root), tmp / 'eeg')

    et_meta = _add_categories(et_meta, dataset.sentences, root)
    probe = {
        'EEG + eye-tracking': _probe_word_len(et_emb, et_meta),
        'EEG-only': _probe_word_len(eeg_emb, et_meta),
    }
    probe = {k: v for k, v in probe.items() if v is not None} or None
    if len(eeg_emb) != len(et_emb):
        _LOG.warning(
            'ET/EEG-only row counts differ (%d vs %d); dropping the toggle.',
            len(et_emb),
            len(eeg_emb),
        )
        eeg_emb = None
    return {
        'emb': et_emb,
        'meta': et_meta,
        'eeg_only_emb': eeg_emb,
        'band_power': et_bp,
        'probe_scores': probe,
        'emergence': _emergence(et_emb, et_meta, et_bp),
        'title': 'ZTE Thought-Space Explorer - synthetic demo',
        'atlas_title': 'ZTE Neuron Atlas - synthetic demo',
    }


def _atlas_out_path(out: str | Path, kind: str) -> Path:
    """Chooses the atlas output path, avoiding a clash with the explorer when both run."""
    p = Path(out)
    if kind == 'both':
        return p.with_name(p.stem + '_atlas' + (p.suffix or '.html'))
    return p


def _build_atlas(inputs: dict, out: Path, kind: str) -> Path:
    """Computes a neuron report from the collected embeddings and writes the Neuron Atlas."""
    from zte.data.schema import BANDS
    from zte.evaluation.interactive import neuron_atlas_html
    from zte.evaluation.neurons import neuron_report

    band_power = inputs.get('band_power')
    report = neuron_report(
        inputs['emb'],
        inputs['meta'],
        band_power=band_power,
        band_names=BANDS if band_power is not None else None,
    )
    out_path = _atlas_out_path(out, kind)
    written = neuron_atlas_html(report, out_path, title=inputs.get('atlas_title', 'ZTE Neuron Atlas'))
    size_kb = written.stat().st_size / 1024
    _LOG.info(
        'Wrote %s (%.0f KB). Open it in any browser -- no server needed.',
        written.resolve(),
        size_kb,
    )
    return written


def main() -> None:
    """Builds the explorer and/or atlas HTML end-to-end from the command line."""
    args = parse_arguments()
    configure_logging(args.log_level)

    kind = 'atlas' if getattr(args, 'atlas', False) else args.kind

    from zte.evaluation.interactive import thought_space_explorer_html

    inputs = _from_synthetic(args) if args.synthetic else _from_run(args)

    # Emit whichever pages were asked for, then report their paths.
    written: list[Path] = []
    if kind in ('explorer', 'both'):
        out = thought_space_explorer_html(
            inputs['emb'],
            inputs['meta'],
            args.out,
            eeg_only_emb=inputs['eeg_only_emb'],
            probe_scores=inputs['probe_scores'],
            dims=args.dims,
            max_points=args.max_points,
            seed=args.seed,
            title=inputs['title'],
            emergence=inputs.get('emergence'),
        )
        size_kb = out.stat().st_size / 1024
        _LOG.info(
            'Wrote %s (%.0f KB). Open it in any browser -- no server needed.',
            out.resolve(),
            size_kb,
        )
        written.append(out)
    if kind in ('atlas', 'both'):
        written.append(_build_atlas(inputs, args.out, kind))

    for p in written:
        print(str(p.resolve()))


if __name__ == '__main__':
    main()
