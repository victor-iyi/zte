"""`zte-explore` -- score brain-region importance and eye-tracking's contribution, straight from band power."""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zte.cli.support.datasets import synthetic_root
from zte.cli.support.sources import add_data_source_args, add_extract_dir, resolve_data_root
from zte.data.dataset import ZuCoDataset
from zte.data.montage.regions import RegionMap, default_region_map, region_importance
from zte.logging_utils import configure_logging, get_logger
from zte.training.metrics import linear_probe

_LOG = get_logger('cli.explore')


def parse_arguments() -> argparse.Namespace:
    """Defines and parses the `zte-explore` command-line arguments.

    Returns:
        argparse.Namespace: The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description='Explore brain-region and eye-tracking contributions in ZuCo band power.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_source_args(parser, include_bundle=True, include_synthetic=True)
    add_extract_dir(parser)

    parser.add_argument('--tasks', type=str, default='SR,NR')
    parser.add_argument('--out', type=Path, default=Path('res/exploration'))
    parser.add_argument(
        '--montage-csv',
        type=str,
        default=None,
        help='Exact channel->region CSV (else the default anterior-posterior map).',
    )
    parser.add_argument('--method', choices=['mutual_info', 'f_score'], default='mutual_info')
    parser.add_argument(
        '--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace) -> ZuCoDataset:
    """Builds/loads a band-power dataset for exploration."""
    from zte.config import DatasetConfig, MissingConfig

    if args.bundle:
        return ZuCoDataset.load(args.bundle)
    if args.synthetic:
        root = synthetic_root(tuple(args.tasks.split(',')))
    else:
        root = resolve_data_root(args)
    cfg = DatasetConfig(
        root=root,
        tasks=tuple(args.tasks.split(',')),
        representation='band_power',
        include_eye_tracking=True,
        missing=MissingConfig(method='mask_only'),
    )
    return ZuCoDataset(cfg).build()


def _region_targets(ds: ZuCoDataset) -> dict[str, tuple[np.ndarray, str]]:
    """Reading vs cognitive targets for region-importance scoring."""
    w = ds.words
    targets: dict[str, tuple[np.ndarray, str]] = {
        'word_len (reading)': (w['word_len'].to_numpy(), 'regression'),
    }
    freq_col = 'corpus_log_freq' if 'corpus_log_freq' in w else 'log_freq'
    targets['frequency (lexical)'] = (w[freq_col].to_numpy(), 'regression')
    if w['task'].nunique() > 1:
        targets['task (cognitive)'] = (pd.factorize(w['task'])[0], 'classification')
    if w['subject'].nunique() > 1:
        targets['subject (identity)'] = (pd.factorize(w['subject'])[0], 'classification')
    return targets


def eye_tracking_contribution(ds: ZuCoDataset, region_map: RegionMap) -> list[dict[str, Any]]:
    """Probes EEG-only vs eye-tracking-only vs both for reading and cognitive targets.

    Args:
        ds (ZuCoDataset): A built band-power dataset.
        region_map (RegionMap): Channel grouping used to build the compact EEG representation.

    Returns:
        list[dict[str, Any]]: Tidy rows of `target`, `representation` and probe `score`, exposing
            eye-tracking's differential value across target types.
    """
    # Three representations to compare: regional EEG, gaze measures, and their union.
    present = ds.presence if ds.presence is not None else np.ones(len(ds.words), bool)
    region_bp = region_map.reduce(ds.band_power_raw, method='mean')
    eeg = np.nan_to_num(region_bp.reshape(len(region_bp), -1))[present]
    et_cols = [c for c in ds.config.eye_tracking_measures if c in ds.words.columns]
    et = np.nan_to_num(ds.words.loc[present, et_cols].to_numpy(dtype=np.float32))
    both = np.concatenate([eeg, et], axis=1)

    reps = {'EEG-only (regions)': eeg, 'eye-tracking-only': et, 'EEG + eye-tracking': both}

    # One reading target and (where available) one cognitive target, probed for every representation.
    targets: dict[str, tuple[np.ndarray, str]] = {
        'word_len (reading)': (ds.words.loc[present, 'word_len'].to_numpy(), 'regression'),
    }
    if ds.words['task'].nunique() > 1:
        targets['task (cognitive)'] = (
            pd.factorize(ds.words.loc[present, 'task'])[0],
            'classification',
        )
    rows: list[dict[str, Any]] = []
    for tname, (y, task) in targets.items():
        for rname, x in reps.items():
            score = linear_probe(x, y, task=task)['score']
            rows.append({'target': tname, 'representation': rname, 'score': round(float(score), 4)})
    return rows


def run_exploration(
    ds: ZuCoDataset,
    out: str | Path,
    montage_csv: str | None = None,
    method: str = 'mutual_info',
) -> dict[str, Any]:
    """Runs the region + eye-tracking exploration and writes all artifacts.

    Args:
        ds (ZuCoDataset): A built band-power `ZuCoDataset`.
        out (str | Path): Output directory for tables, figures and the report.
        montage_csv (str | None): Optional exact channel->region CSV (else the default map).
        method (str): Per-feature scorer (`'mutual_info'` or `'f_score'`).

    Returns:
        dict[str, Any]: A small summary dict (region map info + written paths).

    Raises:
        ValueError: If the dataset has no band-power tensor.
    """
    out = Path(out)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    if ds.band_power_raw is None:
        raise ValueError(
            'Exploration needs band-power features; rebuild with representation band_power.'
        )

    # Score regions, then quantify eye-tracking's contribution over the same channel grouping.
    # This is the LAST stage of a multi-hour run, so a missing or coordinates-only montage degrades to
    # the approximate cap (as `evaluation.report._load_region_map` already does) instead of discarding
    # the whole run -- the montage lives outside the run directory and may not survive a new VM.
    region_map = default_region_map(ds.band_power_raw.shape[-1])
    if montage_csv:
        if Path(montage_csv).is_file():
            try:
                region_map = RegionMap.from_csv(montage_csv, ds.band_power_raw.shape[-1])
            except (OSError, ValueError, KeyError) as exc:
                _LOG.warning(
                    'Could not load montage %s: %r; using approximate regions.', montage_csv, exc
                )
        else:
            _LOG.warning('Montage %s not found; using approximate regions.', montage_csv)
    region_rows = region_importance(
        ds.band_power_raw,
        _region_targets(ds),
        region_map=region_map,
        presence=ds.presence,
        method=method,
    )
    et_rows = eye_tracking_contribution(ds, region_map)

    # Write the tables, figures and report.
    pd.DataFrame(region_rows).to_csv(out / 'region_importance.csv', index=False)
    pd.DataFrame(et_rows).to_csv(out / 'eye_tracking_contribution.csv', index=False)
    figures = _render_figures(region_rows, et_rows, out / 'figures')
    (out / 'report.md').write_text(
        _render_report(ds, region_map, region_rows, et_rows, figures, out), encoding='utf-8'
    )
    _LOG.info('Exploration written to %s (%d figures)', out, len(figures))
    return {
        'region_map_approximate': region_map.approximate,
        'region_sizes': region_map.region_sizes(),
        'region_importance': region_rows,
        'eye_tracking_contribution': et_rows,
    }


def main() -> None:
    """Runs the exploration end-to-end from the command line."""
    args = parse_arguments()
    configure_logging(args.log_level)
    ds = _load_dataset(args)
    summary = run_exploration(ds, args.out, montage_csv=args.montage_csv, method=args.method)
    print(
        json.dumps(
            {
                'region_map_approximate': summary['region_map_approximate'],
                'region_sizes': summary['region_sizes'],
            },
            indent=2,
        )
    )


def _render_figures(
    region_rows: list[dict[str, Any]], et_rows: list[dict[str, Any]], fig_dir: Path
) -> list[Path]:
    """Renders the region-importance heatmap and eye-tracking probe bars."""
    import matplotlib.pyplot as plt

    from zte.evaluation import plots as P

    written: list[Path] = []
    try:
        fig = P.region_importance_heatmap(region_rows)
        path = fig_dir / 'region_importance.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        written.append(path)
    except (ValueError, KeyError) as exc:  # pragma: no cover
        _LOG.warning('Region heatmap skipped: %r', exc)
    try:
        fig = P.bar_probe_comparison(
            [
                {**r, 'linear_score': r['score'], 'knn_score': r['score'], 'baseline': 0.0}
                for r in et_rows
            ],
            'linear_score',
            title='Eye-tracking contribution by target (linear probe)',
        )
        path = fig_dir / 'eye_tracking_contribution.png'
        fig.savefig(path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        written.append(path)
    except (ValueError, KeyError) as exc:  # pragma: no cover
        _LOG.warning('Eye-tracking figure skipped: %r', exc)
    return written


def _render_report(
    ds: ZuCoDataset,
    region_map: RegionMap,
    region_rows: list[dict[str, Any]],
    et_rows: list[dict[str, Any]],
    figures: list[Path],
    out: Path,
) -> str:
    """Builds the exploration Markdown report."""
    region_frame = pd.DataFrame(region_rows).pivot(
        index='region', columns='target', values='importance'
    )
    et_frame = pd.DataFrame(et_rows).pivot(index='representation', columns='target', values='score')
    lines = [
        '# ZTE exploration -- brain regions & eye-tracking',
        '',
        f'Words: **{len(ds.words)}** | region map: '
        f'**{"approximate default" if region_map.approximate else "montage-derived"}** '
        f'({region_map.n_regions} regions)',
        '',
        '## Region importance (share of decodable information)',
        '',
        '`reading` targets vs `cognitive` targets; higher = that region carries more '
        'of the attribute.',
        '',
        '| region | ' + ' | '.join(str(c) for c in region_frame.columns) + ' |',
        '| --- |' + ' --- |' * len(region_frame.columns),
    ]
    for name, row in region_frame.iterrows():
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.2f}' for v in row.to_numpy()) + ' |')
    lines += [
        '',
        '## Eye-tracking contribution (linear-probe score)',
        '',
        'Eye-tracking should help *reading* targets much more than *cognitive* ones -- '
        'the reason ZTE makes it optional (`include_eye_tracking`).',
        '',
        '| representation | ' + ' | '.join(str(c) for c in et_frame.columns) + ' |',
        '| --- |' + ' --- |' * len(et_frame.columns),
    ]
    for name, row in et_frame.iterrows():
        lines.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in row.to_numpy()) + ' |')
    lines += ['', '## Figures', '']
    lines += [f'![{p.stem}]({p.relative_to(out).as_posix()})' for p in figures]
    lines.append('')
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
