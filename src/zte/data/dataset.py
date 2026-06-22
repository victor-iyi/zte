"""The tunable :class:`ZuCoDataset` -- load, process, analyse, select, split, save.

This is the high-level entry point most users touch. It scans ZuCo `.mat` files (or a synthetic tree),
flattens them into a word/sentence table plus aligned band-power and raw EEG tensors,
applies a configurable missing-value strategy and normaliser, and exposes everything needed downstream: analysis summaries,
supervised feature selection, leakage-aware splits (including leave-one-subject-out), a cached on-disk bundle and a bridge to PyTorch.

Lifecycle::

    ds = ZuCoDataset(config).build()      # load (+cache) and process
    ds.analyze()                          # summary statistics
    ds.select_features(target='log_freq') # rank channels x bands
    splits = ds.split('by_subject_loso')  # indices per split
    torch_ds = ds.to_torch(split=splits['train'])
    ds.save('res/bundle')                 # round-trips everything
"""

# pylint: disable=import-outside-toplevel
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from zte.config import DatasetConfig
from zte.data import mat_loader
from zte.data.features import (
    FeatureSelector,
    SelectionMethod,
    SelectionResult,
    channel_mean_features,
    flat_feature_names,
    flatten_band_power,
)
from zte.data.missing import MissingValueImputer
from zte.data.schema import N_CHANNELS
from zte.data.transforms import FeatureNormalizer, bandpass_filter
from zte.logging_utils import get_logger, progress

_LOG = get_logger('data.dataset')


def _word_freq_proxy(word: str) -> float:
    """Dependency-free word-frequency proxy in `(0, 1]` (short words score high).

    Args:
        word (str): Surface word form.

    Returns:
        float: A frequency-like value consistent with the synthetic generator, so models
          behave the same on synthetic and (after a real frequency table is supplied) real data.

    """
    return float(np.clip(1.0 / (1.0 + 0.35 * len(word.strip('.,;:'))), 0.05, 1.0))


class ZuCoDataset:
    """A configurable, cache-backed view over ZuCo EEG/eye-tracking data.

    Attributes:
        config: The :class:`~zte.config.DatasetConfig` controlling everything.
        words (pd.DataFrame): Per-word metadata/scalar table (populated after :meth:`build`).
        sentences (pd.DataFrame): Per-sentence metadata table.
        band_power_raw (np.ndarray | None): `(N, F, C)` band power with `NaN` for omissions.
        features (np.ndarray | None): `(N, F*C)` imputed + normalised band-power matrix.
        presence (np.ndarray | None): `(N,)` boolean mask, `False` for omitted words.
        raw_eeg (np.ndarray | None): `(N, C, T)` raw EEG windows, or `None`.
        feature_names (list[str]): Names for the `F*C` flattened band-power columns.
        bp_feature_names (list[str]): Names for the `F` `(measure, band)` features.
        normalizer (FeatureNormalizer | None): The fitted :class:`~zte.data.transforms.FeatureNormalizer`.
    """

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Initialises an empty dataset.

        Args:
            config (DatasetConfig): Dataset configuration; defaults to :class:`DatasetConfig`.

        """
        self.config = config or DatasetConfig()
        self.words: pd.DataFrame = pd.DataFrame()
        self.sentences: pd.DataFrame = pd.DataFrame()
        self.band_power_raw: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.presence: np.ndarray | None = None
        self.raw_eeg: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.bp_feature_names: list[str] = []
        self.normalizer: FeatureNormalizer | None = None
        self._groups: list[tuple[tuple[str, str, int], np.ndarray]] | None = None

    # -- construction ------------------------------------------------------- #

    def discover_files(self) -> list[Path]:
        """Finds ZuCo `.mat` files under `config.root` matching the filters.

        Returns:
            list[Path]: Sorted list of matching `.mat` paths (filtered by task and, if set, subject).

        Raises:
            FileNotFoundError: If the root does not exist or contains no matches.

        """
        root = Path(self.config.root)
        if not root.exists():
            raise FileNotFoundError(f'Dataset root does not exist: {root}')
        files = sorted(root.rglob('*.mat'))
        keep: list[Path] = []
        for path in files:
            subject, task = mat_loader.parse_subject_task(path)
            if task not in self.config.tasks:
                continue
            if self.config.subjects is not None and subject not in self.config.subjects:
                continue
            keep.append(path)
        if not keep:
            raise FileNotFoundError(
                f'No .mat files under {root} matched tasks={self.config.tasks} '
                f'subjects={self.config.subjects}.'
            )
        return keep

    def build(self, force: bool = False, show_progress: bool = True) -> ZuCoDataset:
        """Loads (using cache when possible) and fully processes the dataset.

        Args:
            force: Ignore any existing cache and rebuild from ``.mat`` files.
            show_progress: Show per-file progress bars.

        Returns:
            ZuCoDataset: `self`, fully populated.

        """
        cache_dir = Path(self.config.cache_dir) / self._cache_key()
        if not force and (cache_dir / 'meta.json').is_file():
            _LOG.info('Loading processed dataset from cache: %s', cache_dir)
            return self.load(cache_dir, into=self)

        self._load_mat(show_progress=show_progress)
        self._process()
        self.save(cache_dir)
        return self

    def _load_mat(self, show_progress: bool = True) -> None:
        """Scans and flattens all matching ``.mat`` files into raw containers."""
        files = self.discover_files()
        cfg = self.config
        load_bp = cfg.representation in {'band_power', 'both'}
        load_raw = cfg.representation in {'raw', 'both'}

        sent_rows: list[dict[str, Any]] = []
        word_rows: list[dict[str, Any]] = []
        bp_chunks: list[np.ndarray] = []
        raw_chunks: list[np.ndarray] = []

        for path in progress(files, description='loading .mat', disable=not show_progress):
            extract = mat_loader.extract_file(
                path,
                measures=cfg.band_power_measures,
                bands=cfg.bands,
                load_band_power=load_bp,
                load_raw=load_raw,
                raw_field=cfg.raw_field,
                raw_window=cfg.raw_window,
            )
            sent_rows.extend(extract.sentence_rows)
            word_rows.extend(extract.word_rows)
            if extract.band_power is not None:
                bp_chunks.append(extract.band_power)
            if extract.raw_eeg is not None:
                raw_chunks.append(extract.raw_eeg)
            self.bp_feature_names = extract.bp_feature_names or self.bp_feature_names

        self.words = pd.DataFrame(word_rows)
        self.sentences = pd.DataFrame(sent_rows)
        self.band_power_raw = np.concatenate(bp_chunks, axis=0) if bp_chunks else None
        self.raw_eeg = np.concatenate(raw_chunks, axis=0) if raw_chunks else None
        _LOG.info(
            'Loaded %d words across %d sentences (%d files).',
            len(self.words),
            len(self.sentences),
            len(files),
        )

    def _process(self) -> None:
        """Adds linguistic features, applies length filters, imputes and normalises."""
        self._add_linguistic_features()
        self._apply_length_filters()

        if self.band_power_raw is not None:
            flat = flatten_band_power(self.band_power_raw)
            imputer = MissingValueImputer(self.config.missing)
            group_ids = self.words['sentence_uid'].to_numpy()
            imputed, presence = imputer.fit_transform(flat, group_ids=group_ids)
            self.presence = presence
            self.normalizer = FeatureNormalizer(self.config.normalize)
            # Fit normaliser on present tokens only to avoid omission contamination.
            self.normalizer.fit(imputed[presence] if presence.any() else imputed)
            self.features = self.normalizer.transform(imputed)
            self.feature_names = flat_feature_names(self.bp_feature_names, N_CHANNELS)
            self._attach_channel_mean_columns()
        else:
            self.presence = self.words['has_word_eeg'].to_numpy(dtype=bool)

        if self.raw_eeg is not None and self.config.bandpass is not None:
            low, high = self.config.bandpass
            self.raw_eeg = np.stack([bandpass_filter(epoch, low, high) for epoch in self.raw_eeg])
        if self.config.missing.method == 'drop':
            self._drop_missing_rows()

    def _add_linguistic_features(self) -> None:
        """Adds word length, frequency proxy, relative position and omission flags."""
        w = self.words
        w['word_len'] = w['word'].str.len().fillna(0).astype(int)
        w['freq'] = w['word'].map(_word_freq_proxy)
        w['log_freq'] = np.log10(w['freq'].astype(float))
        w['is_omitted'] = w['FFD'].isna().astype(int)
        w['sentence_uid'] = (
            w['subject'].astype(str)
            + '|'
            + w['task'].astype(str)
            + '|'
            + w['sentence_idx'].astype(str)
        )
        max_idx = w.groupby('sentence_uid')['word_idx'].transform('max')
        w['rel_pos'] = (w['word_idx'] / (max_idx + 1)).astype(float)

    def _apply_length_filters(self) -> None:
        """Drops words belonging to sentences outside the configured length band."""
        cfg = self.config
        counts = self.words.groupby('sentence_uid')['word_idx'].transform('count')
        keep = counts >= cfg.min_words
        if cfg.max_words is not None:
            keep &= counts <= cfg.max_words
        if keep.all():
            return
        keep_np = keep.to_numpy()
        self._subset_rows(keep_np)
        _LOG.info('Length filter kept %d/%d words.', int(keep_np.sum()), len(keep_np))

    def _attach_channel_mean_columns(self) -> None:
        """Adds compact ``<measure>_<band>_chmean`` columns for analysis/plots."""
        if self.band_power_raw is None:
            return
        means = channel_mean_features(self.band_power_raw)
        for j, name in enumerate(self.bp_feature_names):
            self.words[f'{name}_chmean'] = means[:, j]

    def _drop_missing_rows(self) -> None:
        """Physically removes omitted-word rows when ``missing.method='drop'``."""
        if self.presence is None:
            return
        self._subset_rows(self.presence)
        self.presence = np.ones(len(self.words), dtype=bool)

    def _subset_rows(self, keep: np.ndarray) -> None:
        """Applies a boolean row filter consistently across all aligned stores."""
        self.words = self.words.loc[keep].reset_index(drop=True)
        if self.band_power_raw is not None:
            self.band_power_raw = self.band_power_raw[keep]
        if self.features is not None:
            self.features = self.features[keep]
        if self.raw_eeg is not None:
            self.raw_eeg = self.raw_eeg[keep]
        if self.presence is not None:
            self.presence = self.presence[keep]
        self._groups = None

    # -- grouping & splits -------------------------------------------------- #

    @property
    def groups(self) -> list[tuple[tuple[str, str, int], np.ndarray]]:
        """Returns a list of tuples with the format `[((subject, task, sentence_idx), word_row_indices), ...]`.

        Word indices within each group are ordered by `word_idx` so sequence
        models receive words in reading order.

        Returns:
            list[tuple[tuple[str, str, int], np.ndarray]]: A list of tuples with the format `[((subject, task, sentence_idx), word_row_indices), ...]`.

        """
        if self._groups is None:
            self._groups = []
            ordered = self.words.sort_values(['sentence_uid', 'word_idx'])
            for _uid, block in ordered.groupby('sentence_uid', sort=True):
                subject = str(block['subject'].iloc[0])
                task = str(block['task'].iloc[0])
                s_idx = int(block['sentence_idx'].iloc[0])
                self._groups.append(((subject, task, s_idx), block.index.to_numpy()))
        return self._groups

    def split(
        self,
        strategy: Literal['random', 'by_sentence', 'by_subject_loso', 'by_task'] | None = None,
        val_fraction: float = 0.1,
        holdout_subject: str | None = None,
        holdout_task: str | None = None,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Produces leakage-aware train/val (or train/test) row-index splits.

        Args:
            strategy: `'random'` (per-word), `'by_sentence'` (whole sentences
                kept together), `'by_subject_loso'` (hold out one subject) or
                `'by_task'` (hold out one task). Defaults to `'by_sentence'`.
            val_fraction (float): Held-out fraction for `random`/`by_sentence`.
            holdout_subject (str): Subject to hold out for LOSO (else the last subject).
            holdout_task (str): Task to hold out for `by_task` (else the last task).
            seed (int): RNG seed for the randomised strategies.

        Returns:
            dict[str, np.ndarray]: A mapping with `'train'` and `'val'` row-index arrays.

        """
        strategy = strategy or 'by_sentence'
        rng = np.random.default_rng(seed)
        n = len(self.words)
        idx = np.arange(n)

        if strategy == 'random':
            perm = rng.permutation(n)
            cut = int(n * (1 - val_fraction))
            return {'train': np.sort(perm[:cut]), 'val': np.sort(perm[cut:])}

        if strategy == 'by_sentence':
            uids = self.words['sentence_uid'].unique()
            perm = rng.permutation(len(uids))
            cut = int(len(uids) * (1 - val_fraction))
            train_uids = set(uids[perm[:cut]])
            in_train = self.words['sentence_uid'].isin(train_uids).to_numpy()
            return {'train': idx[in_train], 'val': idx[~in_train]}

        if strategy == 'by_subject_loso':
            subjects = sorted(self.words['subject'].unique())
            hold = holdout_subject or subjects[-1]
            in_val = (self.words['subject'] == hold).to_numpy()
            return {'train': idx[~in_val], 'val': idx[in_val]}

        # by_task
        tasks = sorted(self.words['task'].unique())
        hold = holdout_task or tasks[-1]
        in_val = (self.words['task'] == hold).to_numpy()
        return {'train': idx[~in_val], 'val': idx[in_val]}

    # -- analysis & selection ---------------------------------------------- #

    def analyze(self) -> dict[str, Any]:
        """Computes a compact summary of the dataset (counts, missingness, stats).

        Returns:
            A nested dict with subject/task counts, per-measure missing rates,
            omission statistics and band-power presence -- safe to JSON-dump or
            log.
        """
        w = self.words
        from zte.data.schema import ET_MEASURES

        summary: dict[str, Any] = {
            'n_words': int(len(w)),
            'n_sentences': int(w['sentence_uid'].nunique()),
            'n_subjects': int(w['subject'].nunique()),
            'subjects': sorted(w['subject'].unique().tolist()),
            'tasks': sorted(w['task'].unique().tolist()),
            'words_per_task': w.groupby('task')['word_idx'].count().to_dict(),
            'omission_rate_overall': float(w['is_omitted'].mean()),
            'omission_rate_by_subject': w.groupby('subject')['is_omitted']
            .mean()
            .round(4)
            .to_dict(),
            'missing_rate_by_measure': {
                m: float(w[m].isna().mean()) for m in ET_MEASURES if m in w
            },
        }
        if self.presence is not None:
            summary['word_eeg_present_fraction'] = float(self.presence.mean())
        return summary

    def select_features(
        self,
        target: str = 'log_freq',
        method: SelectionMethod = 'mutual_info',
        k: int | None = 64,
        present_only: bool = True,
    ) -> SelectionResult:
        """Ranks flattened band-power features by importance for a target column.

        Args:
            target: A column in :attr:`words` to predict (e.g. ``'log_freq'`` or
                ``'is_omitted'``).
            method: Selection method (see :class:`FeatureSelector`).
            k: Number of top features to keep.
            present_only: Restrict scoring to present (non-omitted) tokens.

        Returns:
            A :class:`SelectionResult` with selected indices, scores and names.

        Raises:
            RuntimeError: If band-power features were not built.
            KeyError: If ``target`` is not a column of :attr:`words`.
        """
        if self.features is None:
            raise RuntimeError('No band-power features; build with representation band_power/both.')
        if target not in self.words:
            raise KeyError(f'Target {target!r} not in words columns.')
        task = 'classification' if self.words[target].nunique() <= 2 else 'regression'
        selector = FeatureSelector(method=method, k=k, task=task)
        mask = self.presence if (present_only and self.presence is not None) else None
        return selector.select(
            self.features,
            self.words[target].to_numpy(),
            names=self.feature_names,
            sample_mask=mask,
        )

    # -- torch bridge ------------------------------------------------------- #

    def to_torch(self, split: np.ndarray | None = None, **kwargs: Any) -> Any:
        """Builds a PyTorch dataset over (a subset of) these samples.

        Args:
            split: Optional row indices selecting a split (e.g. from :meth:`split`).
            **kwargs: Forwarded to :class:`~zte.data.torch_dataset.ZuCoTorchDataset`.

        Returns:
            A :class:`~zte.data.torch_dataset.ZuCoTorchDataset`.
        """
        from zte.data.torch_dataset import ZuCoTorchDataset

        return ZuCoTorchDataset(self, indices=split, **kwargs)

    # -- persistence -------------------------------------------------------- #

    def _cache_key(self) -> str:
        """Builds a short, deterministic cache subfolder name from the config."""
        cfg = self.config
        parts = [
            '-'.join(cfg.tasks),
            cfg.representation,
            cfg.granularity,
            cfg.missing.method,
            cfg.normalize,
            f'rw{cfg.raw_window}',
        ]
        return '_'.join(parts)

    def save(self, path: str | Path) -> Path:
        """Saves the full processed dataset as a self-contained directory bundle.

        The bundle contains ``arrays.npz`` (tensors + presence), ``words.pkl`` and
        ``sentences.pkl`` (tables), and ``meta.json`` (config, feature names and
        the fitted normaliser state) so :meth:`load` reproduces the object exactly.

        Args:
            path: Destination directory (created if needed).

        Returns:
            The bundle directory path.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        if self.band_power_raw is not None:
            arrays['band_power_raw'] = self.band_power_raw
        if self.features is not None:
            arrays['features'] = self.features
        if self.presence is not None:
            arrays['presence'] = self.presence
        if self.raw_eeg is not None:
            arrays['raw_eeg'] = self.raw_eeg
        np.savez_compressed(out / 'arrays.npz', **arrays)
        self.words.to_pickle(out / 'words.pkl')
        self.sentences.to_pickle(out / 'sentences.pkl')
        meta = {
            'config': asdict(self.config),
            'feature_names': self.feature_names,
            'bp_feature_names': self.bp_feature_names,
            'normalizer': None if self.normalizer is None else self.normalizer.state,
        }
        (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        _LOG.info('Saved dataset bundle to %s', out)
        return out

    @classmethod
    def load(cls, path: str | Path, into: ZuCoDataset | None = None) -> ZuCoDataset:
        """Loads a dataset bundle previously written by :meth:`save`.

        Args:
            path: The bundle directory.
            into: Optionally populate an existing instance (used by the cache
                path in :meth:`build`).

        Returns:
            The populated dataset.
        """
        src = Path(path)
        meta = json.loads((src / 'meta.json').read_text(encoding='utf-8'))
        config = DatasetConfig(**_coerce_dataset_meta(meta['config']))
        ds = into or cls(config)
        ds.config = config
        ds.words = pd.read_pickle(src / 'words.pkl')
        ds.sentences = pd.read_pickle(src / 'sentences.pkl')
        with np.load(src / 'arrays.npz') as arrays:
            ds.band_power_raw = arrays['band_power_raw'] if 'band_power_raw' in arrays else None
            ds.features = arrays['features'] if 'features' in arrays else None
            ds.presence = arrays['presence'] if 'presence' in arrays else None
            ds.raw_eeg = arrays['raw_eeg'] if 'raw_eeg' in arrays else None
        ds.feature_names = meta['feature_names']
        ds.bp_feature_names = meta['bp_feature_names']
        if meta.get('normalizer'):
            ds.normalizer = FeatureNormalizer.from_state(meta['normalizer'])
        ds._groups = None  # pylint: disable=protected-access
        return ds

    # -- remote ------------------------------------------------------------- #

    def save_to_drive(self, remote_dir: str, local_tmp: str | Path = 'res/.drive_tmp') -> str:
        """Saves the bundle locally then uploads it to Google Drive.

        Args:
            remote_dir: Destination Drive folder id or path (see
                :mod:`zte.data.remote`).
            local_tmp: Local staging directory for the bundle.

        Returns:
            The remote location string returned by the uploader.
        """
        from zte.data.remote import upload_directory

        local = self.save(local_tmp)
        return upload_directory(local, remote_dir)

    @classmethod
    def from_drive(cls, remote_spec: str, local_tmp: str | Path = 'res/.drive_dl') -> ZuCoDataset:
        """Downloads a bundle from Google Drive and loads it.

        Args:
            remote_spec: A Drive file/folder id or shareable URL.
            local_tmp: Local directory to download into.

        Returns:
            The loaded dataset.
        """
        from zte.data.remote import download_to_dir

        local = download_to_dir(remote_spec, local_tmp)
        return cls.load(local)

    def __len__(self) -> int:
        """Returns the number of word rows currently held."""
        return len(self.words)

    def __repr__(self) -> str:
        """Returns a concise developer-facing summary."""
        return (
            f'ZuCoDataset(words={len(self.words)}, '
            f'sentences={self.words["sentence_uid"].nunique() if len(self.words) else 0}, '
            f'representation={self.config.representation!r}, '
            f'features={None if self.features is None else self.features.shape})'
        )


def _coerce_dataset_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerces a serialised dataset-config dict back into constructor kwargs."""
    from zte.config import MissingConfig

    out = dict(meta)
    for key in ('tasks', 'bands', 'band_power_measures', 'subjects'):
        if isinstance(out.get(key), list):
            out[key] = tuple(out[key])
    if isinstance(out.get('bandpass'), list):
        out['bandpass'] = tuple(out['bandpass'])
    if isinstance(out.get('missing'), dict):
        out['missing'] = MissingConfig(**out['missing'])
    return out
