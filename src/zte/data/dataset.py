"""The tunable `ZuCoDataset`, whose lifecycle runs `build` -> `analyze`/`select_features` -> `split` -> `to_torch`."""

from __future__ import annotations

import json
import pickle
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import pandas as pd

from zte.config import DatasetConfig
from zte.data.cache import BundleStore
from zte.data.features.alignment import RawSubjectAligner
from zte.data.features.features import (
    FeatureSelector,
    SelectionMethod,
    SelectionResult,
    channel_mean_features,
    flat_feature_names,
    flatten_band_power,
)
from zte.data.features.missing import MissingValueImputer
from zte.data.features.transforms import FeatureNormalizer, bandpass_filter, sanitize_raw_windows
from zte.data.io import mat_loader
from zte.data.schema import N_CHANNELS
from zte.logging_utils import get_logger, progress

_LOG = get_logger('data.dataset')

#: Raw EEG lives beside `arrays.npz`, uncompressed, so a bundle can be memory-mapped instead of inflated.
RAW_ARRAY_FILE: str = 'raw_eeg.npy'


# The only settings `.mat` extraction depends on -- everything else is post-processing, which is why
# one extraction is reusable across configs that differ solely in how they process it.
_EXTRACT_FIELDS: Final[tuple[str, ...]] = (
    'tasks',
    'subjects',
    'granularity',
    'representation',
    'band_power_measures',
    'bands',
    'raw_field',
    'raw_window',
)


def _jsonable(value: Any) -> Any:
    """Renders a config value in a stable, JSON-serialisable form for hashing."""
    if isinstance(value, tuple | list):
        return [_jsonable(v) for v in value]
    return value


def _extract_raw_member(bundle: Path) -> bool:
    """Upgrades a pre-mmap bundle in place by streaming `raw_eeg` out of `arrays.npz` into its own file.

    A member of an `.npz` *is* a `.npy`, so copying the decompressed stream byte-for-byte yields a valid,
    mappable file without ever holding the ~24 GB array. Cheaper than re-deriving the bundle, and it works
    on a runtime far too small to load the old layout at all.
    """
    import shutil
    import zipfile

    archive = bundle / 'arrays.npz'
    if not archive.is_file():
        return False

    target = bundle / RAW_ARRAY_FILE
    partial = target.with_suffix('.npy.partial')
    try:
        with zipfile.ZipFile(archive) as zf:
            name = next((n for n in zf.namelist() if n in {'raw_eeg.npy', 'raw_eeg'}), None)
            if name is None:
                return False
            _LOG.info(
                'Upgrading %s to the memory-mapped layout (streaming, no full load) ...',
                bundle.name,
            )
            with zf.open(name) as source, partial.open('wb') as sink:
                shutil.copyfileobj(source, sink, length=32 << 20)
        partial.replace(target)
    except (OSError, zipfile.BadZipFile) as exc:
        partial.unlink(missing_ok=True)
        _LOG.warning('Could not upgrade %s (%r); falling back to the in-memory path.', bundle.name, exc)
        return False
    return True


def _word_freq_proxy(word: str) -> float:
    """Dependency-free word-frequency proxy in `(0, 1]` where short words score high.

    Returns:
        float: A frequency-like value matching the synthetic generator, so models behave the same on both.
    """
    return float(np.clip(1.0 / (1.0 + 0.35 * len(word.strip('.,;:'))), 0.05, 1.0))


class ZuCoDataset:
    """A configurable, cache-backed view over ZuCo EEG/eye-tracking data.

    Attributes:
        config (DatasetConfig): The configuration controlling everything.
        words (pd.DataFrame): Per-word metadata/scalar table (populated after `build`).
        sentences (pd.DataFrame): Per-sentence metadata table.
        band_power_raw (np.ndarray | None): `(n_words, n_bp_features, n_channels)` band power with `NaN` for omissions.
        features (np.ndarray | None): `(n_words, n_features)` imputed and normalised band-power matrix.
        presence (np.ndarray | None): `(n_words,)` boolean mask, `False` for omitted words.
        raw_eeg (np.ndarray | None): `(n_words, n_channels, time_steps)` raw EEG windows, or `None`.
        feature_names (list[str]): Names for the `n_bp_features * n_channels` flattened band-power columns.
        bp_feature_names (list[str]): Names for the `n_bp_features` `(measure, band)` features.
        normalizer (FeatureNormalizer | None): The fitted `FeatureNormalizer`.
    """

    def __init__(self, config: DatasetConfig | None = None) -> None:
        """Initialises an empty dataset.

        Args:
            config (DatasetConfig | None): Dataset configuration; `None` uses the defaults.
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
        self.aligner: RawSubjectAligner | None = None
        self._aligned = False
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
                f'No .mat files under {root} matched tasks={self.config.tasks} subjects={self.config.subjects}.'
            )
        return keep

    def build(self, force: bool = False, show_progress: bool = True) -> ZuCoDataset:
        """Loads (using cache when possible) and fully processes the dataset.

        Args:
            force (bool): Ignore any existing cache and rebuild from `.mat` files.
            show_progress (bool): Show per-file progress bars.

        Returns:
            ZuCoDataset: The fully populated dataset.

        Raises:
            NotImplementedError: If a reserved option is requested; only `granularity='word'` and `cache_format='npz'`
                are implemented.
        """
        if self.config.granularity != 'word':
            raise NotImplementedError(
                "Only granularity='word' is implemented. For sentence-level "
                "embeddings, train on words and pool via ZTEEmbedder(level='sentence')."
            )
        if self.config.cache_format != 'npz':
            raise NotImplementedError(
                f"cache_format={self.config.cache_format!r} is reserved; only 'npz' is currently implemented."
            )
        store = BundleStore.create(self.config.cache_dir, self.config.cache_remote)
        key, extract_key = self._cache_key(), self._extract_key()

        # Level 2: the finished bundle for exactly this config. An unreadable entry costs a rebuild,
        # never the run: it is cleared and the build falls through to processing, checkpoint-style.
        if not force:
            hit = store.find(key)
            if hit is not None:
                _LOG.info('Loading processed dataset from cache: %s', hit)
                try:
                    self.load(hit, into=self)
                except (OSError, KeyError, ValueError, EOFError, pickle.UnpicklingError) as exc:
                    _LOG.warning('Cache entry %s is unreadable (%r); discarding it and rebuilding.', hit, exc)
                    shutil.rmtree(hit, ignore_errors=True)
                else:
                    store.publish(key)  # a local-only hit still gets persisted
                    return self

        # Level 1: the `.mat` extraction, which depends on far fewer settings than the processing does,
        # so a config that only changes normalisation/imputation/filters skips the expensive parse.
        extract_hit = None if force else store.find(extract_key, kind='extract')
        loaded_extract = False
        if extract_hit is not None:
            _LOG.info('Reusing cached .mat extraction: %s', extract_hit)
            try:
                self._load_extract(extract_hit)
                loaded_extract = True
            except (OSError, KeyError, ValueError, EOFError, pickle.UnpicklingError) as exc:
                _LOG.warning('Cached extraction %s is unreadable (%r); discarding it and re-parsing.', extract_hit, exc)
                shutil.rmtree(extract_hit, ignore_errors=True)
        if not loaded_extract:
            self._load_mat(show_progress=show_progress)
            if self.config.cache_extracts:
                self._save_extract(store.reserve(extract_key, kind='extract'))
                store.publish(extract_key, kind='extract')

        self._process()
        self.save(store.reserve(key))
        store.publish(key)
        return self

    def _load_mat(self, show_progress: bool = True) -> None:
        """Scans and flattens all matching `.mat` files into raw containers."""
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
        self._attach_categories()
        self._apply_length_filters()

        if self.band_power_raw is not None:
            flat = flatten_band_power(self.band_power_raw)
            imputer = MissingValueImputer(self.config.missing)
            group_ids = self.words['sentence_uid'].to_numpy()
            imputed, presence = imputer.fit_transform(flat, group_ids=group_ids)
            self.presence = presence
            names = flat_feature_names(self.bp_feature_names, N_CHANNELS)
            combined, names = self._maybe_add_eye_tracking(imputed, names, presence)
            self.normalizer = FeatureNormalizer(self.config.normalize)
            subjects = self.words['subject'].to_numpy()
            # Fit normaliser on present tokens only to avoid omission contamination.
            fit_mask = presence if presence.any() else np.ones(len(combined), dtype=bool)
            self.normalizer.fit(combined[fit_mask], subjects=subjects[fit_mask])
            self.features = self.normalizer.transform(combined, subjects=subjects)
            self.feature_names = names
            self._attach_channel_mean_columns()
        else:
            self.presence = self.words['has_word_eeg'].to_numpy(dtype=bool)

        if self.raw_eeg is not None and self.config.bandpass is not None:
            low, high = self.config.bandpass
            self.raw_eeg = np.stack([bandpass_filter(epoch, low, high) for epoch in self.raw_eeg])
        if self.raw_eeg is not None:
            # The raw counterpart of the band-power imputation above, so every consumer sees model-safe windows.
            self.raw_eeg = sanitize_raw_windows(self.raw_eeg)

        if self.config.missing.method == 'drop' or not self.config.include_omitted:
            self._drop_missing_rows()

    def align_raw(self, train_indices: np.ndarray | None = None) -> None:
        """Whitens raw windows per subject and computes each subject's signature.

        Applied after the cached bundle loads rather than inside `_process`, so enabling alignment never invalidates
        a prepared bundle. Idempotent: a second call is a no-op.

        Args:
            train_indices (np.ndarray | None): Training rows, used only when `config.raw_align_fit == 'train'`.
        """
        if self._aligned or self.raw_eeg is None or self.config.raw_align == 'none':
            return

        subjects = self.words['subject'].to_numpy()
        present = self.presence if self.presence is not None else np.ones(len(self.words), dtype=bool)

        # `train` restricts the fit to training rows; `all` lets every subject supply their own map.
        fit_mask = present.copy()
        if self.config.raw_align_fit == 'train' and train_indices is not None:
            train_mask = np.zeros(len(self.words), dtype=bool)
            keep = np.asarray(train_indices, dtype=int)
            train_mask[keep[(keep >= 0) & (keep < len(self.words))]] = True
            fit_mask &= train_mask

        if not fit_mask.any():
            _LOG.warning('align_raw: no usable rows to fit on; skipping alignment.')
            return

        self.aligner = RawSubjectAligner(match_amplitude=self.config.raw_align_amplitude).fit(
            self.raw_eeg, subjects, present=fit_mask, region_index=self._region_index()
        )
        self._write_aligned(subjects)
        _LOG.info(
            'Euclidean-aligned raw windows for %d subjects (fit=%s, signature dim %d).',
            len(self.aligner.references),
            self.config.raw_align_fit,
            self.aligner.signature_dim,
        )

    def set_normalizer_state(self, state: dict[str, Any] | None) -> None:
        """Installs a previously fitted feature normaliser and re-transforms every row through it.

        A frozen encoder only behaves as measured if its inputs arrive on the scale it was trained on, so a decoder
        stage restores the source run's statistics instead of fitting its own. Getting this wrong is silent: the model
        loads, trains and simply underperforms.

        Args:
            state (dict[str, Any] | None): A `FeatureNormalizer.state`; `None` leaves the current fit alone.
        """
        if state is None:
            return
        if self.features is None or self.normalizer is None:
            _LOG.warning('set_normalizer_state: this dataset holds no band-power features.')
            return
        subjects = self.words['subject'].to_numpy()
        combined = self.normalizer.inverse_transform(self.features, subjects=subjects).copy()
        normalizer = FeatureNormalizer.from_state(state)
        self.features = normalizer.transform(combined, subjects=subjects)
        self.normalizer = normalizer
        _LOG.info('Installed a source feature normaliser over %d rows.', len(self.words))

    def set_aligner_state(self, state: dict[str, Any] | None) -> None:
        """Installs a previously fitted raw subject aligner and whitens the raw windows with it.

        Args:
            state (dict[str, Any] | None): A `RawSubjectAligner.state`; `None` leaves the windows unaligned.
        """
        if state is None or self.raw_eeg is None:
            return
        if self._aligned:
            _LOG.warning('set_aligner_state: the raw windows are already aligned; leaving them be.')
            return
        self.aligner = RawSubjectAligner.from_state(state)
        self._write_aligned(self.words['subject'].to_numpy())
        _LOG.info(
            'Installed a source raw aligner for %d subjects (signature dim %d).',
            len(self.aligner.references),
            self.aligner.signature_dim,
        )

    def _write_aligned(self, subjects: np.ndarray) -> None:
        """Streams the whitened windows into their own memmap and rebinds `raw_eeg` onto it.

        The source may be a read-only mapping, and materialising a second ~24 GB tensor is what kills the runtime.
        """
        if self.aligner is None or self.raw_eeg is None:
            return
        target = self._aligned_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        dest = np.lib.format.open_memmap(target, mode='w+', dtype=np.float32, shape=self.raw_eeg.shape)
        self.aligner.transform(self.raw_eeg, subjects, out=dest)
        dest.flush()
        del dest
        self.raw_eeg = np.load(target, mmap_mode='r')
        self._aligned = True

    def _aligned_path(self) -> Path:
        """Scratch memmap for the whitened windows, kept out of the shared bundle (which stays unaligned)."""
        # Amplitude matching yields different windows from the same bundle, so it has to key differently.
        suffix = '_amp' if self.config.raw_align_amplitude else ''
        name = f'{self._cache_key()}_{self.config.raw_align_fit}{suffix}.npy'

        return Path(self.config.cache_dir) / '_aligned' / name

    def _region_index(self) -> np.ndarray | None:
        """Per-electrode scalp-region id from the montage, or `None` to fall back to contiguous blocks."""
        if not self.config.montage_csv:
            return None
        try:
            from zte.data.montage.regions import RegionMap

            return np.asarray(RegionMap.from_csv(self.config.montage_csv).channel_region, dtype=int)
        except Exception as exc:  # montage is optional; a coarse partition is a fine fallback
            _LOG.warning('align_raw: could not read montage regions (%s); using contiguous blocks.', exc)
            return None

    def subject_signatures(self) -> dict[str, np.ndarray]:
        """Standardised signature per subject code (empty when signatures are disabled)."""
        if self.aligner is None or not self.config.subject_signature:
            return {}
        return {
            str(code): self.aligner.signature_for(str(code)) for code in np.unique(self.words['subject'].to_numpy())
        }

    def refit_normalizer(self, train_indices: np.ndarray) -> None:
        """Re-fits the feature normaliser (and eye-tracking fill) on train rows only.

        `_process` fits on every present token, which leaks val/test statistics into the training features, so the
        pipeline calls this once the split is known. Every row is then re-transformed with the train-only statistics
        and `normalizer` updated, so the checkpoint contract reflects them.

        Inverting the current normaliser recovers the pre-normalisation matrix, which makes this idempotent and equally
        valid on a freshly built or cached dataset. A no-op when `config.normalizer_fit == 'all'` or without band power.

        Args:
            train_indices (np.ndarray): Word-row indices belonging to the training split.
        """
        if self.config.normalizer_fit == 'all':
            return
        if self.features is None or self.normalizer is None:
            return

        subjects = self.words['subject'].to_numpy()
        presence = self.presence if self.presence is not None else np.ones(len(self.words), dtype=bool)
        # Recover the raw (pre-normalisation) matrix from the currently-stored features.
        combined = self.normalizer.inverse_transform(self.features, subjects=subjects).copy()

        train_mask = np.zeros(len(self.words), dtype=bool)
        keep = np.asarray(train_indices, dtype=int)
        keep = keep[(keep >= 0) & (keep < len(self.words))]
        train_mask[keep] = True
        fit_mask = train_mask & presence
        if not fit_mask.any():  # degenerate split; keep the existing (all-data) fit.
            _LOG.warning('refit_normalizer: no present train rows; keeping existing statistics.')
            return

        # Re-fit the eye-tracking fill on train-present rows; present rows keep their real gaze scalars either way.
        et_cols = [i for i, name in enumerate(self.feature_names) if name.startswith('ET::')]
        absent = ~presence
        if et_cols and absent.any():
            col_mean = np.nan_to_num(np.nanmean(combined[fit_mask][:, et_cols], axis=0))
            combined[np.ix_(absent, et_cols)] = col_mean[None, :]

        normalizer = FeatureNormalizer(self.config.normalize, eps=self.normalizer.eps)
        normalizer.fit(combined[fit_mask], subjects=subjects[fit_mask])
        self.features = normalizer.transform(combined, subjects=subjects)
        self.normalizer = normalizer
        _LOG.info(
            'Refit normaliser on %d train-present rows (of %d).',
            int(fit_mask.sum()),
            len(self.words),
        )

    def _add_linguistic_features(self) -> None:
        """Adds word length, frequency, relative position and omission flags."""
        from zte.data.targets.categories import corpus_frequencies

        w = self.words
        w['word_len'] = w['word'].str.len().fillna(0).astype(int)
        w['freq'] = w['word'].map(_word_freq_proxy)
        w['log_freq'] = np.log10(w['freq'].astype(float))
        # Corpus-derived term frequency, which degrades to the proxy scale on tiny/synthetic corpora.
        w['corpus_freq'] = corpus_frequencies(w['word']).to_numpy()
        w['corpus_log_freq'] = np.log10(np.clip(w['corpus_freq'].astype(float), 1e-6, None))
        w['is_omitted'] = w['FFD'].isna().astype(int)
        w['sentence_uid'] = w['subject'].astype(str) + '|' + w['task'].astype(str) + '|' + w['sentence_idx'].astype(str)
        max_idx = w.groupby('sentence_uid')['word_idx'].transform('max')
        w['rel_pos'] = (w['word_idx'] / (max_idx + 1)).astype(float)

    def _attach_categories(self) -> None:
        """Labels sentences with a category and length band, propagating both to words.

        Also attaches a subject-agnostic `stimulus_key`, the normalised sentence text, so the same sentence read by
        different subjects shares one key. That is what lets `by_stimulus` keep a stimulus wholly on one side of the
        split, and what the torch bridge hashes into a cross-subject `content_id`.
        """
        from zte.data.targets.categories import normalise_text, sentence_categories

        self.sentences = sentence_categories(self.sentences, root=self.config.root)
        self.sentences['stimulus_key'] = self.sentences['text'].map(normalise_text)
        cols = [
            'subject',
            'task',
            'sentence_idx',
            'category',
            'category_scheme',
            'length_band',
            'stimulus_key',
        ]
        self.words = self.words.merge(self.sentences[cols], on=['subject', 'task', 'sentence_idx'], how='left')

    def _maybe_add_eye_tracking(
        self, band_flat: np.ndarray, names: list[str], presence: np.ndarray
    ) -> tuple[np.ndarray, list[str]]:
        """Appends per-word eye-tracking scalars to the band-power matrix if enabled.

        Returns:
            tuple[np.ndarray, list[str]]: `(features, names)` -- unchanged when `include_eye_tracking` is `False` or no
                eye-tracking columns exist, else the concatenated matrix with `ET::`-prefixed names appended.
        """
        if not self.config.include_eye_tracking:
            _LOG.info('Eye-tracking excluded (EEG-only representation).')
            return band_flat, names
        cols = [c for c in self.config.eye_tracking_measures if c in self.words.columns]
        if not cols:
            return band_flat, names
        et = self.words[cols].to_numpy(dtype=np.float32)  # NaN for omitted words
        ref = et[presence] if presence.any() else et
        col_mean = np.nan_to_num(np.nanmean(np.where(np.isnan(ref), np.nan, ref), axis=0))
        filled = np.where(np.isnan(et), col_mean[None, :], et).astype(np.float32)
        combined = np.concatenate([band_flat, filled], axis=1)
        _LOG.info('Appended %d eye-tracking feature(s): %s', len(cols), ', '.join(cols))
        return combined, [*names, *(f'ET::{c}' for c in cols)]

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
        """Adds compact `<measure>_<band>_chmean` columns for analysis/plots."""
        if self.band_power_raw is None:
            return
        means = channel_mean_features(self.band_power_raw)
        for j, name in enumerate(self.bp_feature_names):
            self.words[f'{name}_chmean'] = means[:, j]

    def _drop_missing_rows(self) -> None:
        """Physically removes omitted-word rows when `missing.method='drop'`."""
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
        """Groups word rows by sentence into `((subject, task, sentence_idx), word_row_indices)`.

        Word indices within each group are ordered by `word_idx` so sequence models receive words in reading order.

        Returns:
            list[tuple[tuple[str, str, int], np.ndarray]]: One pair per sentence.
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

    @property
    def subjects(self) -> list[str]:
        """The sorted, unique subject codes present in the dataset.

        Returns:
            list[str]: The sorted, unique subject codes present in the dataset.
        """
        if len(self.words):
            return sorted(self.words['subject'].unique().tolist())
        return []

    def split(
        self,
        strategy: Literal[
            'random',
            'by_sentence',
            'by_stimulus',
            'by_subject_loso',
            'by_task',
            'by_subject_and_stimulus',
        ]
        | None = None,
        val_fraction: float = 0.1,
        test_fraction: float = 0.0,
        holdout_subject: str | None = None,
        holdout_task: str | None = None,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """Produces leakage-aware train/val (and optional test) row-index splits.

        Args:
            strategy (str | None): Split strategy, defaulting to `by_sentence`. `random` splits per word;
                `by_sentence` keys on `subject|task|sentence_idx`, so the same text read by different subjects can
                still land on both sides; `by_stimulus` keys on the normalised text and closes that cross-subject
                leak; `by_subject_loso` and `by_task` hold out one group as `val`; `by_subject_and_stimulus`
                crosses a held-out subject with a stimulus partition (see `_split_subject_and_stimulus`).
            val_fraction (float): Validation fraction for `random`/`by_sentence`/`by_subject_and_stimulus`.
            test_fraction (float): Disjoint test fraction for `random`/`by_sentence` (`0` omits the `test` key).
                Ignored by the hold-out-group strategies, whose held-out group is already `val`, and required to be
                positive by `by_subject_and_stimulus`, whose test cell is a stimulus partition.
            holdout_subject (str | None): Subject to hold out for LOSO (else the last subject).
            holdout_task (str | None): Task to hold out for `by_task` (else the last task).
            seed (int): RNG seed for the randomised strategies.

        Returns:
            dict[str, np.ndarray]: Disjoint `train`, `val` and (when `test_fraction > 0`) `test` row indices, plus
                `test_seen_stim` for `by_subject_and_stimulus`.
        """
        strategy = strategy or 'by_sentence'
        rng = np.random.default_rng(seed)
        n = len(self.words)
        idx = np.arange(n)

        if strategy == 'random':
            return _partition(rng.permutation(n), val_fraction, test_fraction)

        if strategy == 'by_sentence':
            uids = self.words['sentence_uid'].unique()
            perm_uids = uids[rng.permutation(len(uids))]
            buckets = _partition(np.arange(len(perm_uids)), val_fraction, test_fraction)
            out: dict[str, np.ndarray] = {}
            for name, positions in buckets.items():
                keep = set(perm_uids[positions].tolist())
                out[name] = idx[self.words['sentence_uid'].isin(keep).to_numpy()]
            return out

        if strategy == 'by_stimulus':
            # Keyed on normalised text, so a stimulus is indivisible across subjects and tasks.
            keys = self.words['stimulus_key'].fillna('').to_numpy()
            unique = np.array(sorted(set(keys.tolist())), dtype=object)
            perm_keys = unique[rng.permutation(len(unique))]
            buckets = _partition(np.arange(len(perm_keys)), val_fraction, test_fraction)
            out = {}
            for name, positions in buckets.items():
                keep = set(perm_keys[positions].tolist())
                out[name] = idx[self.words['stimulus_key'].fillna('').isin(keep).to_numpy()]
            return out

        if strategy == 'by_subject_loso':
            hold = holdout_subject or self.subjects[-1]
            in_val = (self.words['subject'] == hold).to_numpy()
            return {'train': idx[~in_val], 'val': idx[in_val]}

        if strategy == 'by_subject_and_stimulus':
            return self._split_subject_and_stimulus(idx, val_fraction, test_fraction, holdout_subject, seed)

        # by_task
        tasks = sorted(self.words['task'].unique())
        hold = holdout_task or tasks[-1]
        in_val = (self.words['task'] == hold).to_numpy()
        return {'train': idx[~in_val], 'val': idx[in_val]}

    def _split_subject_and_stimulus(
        self,
        idx: np.ndarray,
        val_fraction: float,
        test_fraction: float,
        holdout_subject: str | None,
        seed: int,
    ) -> dict[str, np.ndarray]:
        """Crosses a held-out subject with a seeded stimulus partition, naming each cell's generalisation axis.

        Every cell states what it generalises over: `val` over language only (seen subjects, unseen stimuli, so it is
        the model-selection cell), `test` over both (unseen subject reading unseen stimuli, the honest headline) and
        `test_seen_stim` over the brain only (unseen subject reading training stimuli, a diagnostic that must never be
        collapsed into `test`).

        Args:
            idx (np.ndarray): `(n_words,)` row indices of the dataset.
            val_fraction (float): Fraction of unique stimulus keys reserved for `val`.
            test_fraction (float): Fraction of unique stimulus keys reserved for `test`.
            holdout_subject (str | None): Subject to hold out (else the last subject).
            seed (int): RNG seed for the stimulus permutation.

        Returns:
            dict[str, np.ndarray]: `train`, `val`, `test` and `test_seen_stim` row indices.

        Raises:
            ValueError: If `test_fraction` is not positive, which would leave the headline cell empty.
        """
        if test_fraction <= 0:
            raise ValueError(
                'by_subject_and_stimulus needs test_fraction > 0: its test cell is the unseen-subject x '
                'unseen-stimulus partition, which is empty without held-out stimuli.'
            )
        hold = holdout_subject or self.subjects[-1]
        keys = self.words['stimulus_key'].fillna('')
        unique = np.array(sorted(set(keys.tolist())), dtype=object)

        # Drawn independently of the subject mask, so every LOSO fold holds out the same texts and the folds pool.
        rng = np.random.default_rng(seed)
        perm_keys = unique[rng.permutation(len(unique))]
        buckets = _partition(np.arange(len(perm_keys)), val_fraction, test_fraction)
        key_sets = {name: set(perm_keys[positions].tolist()) for name, positions in buckets.items()}

        in_hold = (self.words['subject'] == hold).to_numpy()
        in_train_keys = keys.isin(key_sets['train']).to_numpy()
        return {
            'train': idx[~in_hold & in_train_keys],
            'val': idx[~in_hold & keys.isin(key_sets['val']).to_numpy()],
            'test': idx[in_hold & keys.isin(key_sets['test']).to_numpy()],
            'test_seen_stim': idx[in_hold & in_train_keys],
        }

    # -- analysis & selection ---------------------------------------------- #

    def analyze(self) -> dict[str, Any]:
        """Computes a compact summary of the dataset (counts, missingness, stats).

        Returns:
            dict[str, Any]: JSON-safe subject/task counts, per-measure missing rates, omission statistics and
                band-power presence.
        """
        w = self.words
        from zte.data.schema import ET_MEASURES

        summary: dict[str, Any] = {
            'n_words': int(len(w)),
            'n_sentences': int(w['sentence_uid'].nunique()),
            'n_subjects': int(w['subject'].nunique()),
            'subjects': sorted(w['subject'].unique().tolist()),
            'tasks': sorted(w['task'].unique().tolist()),
            # `int(...)`: pandas counts are `np.int64`, which is NOT an `int` subclass, so it would be
            # stringified by `json.dumps(default=str)` and read back as `"18432"`.
            'words_per_task': {str(k): int(v) for k, v in w.groupby('task')['word_idx'].count().to_dict().items()},
            'omission_rate_overall': float(w['is_omitted'].mean()),
            'omission_rate_by_subject': w.groupby('subject')['is_omitted'].mean().round(4).to_dict(),
            'missing_rate_by_measure': {m: float(w[m].isna().mean()) for m in ET_MEASURES if m in w},
            'include_eye_tracking': bool(self.config.include_eye_tracking),
            'n_features': int(self.features.shape[1]) if self.features is not None else 0,
        }
        if 'category' in w:
            summary['category_scheme'] = sorted(w['category_scheme'].dropna().unique().tolist())
            summary['sentences_by_category'] = (
                {
                    str(k): int(v)
                    for k, v in self.sentences.groupby('category')['sentence_idx'].count().to_dict().items()
                }
                if 'category' in self.sentences
                else {}
            )
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
            target (str): A column in `words` to predict (e.g. `'log_freq'` or `'is_omitted'`).
            method (SelectionMethod): Selection method (see `FeatureSelector`).
            k (int | None): Number of top features to keep.
            present_only (bool): Restrict scoring to present (non-omitted) tokens.

        Returns:
            SelectionResult: The selected indices, scores and names.

        Raises:
            RuntimeError: If band-power features were not built.
            KeyError: If `target` is not a column of `words`.
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
            split (np.ndarray | None): Optional row indices selecting a split (e.g. from `split`).
            **kwargs (Any): Additional keyword arguments forwarded to `ZuCoTorchDataset`.

        Returns:
            ZuCoTorchDataset: The torch-side view over the selected rows.
        """
        from zte.data.torch_dataset import ZuCoTorchDataset

        return ZuCoTorchDataset(self, indices=split, **kwargs)

    # -- persistence -------------------------------------------------------- #

    def _extract_key(self) -> str:
        """Builds the cache key for the raw `.mat` extraction.

        Only the fields `discover_files` and `_load_mat` actually read enter this key, which is what
        makes the extraction reusable: everything `_process` consumes (normalisation, imputation,
        eye-tracking, length filters, band-pass) is excluded, so changing any of those re-derives a
        bundle in seconds instead of re-parsing every `.mat` file.
        """
        import hashlib

        cfg = self.config
        payload = {field: _jsonable(getattr(cfg, field)) for field in _EXTRACT_FIELDS}
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
        synthetic = 'synthetic' in str(cfg.root).lower()
        readable = '_'.join(['-'.join(cfg.tasks), cfg.representation, f'rw{cfg.raw_window}'])
        return f'{"synthetic_" if synthetic else ""}{readable}_{digest}'

    def _cache_key(self) -> str:
        """Builds a deterministic cache subfolder name: a readable prefix plus a config hash.

        The cache is shared across experiments and sessions, so the hash must cover every field that changes the
        processed arrays -- a coarse key would load the wrong tensors for configs differing only in, say, `bands`.
        Location and serialisation settings are excluded, as is `montage_csv`, which never enters the arrays.
        """
        import dataclasses
        import hashlib
        import json

        cfg = self.config
        payload = dataclasses.asdict(cfg)
        # Excluded because they say WHERE/WHETHER to cache, or are applied by `align_raw` after the
        # bundle loads -- baking either in would invalidate bundles that took hours to build.
        for ignore in (
            'root',
            'cache_dir',
            'cache_remote',
            'cache_extracts',
            'cache_format',
            'montage_csv',
            'raw_align',
            'raw_align_fit',
            'raw_align_amplitude',
            'subject_signature',
        ):
            payload.pop(ignore, None)
        # `root` is excluded so one recording keys the same from any machine, but synthetic must never
        # share a key with real ZuCo. The prefix carries that split, leaving real-data digests untouched.
        synthetic = 'synthetic' in str(cfg.root).lower()
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]
        readable = '_'.join(['-'.join(cfg.tasks), cfg.representation, cfg.normalize, f'rw{cfg.raw_window}'])
        return f'{"synthetic_" if synthetic else ""}{readable}_{digest}'

    def _save_extract(self, path: str | Path) -> Path:
        """Saves the raw `.mat` extraction (pre-processing) so other configs can skip the parse.

        Only what `_load_mat` produced is stored: the word/sentence tables and the raw arrays. The

        processed `features`, `presence` and fitted normaliser are deliberately absent -- they are the

        cheap, config-specific part that `_process` re-derives.

        Args:
            path (str | Path): Destination directory (created if needed).
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        if self.band_power_raw is not None:
            arrays['band_power_raw'] = self.band_power_raw
        if self.raw_eeg is not None:
            arrays['raw_eeg'] = self.raw_eeg
        np.savez_compressed(out / 'arrays.npz', **arrays)
        self.words.to_pickle(out / 'words.pkl')
        self.sentences.to_pickle(out / 'sentences.pkl')
        meta = {
            'extract_key': self._extract_key(),
            'bp_feature_names': self.bp_feature_names,
            # Informational only: the fields below are what the extraction actually depends on.
            'extract_config': {field: _jsonable(getattr(self.config, field)) for field in _EXTRACT_FIELDS},
        }
        (out / 'meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
        _LOG.info('Saved .mat extraction to %s', out)
        return out

    def _load_extract(self, path: str | Path) -> ZuCoDataset:
        """Restores a raw extraction saved by `_save_extract`, leaving `self.config` untouched.

        The whole point of the extraction cache is to serve a *different* config from the one that built
        it, so unlike `load` this must not adopt the stored config.
        """
        src = Path(path)
        meta = json.loads((src / 'meta.json').read_text(encoding='utf-8'))
        self.words = pd.read_pickle(src / 'words.pkl')
        self.sentences = pd.read_pickle(src / 'sentences.pkl')
        with np.load(src / 'arrays.npz') as arrays:
            self.band_power_raw = arrays['band_power_raw'] if 'band_power_raw' in arrays else None
            self.raw_eeg = arrays['raw_eeg'] if 'raw_eeg' in arrays else None
        self.bp_feature_names = meta['bp_feature_names']
        self.features = None
        self.presence = None
        self.normalizer = None
        self._groups = None
        return self

    def save(self, path: str | Path) -> Path:
        """Saves the full processed dataset as a self-contained directory bundle.

        The bundle holds `arrays.npz`, `words.pkl`, `sentences.pkl` and a `meta.json` carrying the config, feature
        names and fitted normaliser state, so `load` reproduces the object exactly.

        Args:
            path (str | Path): Destination directory (created if needed).

        Returns:
            Path: The bundle directory.
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
        np.savez_compressed(out / 'arrays.npz', **arrays)

        # Raw windows get their own UNCOMPRESSED .npy so `load` can memory-map them: at ~24 GB a
        # compressed .npz member must be inflated into RAM in full before one window can be read.
        if self.raw_eeg is not None:
            np.save(out / RAW_ARRAY_FILE, self.raw_eeg)
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
        """Loads a dataset bundle previously written by `save`.

        Args:
            path (str | Path): The bundle directory.
            into (ZuCoDataset | None): Optionally populate an existing instance, as the cache path in `build` does.

        Returns:
            ZuCoDataset: The populated dataset.
        """
        src = Path(path)
        meta = json.loads((src / 'meta.json').read_text(encoding='utf-8'))
        config = DatasetConfig(**_coerce_dataset_meta(meta['config']))

        ds = into if into is not None else cls(config)
        ds.config = config
        ds.words = pd.read_pickle(src / 'words.pkl')
        ds.sentences = pd.read_pickle(src / 'sentences.pkl')
        with np.load(src / 'arrays.npz') as arrays:
            ds.band_power_raw = arrays['band_power_raw'] if 'band_power_raw' in arrays else None
            ds.features = arrays['features'] if 'features' in arrays else None
            ds.presence = arrays['presence'] if 'presence' in arrays else None
            has_legacy_raw = 'raw_eeg' in arrays

        # Memory-mapped so only the windows a batch touches become resident. Attempted BEFORE reading
        # the npz member, since that read is exactly the multi-GB allocation being avoided.
        raw_file = src / RAW_ARRAY_FILE
        if not raw_file.is_file() and has_legacy_raw:
            _extract_raw_member(src)

        if raw_file.is_file():
            ds.raw_eeg = np.load(raw_file, mmap_mode='r')

            # Bundles predating source sanitisation may hold NaN/unscaled windows; a sampled check is
            # cheap, and only a bad one pays for the in-memory repair.
            probe = ds.raw_eeg[:: max(1, len(ds.raw_eeg) // 256)]
            if not np.isfinite(probe).all():
                _LOG.warning('Bundle %s holds unsanitised windows; repairing in memory.', src.name)
                ds.raw_eeg = sanitize_raw_windows(np.array(ds.raw_eeg))  # writable copy; repairs in place
        elif has_legacy_raw:
            with np.load(src / 'arrays.npz') as arrays:
                ds.raw_eeg = sanitize_raw_windows(arrays['raw_eeg'])
        ds.feature_names = meta['feature_names']
        ds.bp_feature_names = meta['bp_feature_names']
        if meta.get('normalizer'):
            ds.normalizer = FeatureNormalizer.from_state(meta['normalizer'])
        ds._groups = None
        ds._backfill_derived_columns()
        return ds

    def _backfill_derived_columns(self) -> None:
        """Rebuilds the cheap derived word/sentence columns a stale cached bundle may lack.

        `sentence_uid`, the linguistic features and the category labels are pure functions of the base
        columns, so an older bundle (built before a column was added) is repaired in place on load rather
        than triggering a multi-hour reprocess.
        """
        if self.words is None or not len(self.words):
            return

        # Re-read `self.words` at each step, never a captured frame: under pandas Copy-on-Write an
        # in-place column add can detach a stale reference, dropping the fresh `sentence_uid`.
        if 'sentence_uid' not in self.words.columns:
            _LOG.info('Cached bundle predates `sentence_uid`; backfilling linguistic features.')
            self._add_linguistic_features()

        if 'stimulus_key' not in self.words.columns or 'category' not in self.words.columns:
            # Drop any partial category columns first, so `_attach_categories`'s merge cannot collide.
            stale = ['category', 'category_scheme', 'length_band', 'stimulus_key']
            self.words = self.words.drop(columns=[c for c in stale if c in self.words.columns])
            try:
                self._attach_categories()
            except (OSError, KeyError, ValueError) as exc:  # pragma: no cover - defensive
                _LOG.warning('Could not backfill categories on load: %r', exc)

    # -- remote ------------------------------------------------------------- #

    def save_to_drive(self, remote_dir: str, local_tmp: str | Path = 'res/.drive_tmp') -> str:
        """Saves the bundle locally then uploads it to Google Drive.

        Args:
            remote_dir (str): Destination Drive folder id or path (see `zte.data.io.remote`).
            local_tmp (str | Path): Local staging directory for the bundle.

        Returns:
            str: The remote location string returned by the uploader.
        """
        from zte.data.io.remote import upload_directory

        local = self.save(local_tmp)
        return upload_directory(local, remote_dir)

    @classmethod
    def from_drive(cls, remote_spec: str, local_tmp: str | Path = 'res/.drive_dl') -> ZuCoDataset:
        """Downloads a bundle from Google Drive and loads it.

        Args:
            remote_spec (str): A Drive file/folder id or shareable URL.
            local_tmp (str | Path): Local directory to download into.

        Returns:
            ZuCoDataset: The loaded dataset.
        """
        from zte.data.io.remote import download_to_dir

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


def _partition(items: np.ndarray, val_fraction: float, test_fraction: float) -> dict[str, np.ndarray]:
    """Splits a (pre-shuffled) index array into train/val (and optional test).

    Returns:
        dict[str, np.ndarray]: Disjoint `train`, `val` and (when `test_fraction > 0`) `test` index arrays.
    """
    total = len(items)
    n_test = int(round(total * test_fraction))
    n_val = int(round(total * val_fraction))
    test, val, train = items[:n_test], items[n_test : n_test + n_val], items[n_test + n_val :]
    out = {'train': np.sort(train), 'val': np.sort(val)}
    if test_fraction > 0:
        out['test'] = np.sort(test)
    return out


def _coerce_dataset_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Coerces a serialised dataset-config dict back into constructor kwargs."""
    from zte.config import MissingConfig

    out = dict(meta)
    for key in ('tasks', 'bands', 'band_power_measures', 'subjects', 'eye_tracking_measures'):
        if isinstance(out.get(key), list):
            out[key] = tuple(out[key])
    if isinstance(out.get('bandpass'), list):
        out['bandpass'] = tuple(out['bandpass'])
    if isinstance(out.get('missing'), dict):
        out['missing'] = MissingConfig(**out['missing'])
    return out
