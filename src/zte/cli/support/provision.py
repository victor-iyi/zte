"""Turn-key provisioning of the two heavy, hand-wired experiment ingredients: exact scalp geometry and the word-meaning target.


Usage::

    zte-run --config ... --spatial exact --meaning contextual
    zte-run --config ... --spatial attention --meaning static

`--spatial` bundles the electrode-geometry knobs (`model.spatial_encoding` + `dataset.montage_csv`):

    keep       leave the config as-is (default)
    off        no spatial encoding (`spatial_encoding='none'`, montage cleared)
    approx     spherical-harmonic encoding on the coordinate-free fallback cap (no montage)
    exact      spherical-harmonic encoding on the **exact** ZuCo-105 montage (built + wired here)
    attention  Défossez-style learned spatial attention on the exact montage

`--meaning` selects the distillation target (`objective.meaning_source`/`meaning_contextual`/…):

    keep       leave the config as-is (default)
    hash       deterministic hash target (mechanism only, no semantics)
    static     word-type GloVe vectors, restricted to the dataset vocabulary (built + wired here)
    contextual per-occurrence contextual target from a frozen encoder (e.g. BERT mid-layer)

"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Final, Literal

from zte.config import ZTEConfig
from zte.logging_utils import get_logger

_LOG = get_logger('cli.provision')

type SpatialChoice = Literal['keep', 'off', 'approx', 'exact', 'attention']
type MeaningChoice = Literal['keep', 'hash', 'static', 'contextual']

SPATIAL_CHOICES: Final[tuple[SpatialChoice, ...]] = ('keep', 'off', 'approx', 'exact', 'attention')
MEANING_CHOICES: Final[tuple[MeaningChoice, ...]] = ('keep', 'hash', 'static', 'contextual')

# The exact-montage choices need coordinates built from a standard montage.
_EXACT_SPATIAL: Final[frozenset[str]] = frozenset({'exact', 'attention'})
_SPATIAL_ENCODING: Final[dict[str, str]] = {
    'off': 'none',
    'approx': 'spherical_harmonics',
    'exact': 'spherical_harmonics',
    'attention': 'spatial_attention',
}

# Default paths for the montage and meaning files.
DEFAULT_MONTAGE_OUT: Final[Path] = Path('res/montage_gsn105.csv')
DEFAULT_MEANING_OUT: Final[Path] = Path('res/vectors/glove.300d.txt')
DEFAULT_CONTEXTUAL_MODEL: Final[str] = 'bert-base-uncased'
DEFAULT_CONTEXTUAL_LAYER: Final[int] = 8
DEFAULT_CONTEXTUAL_DIM: Final[int] = 768


def add_provision_args(parser: argparse.ArgumentParser) -> None:
    """Adds the `--spatial` / `--meaning` provisioning flags (and their overrides) to a parser."""
    group = parser.add_argument_group('provisioning (build + wire heavy ingredients per run)')
    group.add_argument(
        '--spatial',
        choices=SPATIAL_CHOICES,
        default='keep',
        help='Electrode spatial encoding + montage. exact/attention build the ZuCo-105 montage '
        'and wire it; approx uses the coordinate-free fallback; off disables it; keep = leave config.',
    )
    group.add_argument(
        '--meaning',
        choices=MEANING_CHOICES,
        default='keep',
        help='Word-meaning distillation target. static builds vocab-restricted GloVe; contextual '
        'wires a frozen contextual encoder (e.g. BERT mid-layer); hash = mechanism only; keep = leave config.',
    )
    group.add_argument(
        '--montage-out',
        type=Path,
        default=DEFAULT_MONTAGE_OUT,
        help='Where --spatial exact/attention writes (and reads) the montage CSV.',
    )
    group.add_argument(
        '--montage-name',
        default=None,
        help='MNE standard montage name for --spatial exact/attention (default the ZuCo 129-net).',
    )
    group.add_argument(
        '--meaning-out',
        default=DEFAULT_MEANING_OUT,
        help='Where --meaning static writes (and reads) the GloVe vectors file.',
    )
    group.add_argument(
        '--meaning-model',
        default=None,
        help='Override the model id: a gensim GloVe name for --meaning static, or a HuggingFace '
        'encoder id for --meaning contextual.',
    )
    group.add_argument(
        '--meaning-layer',
        type=int,
        default=DEFAULT_CONTEXTUAL_LAYER,
        help='Hidden layer for --meaning contextual (a mid layer ~7-9 aligns best with brain data).',
    )
    group.add_argument(
        '--meaning-weight',
        type=float,
        default=None,
        help='Set objective.meaning_distill_weight (the distillation must be on for --meaning to '
        'take effect). Omit to keep the config value.',
    )


def apply_spatial(
    config: ZTEConfig,
    choice: str,
    *,
    montage_out: Path = DEFAULT_MONTAGE_OUT,
    montage_name: str | None = None,
) -> None:
    """Wires `model.spatial_encoding` + `dataset.montage_csv` for a `--spatial` choice (building the CSV if needed).

    On an exact choice with `mne` unavailable, this degrades to the coordinate-free fallback (a warning,
    `montage_csv=None`) rather than failing the run.
    """
    if choice == 'keep':
        return
    config.model.spatial_encoding = _SPATIAL_ENCODING[choice]  # type: ignore[assignment]
    if choice in _EXACT_SPATIAL:
        try:
            from zte.data.montage.montage import DEFAULT_MONTAGE, build_montage_csv

            path = build_montage_csv(
                montage_out, montage=montage_name or DEFAULT_MONTAGE, zuco105=True
            )
            config.dataset.montage_csv = str(path)
            _LOG.info(
                'Provisioned exact montage -> %s (spatial_encoding=%s).',
                path,
                config.model.spatial_encoding,
            )
        except ImportError:
            config.dataset.montage_csv = None
            _LOG.warning(
                '--spatial %s needs the optional `mne` dependency to build exact coordinates; it is not '
                'installed, so falling back to the coordinate-free approximate cap for %s.',
                choice,
                config.model.spatial_encoding,
            )
    else:  # off / approx -> no exact coordinates
        config.dataset.montage_csv = None
    if config.model.spatial_encoding == 'none':
        _LOG.info('Spatial encoding disabled (--spatial off).')


def apply_meaning(
    config: ZTEConfig,
    choice: str,
    *,
    model: str | None = None,
    layer: int = DEFAULT_CONTEXTUAL_LAYER,
    meaning_out: Path = DEFAULT_MEANING_OUT,
    weight: float | None = None,
    vocab: Iterable[str] | None = None,
) -> None:
    """Wires the `objective.meaning_*` fields for a `--meaning` choice (building the GloVe file if needed).

    Args:
        config (ZTEConfig): The config to mutate in place.
        choice (str): One of `MEANING_CHOICES`.
        model (str | None): Override model id (GloVe name for `static`, HF encoder for `contextual`).
        layer (int): Contextual hidden layer.
        meaning_out (Path): Destination/lookup path for the `static` GloVe file.
        weight (float | None): If given, set `objective.meaning_distill_weight`.
        vocab (Iterable[str] | None): Training vocabulary to restrict the `static` GloVe file to (keeps it tiny).
    """
    obj = config.objective
    if weight is not None:
        obj.meaning_distill_weight = weight
    if choice == 'keep':
        return
    if choice == 'hash':
        obj.meaning_source, obj.meaning_contextual = 'hash', None
    elif choice == 'static':
        from zte.data.targets.glove import DEFAULT_MODEL, provision_glove

        path, dim = provision_glove(meaning_out, vocab=vocab, model=model or DEFAULT_MODEL)
        obj.meaning_source, obj.meaning_dim, obj.meaning_contextual = str(path), dim, None
        _LOG.info('Provisioned static meaning vectors -> %s (dim %d).', path, dim)
    elif choice == 'contextual':
        obj.meaning_contextual = model or DEFAULT_CONTEXTUAL_MODEL
        obj.meaning_context_layer = layer
        obj.meaning_dim = (
            DEFAULT_CONTEXTUAL_DIM  # informational; the true hidden size is read at build time
        )
        _LOG.info('Wired contextual meaning target %s (layer %d).', obj.meaning_contextual, layer)
    # A meaning target is inert unless the distillation loss is on; warn instead of silently no-op'ing.
    if choice != 'keep' and obj.meaning_distill_weight <= 0.0:
        _LOG.warning(
            '--meaning %s set a target but objective.meaning_distill_weight is %s (<= 0), so it will '
            'have no effect. Pass --meaning-weight to enable distillation.',
            choice,
            obj.meaning_distill_weight,
        )


def provision_from_args(
    config: ZTEConfig,
    args: argparse.Namespace,
    *,
    vocab: Iterable[str] | None = None,
) -> None:
    """Applies both `--spatial` and `--meaning` (from parsed args) to `config`, in place.

    Note:
        `vocab` (the training word set) is used only to shrink the `--meaning static` GloVe file; it is safe
        to omit (the top-N words are kept instead).

    Args:
        config (ZTEConfig): The config to mutate in place.
        args (argparse.Namespace): The parsed arguments.
        vocab (Iterable[str] | None): Training vocabulary to restrict the `static` GloVe file to (keeps it tiny).

    """
    # Apply the spatial choice.
    apply_spatial(
        config,
        getattr(args, 'spatial', 'keep'),
        montage_out=getattr(args, 'montage_out', DEFAULT_MONTAGE_OUT),
        montage_name=getattr(args, 'montage_name', None),
    )

    # Apply the meaning choice.
    apply_meaning(
        config,
        getattr(args, 'meaning', 'keep'),
        model=getattr(args, 'meaning_model', None),
        layer=getattr(args, 'meaning_layer', DEFAULT_CONTEXTUAL_LAYER),
        meaning_out=getattr(args, 'meaning_out', DEFAULT_MEANING_OUT),
        weight=getattr(args, 'meaning_weight', None),
        vocab=vocab,
    )
