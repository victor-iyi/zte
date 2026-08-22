"""The cross-level atlas: token, word and sentence vectors under one jointly fitted projection, drawn as figures."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np

from zte.logging_utils import get_logger

_LOG = get_logger('alignment.atlas')

type Level = Literal['token', 'word', 'sentence']
"""Which rung of the hierarchy a point sits on."""

type Method = Literal['pca', 'tsne', 'umap']
"""The projection asked for; anything but PCA may degrade to PCA, and the payload says so when it does."""

type ColourBy = Literal['level', 'subject', 'task']
"""Which attribute the marker colour encodes; the level is always carried by the marker symbol as well."""

# plotly is untyped upstream; two aliases keep the figure builders honest instead of leaking bare `Any`.
type PlotlyModule = Any
"""The lazily resolved `plotly.graph_objects` module."""

type FigureJSON = dict[str, Any]
"""One plotly figure as plain JSON -- `data`, `layout`, `frames` -- ready for a kernel that has no ZTE."""

# Fine-to-coarse, fixed so the legend and the trace order read the same way in every atlas.
LEVELS: Final[tuple[Level, ...]] = ('token', 'word', 'sentence')
"""The three rungs the atlas draws together."""

# Colour-blind-safe triad; deliberately not the dashboard's condition ramp, whose colours already mean something else.
LEVEL_COLOURS: Final[dict[str, str]] = {'token': '#0072b2', 'word': '#d55e00', 'sentence': '#009e73'}
"""Marker colour per level when colouring by level."""

# Colouring by subject or task takes the colour channel away from the level, so the symbol has to carry it instead.
LEVEL_SYMBOLS: Final[dict[str, str]] = {'token': 'circle', 'word': 'diamond', 'sentence': 'square'}
"""Marker symbol per level, valid for both the 2D and the 3D scatter."""

# A sentence point stands for far more EEG than a token point, so size tracks what a mark summarises.
LEVEL_SIZES: Final[dict[str, int]] = {'token': 4, 'word': 6, 'sentence': 9}
"""Marker size per level, in pixels."""

# Colour-blind-safe ramp for subject and task colourings; the level ramp stays reserved for levels.
GROUP_COLOURS: Final[tuple[str, ...]] = (
    '#0072b2',
    '#d55e00',
    '#009e73',
    '#cc79a7',
    '#e69f00',
    '#56b4e9',
    '#7a5195',
    '#bc5090',
)
"""Qualitative ramp cycled over subject or task groups."""

# Past a few thousand marks the picture is a smear and the payload is megabytes; the cap is bounded, not adaptive.
MAX_POINTS_PER_LEVEL: Final[int] = 1500
"""Upper bound on the points one level contributes to the atlas."""

DISCLAIMER: Final[str] = (
    'The atlas is inspection, not evidence. A 2D or 3D view of a high-dimensional space drops most of the '
    'variance, and a picture in which the levels look separated is not a retrieval result -- read the '
    'cross-level table for that.'
)
"""The honest caption every atlas payload carries."""


@dataclass(slots=True, frozen=True, kw_only=True)
class LevelPoints:
    """One level's vectors plus the labels that make a hovered point legible."""

    level: Level
    """Which rung these vectors sit on."""

    vectors: np.ndarray
    """`(n, d)` embeddings, in the same space as every other level of the atlas."""

    labels: Sequence[str]
    """`(n,)` the token, word or sentence text each vector came from."""

    subjects: Sequence[str] | None = None
    """`(n,)` subject code per vector, or `None` when the level is not per-subject."""

    tasks: Sequence[str] | None = None
    """`(n,)` ZuCo task per vector, or `None` when the level carries no task."""

    def __post_init__(self) -> None:
        """Rejects a ragged level rather than drawing points whose labels belong to other rows."""
        if self.level not in LEVELS:
            raise ValueError(f'Unknown level {self.level!r}; expected one of {", ".join(LEVELS)}.')

        vectors = np.asarray(self.vectors)
        if vectors.ndim != 2 or vectors.shape[0] == 0 or vectors.shape[1] == 0:
            raise ValueError(f'Level {self.level!r} needs a non-empty (n, d) matrix; got shape {vectors.shape}.')

        n = int(vectors.shape[0])
        columns = (('labels', self.labels), ('subjects', self.subjects), ('tasks', self.tasks))
        for name, column in columns:
            if column is not None and len(column) != n:
                raise ValueError(f'Level {self.level!r} has {n} vectors but {len(column)} {name}.')


def build_atlas(
    levels: Sequence[LevelPoints],
    *,
    method: Method = 'pca',
    colour_by: ColourBy = 'level',
    max_points_per_level: int = MAX_POINTS_PER_LEVEL,
    seed: int = 0,
) -> dict[str, Any]:
    """Projects every level into one shared space and returns the 2D and 3D figures as plotly figure JSON.

    The projection is fitted once over the stacked rows of every level, so the three point sets are three
    views of one geometry rather than three unrelated scatter plots. Under PCA the 2D view is the 3D view's
    first two axes; t-SNE and UMAP fit each view separately, and the payload says which case applies.

    Args:
        levels (Sequence[LevelPoints]): One entry per level, each in the same embedding space.
        method (Method, optional): Projection to fit. UMAP degrades to PCA when it is not installed.
            Defaults to 'pca'.
        colour_by (ColourBy, optional): Attribute the marker colour encodes. Defaults to 'level'.
        max_points_per_level (int, optional): Cap on the points one level contributes.
            Defaults to MAX_POINTS_PER_LEVEL.
        seed (int, optional): Seed for the subsample and the manifold fit. Defaults to 0.

    Returns:
        dict[str, Any]: `projection` (what was fitted, on how many rows, and the variance retained),
            `levels` (per-level counts), `figures` with the `2d` and `3d` figure JSON, and `disclaimer`.

    Raises:
        ValueError: If no levels are given, a level appears twice, or the levels disagree on dimension --
            a joint projection is only meaningful inside one space.
        ImportError: If plotly is not installed.
    """
    if not levels:
        raise ValueError('The atlas needs at least one level; there is nothing to project.')

    named = [points.level for points in levels]
    if len(set(named)) != len(named):
        raise ValueError(f'Each level may appear at most once; got {named}.')

    widths = sorted({int(np.asarray(points.vectors).shape[1]) for points in levels})
    if len(widths) != 1:
        raise ValueError(f'A joint projection needs one shared space; the levels carry {widths} dimensions.')

    # Thin first, fit second: the projection is then fitted on exactly the rows the picture draws.
    ordered = sorted(levels, key=lambda points: LEVELS.index(points.level))
    drawn = [_thinned(points, max_points_per_level, seed) for points in ordered]
    matrix = np.concatenate([np.asarray(points.vectors, dtype=np.float64) for points, _ in drawn], axis=0)

    used, degraded_reason = _resolve_method(method)
    coords_2d, coords_3d, variance = _fit(matrix, used, seed)

    sizes = [int(np.asarray(points.vectors).shape[0]) for points, _ in drawn]
    split_2d = _split(coords_2d, sizes)
    split_3d = _split(coords_3d, sizes)

    points_only = [points for points, _ in drawn]
    retained_3d = variance['explained_variance_3d']
    caption = f'{used.upper()} -- {retained_3d:.1%} of variance in 3D' if retained_3d is not None else used.upper()

    figures = {
        '2d': _scatter_figure(
            points_only,
            split_2d,
            dims=2,
            colour_by=colour_by,
            title=f'ZTE cross-level atlas ({caption})',
            axis_labels=_axis_labels(used, variance, 2),
        ),
        '3d': _scatter_figure(
            points_only,
            split_3d,
            dims=3,
            colour_by=colour_by,
            title=f'ZTE cross-level atlas ({caption})',
            axis_labels=_axis_labels(used, variance, 3),
        ),
    }

    return {
        'schema': 'zte.alignment.atlas/1',
        'method': used,
        'method_requested': method,
        'degraded': degraded_reason is not None,
        'degraded_reason': degraded_reason,
        'colour_by': colour_by,
        'seed': seed,
        'projection': {
            'fitted_on': 'all levels jointly',
            'n_fit_rows': int(matrix.shape[0]),
            'embed_dim': widths[0],
            'views_share_a_basis': used == 'pca',
            **variance,
        },
        'levels': [
            {
                'level': points.level,
                'n': int(np.asarray(source.vectors).shape[0]),
                'n_plotted': int(np.asarray(points.vectors).shape[0]),
                'subsampled': thinned,
            }
            for source, (points, thinned) in zip(ordered, drawn, strict=True)
        ],
        'figures': figures,
        'disclaimer': DISCLAIMER,
    }


def contrastive_figure(report: dict[str, Any]) -> FigureJSON:
    """Draws the positive/negative similarity gap per level, with its bootstrap CI, as plotly figure JSON.

    The gap is what the contrastive term is paid to open: how much closer a positive pair sits than an
    average negative one. A bar whose interval crosses zero bought nothing at that level.

    Args:
        report (dict[str, Any]): A `zte.alignment.contrastive.contrastive_geometry` payload.

    Returns:
        FigureJSON: One horizontal bar per level, with asymmetric error bars from the bootstrap CI.

    Raises:
        ValueError: If the report carries no level blocks.
        ImportError: If plotly is not installed.
    """
    blocks = report.get('levels') or {}
    ordered = [(level, blocks[level]) for level in LEVELS if level in blocks]
    if not ordered:
        raise ValueError('The contrastive report carries no level blocks; nothing to draw.')

    go = _go()
    gaps = [float(block['positive_negative_gap']) for _, block in ordered]
    los = [float(block['positive_negative_gap_ci'][0]) for _, block in ordered]
    his = [float(block['positive_negative_gap_ci'][1]) for _, block in ordered]
    hover = [
        f'<b>{level}</b><br>gap: {gap:.3f} [{lo:.3f}, {hi:.3f}]'
        f'<br>alignment (mean positive cosine): {_show(block.get("alignment"))}'
        f'<br>uniformity: {_show(block.get("uniformity"))}'
        f'<br>effective rank: {_show(block.get("effective_rank"))}'
        f' / {block.get("embed_dim", "?")}'
        for (level, block), gap, lo, hi in zip(ordered, gaps, los, his, strict=True)
    ]

    fig = go.Figure(
        go.Bar(
            x=gaps,
            y=[level for level, _ in ordered],
            orientation='h',
            marker={'color': [LEVEL_COLOURS[level] for level, _ in ordered]},
            error_x={
                'type': 'data',
                'symmetric': False,
                'array': [hi - gap for gap, hi in zip(gaps, his, strict=True)],
                'arrayminus': [gap - lo for gap, lo in zip(gaps, los, strict=True)],
                'color': '#333333',
                'thickness': 1.4,
            },
            hovertext=hover,
            hoverinfo='text',
        )
    )
    fig.add_vline(x=0.0, line={'width': 1, 'dash': 'dot', 'color': '#888888'})
    fig.update_layout(
        title='What the contrastive term bought: positive minus negative cosine, per level',
        xaxis_title='mean positive cosine - mean negative cosine (95% bootstrap CI)',
        yaxis_title='',
        height=320,
        template='plotly_white',
        margin={'l': 90, 'r': 30, 't': 60, 'b': 60},
        font={'family': 'system-ui, -apple-system, Segoe UI, sans-serif', 'size': 12},
        showlegend=False,
    )

    return _figure_json(fig)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #


def _resolve_method(method: Method) -> tuple[Method, str | None]:
    """Falls back to PCA when the requested projector is unavailable, returning the reason to publish."""
    if method not in ('pca', 'tsne', 'umap'):
        raise ValueError(f'Unknown projection {method!r}; expected pca, tsne or umap.')

    if method != 'umap':
        return method, None

    try:
        _umap_reducer()
    except ImportError as exc:
        reason = f'umap is not installed ({exc}); projected with PCA instead'
        _LOG.warning('UMAP unavailable, falling back to PCA: %s', exc)

        return 'pca', reason

    return method, None


def _umap_reducer() -> Any:
    """Returns the UMAP class; the optional dependency is imported here so the fallback has one seam."""
    from umap import UMAP

    return UMAP


def _fit(matrix: np.ndarray, method: Method, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fits one projection over the stacked rows of every level and returns the 2D view, 3D view and variance."""
    match method:
        case 'pca':
            return _fit_pca(matrix)
        case _:
            coords_2d = _fit_manifold(matrix, method, 2, seed)
            coords_3d = _fit_manifold(matrix, method, 3, seed)
            note = f'{method} is a neighbourhood embedding, not a linear projection: no variance is retained by axis.'

            return coords_2d, coords_3d, _variance_block(None, note)


def _fit_pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One centred SVD over every level's rows, so the 2D view is the 3D view's first two axes."""
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)

    coords = u[:, :3] * s[:3]
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))

    total = float((s**2).sum())
    ratios = (s**2 / total) if total > 0.0 else np.zeros_like(s)
    note = 'Variance retained by the joint PCA basis, over the stacked rows of every level.'

    return coords[:, :2], coords, _variance_block(ratios, note)


def _fit_manifold(matrix: np.ndarray, method: Method, dims: int, seed: int) -> np.ndarray:
    """Fits t-SNE or UMAP once over the stacked rows, so every level lands in the same embedding."""
    n = int(matrix.shape[0])
    if method == 'umap':
        reducer = _umap_reducer()(n_components=dims, random_state=seed, n_neighbors=min(15, max(2, n - 1)))

        return np.asarray(reducer.fit_transform(matrix), dtype=np.float64)

    from sklearn.manifold import TSNE

    # Perplexity has to stay under the sample count, and t-SNE is initialised from PCA so the fit is reproducible.
    perplexity = float(min(30.0, max(5.0, (n - 1) / 3.0)))
    tsne = TSNE(n_components=dims, random_state=seed, init='pca', perplexity=min(perplexity, max(1.0, n - 1.5)))

    return np.asarray(tsne.fit_transform(matrix), dtype=np.float64)


def _variance_block(ratios: np.ndarray | None, note: str) -> dict[str, Any]:
    """Packs the retained-variance statement; `None` ratios mean the method has no linear axes to report."""
    if ratios is None:
        return {
            'explained_variance_ratio': None,
            'explained_variance_2d': None,
            'explained_variance_3d': None,
            'explained_variance_note': note,
        }

    return {
        'explained_variance_ratio': [float(v) for v in ratios[:3]],
        'explained_variance_2d': float(ratios[:2].sum()),
        'explained_variance_3d': float(ratios[:3].sum()),
        'explained_variance_note': note,
    }


def _thinned(points: LevelPoints, cap: int, seed: int) -> tuple[LevelPoints, bool]:
    """Seeded subsample of one level down to `cap` rows, keeping the input order so the picture stays stable."""
    vectors = np.asarray(points.vectors)
    n = int(vectors.shape[0])
    if cap <= 0:
        raise ValueError(f'max_points_per_level must be positive; got {cap}.')
    if n <= cap:
        return points, False

    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(n, size=cap, replace=False))

    return (
        LevelPoints(
            level=points.level,
            vectors=vectors[keep],
            labels=[points.labels[int(i)] for i in keep],
            subjects=_gather(points.subjects, keep),
            tasks=_gather(points.tasks, keep),
        ),
        True,
    )


def _gather(column: Sequence[str] | None, keep: np.ndarray) -> list[str] | None:
    """Indexes a label column by the kept rows, leaving an absent column absent."""
    return None if column is None else [column[int(i)] for i in keep]


def _split(coords: np.ndarray, sizes: Sequence[int]) -> list[np.ndarray]:
    """Cuts the jointly projected coordinates back into one block per level, in the order they were stacked."""
    bounds = np.cumsum([0, *sizes])

    return [coords[bounds[i] : bounds[i + 1]] for i in range(len(sizes))]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def _go() -> PlotlyModule:
    """Imports `plotly.graph_objects`, naming the dependency group when it is missing."""
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - viz is a default group
        raise ImportError('The atlas needs plotly: `uv sync --group viz`.') from exc

    return go


def _scatter_figure(
    drawn: Sequence[LevelPoints],
    coords: Sequence[np.ndarray],
    *,
    dims: int,
    colour_by: ColourBy,
    title: str,
    axis_labels: Sequence[str],
) -> FigureJSON:
    """Builds one scatter over every level's projected points, colour by attribute and symbol by level."""
    go = _go()
    palette = _palette(drawn, colour_by)
    fig = go.Figure()

    legended: set[str] = set()
    for points, view in zip(drawn, coords, strict=True):
        keys = np.asarray(_colour_keys(points, colour_by), dtype=object)
        hover = _hover(points)
        for group in sorted({str(key) for key in keys}):
            mask = keys == group
            marker = {
                'size': LEVEL_SIZES[points.level],
                'color': palette[group],
                'symbol': LEVEL_SYMBOLS[points.level],
                'opacity': 0.85,
            }
            shared = {
                'mode': 'markers',
                'name': group if colour_by == 'level' else f'{group} · {points.level}',
                'legendgroup': group,
                'showlegend': group not in legended,
                'marker': marker,
                'hovertext': [hover[i] for i in np.flatnonzero(mask)],
                'hoverinfo': 'text',
            }
            legended.add(group)

            xs = view[mask, 0].tolist()
            ys = view[mask, 1].tolist()
            if dims == 3:
                fig.add_trace(go.Scatter3d(x=xs, y=ys, z=view[mask, 2].tolist(), **shared))
            else:
                fig.add_trace(go.Scatter(x=xs, y=ys, **shared))

    layout: dict[str, Any] = {
        'title': title,
        'template': 'plotly_white',
        'height': 640,
        'margin': {'l': 60, 'r': 30, 't': 60, 'b': 60},
        'font': {'family': 'system-ui, -apple-system, Segoe UI, sans-serif', 'size': 12},
        'legend': {'itemsizing': 'constant'},
    }
    if dims == 3:
        layout['scene'] = {
            'xaxis': {'title': {'text': axis_labels[0]}},
            'yaxis': {'title': {'text': axis_labels[1]}},
            'zaxis': {'title': {'text': axis_labels[2]}},
        }
    else:
        layout['xaxis_title'] = axis_labels[0]
        layout['yaxis_title'] = axis_labels[1]

    fig.update_layout(**layout)

    return _figure_json(fig)


def _palette(drawn: Sequence[LevelPoints], colour_by: ColourBy) -> dict[str, str]:
    """Maps every colour group to a stable colour; levels keep their own ramp."""
    if colour_by == 'level':
        return dict(LEVEL_COLOURS)

    groups = sorted({key for points in drawn for key in _colour_keys(points, colour_by)})

    return {group: GROUP_COLOURS[i % len(GROUP_COLOURS)] for i, group in enumerate(groups)}


def _colour_keys(points: LevelPoints, colour_by: ColourBy) -> list[str]:
    """The colour group of each point; a level with no subject or task reads as `unknown`, never as a blank."""
    n = int(np.asarray(points.vectors).shape[0])
    match colour_by:
        case 'level':
            return [points.level] * n
        case 'subject':
            column = points.subjects
        case 'task':
            column = points.tasks
        case unknown:
            raise ValueError(f'Unknown colour_by {unknown!r}; expected level, subject or task.')

    return ['unknown'] * n if column is None else [str(value) for value in column]


def _hover(points: LevelPoints) -> list[str]:
    """One hover string per point, carrying the text it came from, its level, its subject and its task."""
    n = int(np.asarray(points.vectors).shape[0])
    out: list[str] = []
    for i in range(n):
        lines = [f'<b>{_esc(points.labels[i])}</b>', f'level: {points.level}']
        if points.subjects is not None:
            lines.append(f'subject: {_esc(str(points.subjects[i]))}')
        if points.tasks is not None:
            lines.append(f'task: {_esc(str(points.tasks[i]))}')
        out.append('<br>'.join(lines))

    return out


def _axis_labels(method: Method, variance: dict[str, Any], dims: int) -> list[str]:
    """Axis titles that state the variance each principal axis kept, or the method when there is none to state."""
    ratios = variance.get('explained_variance_ratio')
    if ratios is None:
        return [f'{method} {i + 1}' for i in range(dims)]

    return [f'PC{i + 1} ({ratios[i]:.1%})' if i < len(ratios) else f'PC{i + 1}' for i in range(dims)]


def _figure_json(fig: Any) -> FigureJSON:
    """Round-trips a plotly figure through its own encoder, so numpy and non-finite values land JSON-safe."""
    return json.loads(fig.to_json())


def _esc(text: str) -> str:
    """Escapes the HTML plotly renders inside a hover label, so sentence text cannot break the tooltip."""
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _show(value: Any) -> str:
    """Formats a metric for a hover label, printing a dash rather than `None`."""
    return '--' if value is None else f'{float(value):.3f}'
