"""Aggregates parallax transfer cells into `PARALLAX.json`, `PARALLAX.md` and the chamber's data file."""

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from zte.cli.support.io import read_json, write_json
from zte.evaluation.audit.rebaseline import fit_postprocess
from zte.logging_utils import get_logger
from zte.parallax.study import PARALLAX_TASKS, parse_cell_name
from zte.parallax.transfer import linear_cka
from zte.utils.provenance import git_info

_LOG = get_logger('parallax.report')

# k-means over the aligned mean view: 8 clusters is a readable chamber legend, not a scientific claim.
N_CLUSTERS: Final[int] = 8
"""Cluster count for the chamber's point colouring."""

# One point per distinct sentence; the full real SR+NR gallery is exactly this size, so nothing real is cut.
MAX_POINTS: Final[int] = 700
"""Upper bound on chamber points per eval task."""


@dataclass(slots=True, frozen=True, kw_only=True)
class TransferCell:
    """One scored transfer cell as found on disk."""

    train_task: str
    eval_task: str
    seed: int
    path: Path
    report: dict[str, Any]


def load_cells(transfers: str | Path) -> list[TransferCell]:
    """Loads every `<train>_to_<eval>_s<seed>/transfer.json` under a directory.

    Args:
        transfers (str | Path): Directory holding transfer cell directories.

    Returns:
        list[TransferCell]: Cells sorted by (train, eval, seed); unrecognised entries are skipped with a log line.
    """
    root = Path(transfers)
    cells: list[TransferCell] = []
    for entry in sorted(root.iterdir()) if root.is_dir() else []:
        parsed = parse_cell_name(entry.name)
        if parsed is None or not (entry / 'transfer.json').is_file():
            if entry.is_dir():
                _LOG.info('Skipping %s: not a transfer cell.', entry.name)
            continue
        train, eval_task, seed = parsed
        cells.append(
            TransferCell(
                train_task=train,
                eval_task=eval_task,
                seed=seed,
                path=entry,
                report=read_json(entry / 'transfer.json'),
            )
        )

    cells.sort(key=lambda c: (PARALLAX_TASKS.index(c.train_task), PARALLAX_TASKS.index(c.eval_task), c.seed))
    return cells


def build_report(
    transfers: str | Path,
    out: str | Path,
    *,
    n_clusters: int = N_CLUSTERS,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregates a directory of transfer cells and writes the three report files.

    Writes `PARALLAX.json` (the matrix with per-seed summaries and CKA pairs), `PARALLAX.md` (the
    readable account, honest-reading section included) and `CHAMBER_DATA.json` (the chamber's
    pre-reduced 3D geometry -- the chamber renderer only draws it).

    Args:
        transfers (str | Path): Directory of transfer cell directories.
        out (str | Path): Directory the three files are written to.
        n_clusters (int, optional): k for the chamber's k-means colouring. Defaults to 8.
        seed (int, optional): Seed for the k-means initialisation. Defaults to 0.

    Returns:
        dict[str, Any]: The dict written to `PARALLAX.json`.

    Raises:
        ValueError: If the directory holds no transfer cells.
    """
    cells = load_cells(transfers)
    if not cells:
        raise ValueError(f'No transfer cells found under {transfers}.')

    holdouts = sorted({str(c.report.get('holdout')) for c in cells})
    if len(holdouts) > 1:
        _LOG.warning('Cells disagree on the holdout subject (%s); the matrix mixes holdouts.', ', '.join(holdouts))
    holdout = holdouts[0]

    nested: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cell in cells:
        nested.setdefault(cell.train_task, {}).setdefault(cell.eval_task, []).append(_summary(cell.report))

    cka = _cka_pairs(cells)
    parallax: dict[str, Any] = {
        'study': 'parallax',
        'holdout': holdout,
        'tasks': list(PARALLAX_TASKS),
        'seeds': sorted({c.seed for c in cells}),
        'cells': nested,
        'cka': cka,
        'menu_decomposition': _menu_decomposition(cells, holdout),
        'provenance': {'transfers_dir': str(Path(transfers).resolve()), 'n_cells': len(cells), 'git': git_info()},
    }

    out_dir = Path(out)
    write_json(out_dir / 'PARALLAX.json', parallax, default=str)
    (out_dir / 'PARALLAX.md').write_text(render_markdown(parallax), encoding='utf-8')
    chamber = chamber_data(cells, cka, holdout=holdout, n_clusters=n_clusters, seed=seed)
    write_json(out_dir / 'CHAMBER_DATA.json', chamber, default=str)

    _LOG.info('Parallax report written to %s (PARALLAX.json, PARALLAX.md, CHAMBER_DATA.json).', out_dir)
    return parallax


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    """One seed's headline numbers, lifted from a cell's transfer.json."""
    held = report.get('held_out') or {}
    length = report.get('held_out_length_stratified') or {}
    menu = report.get('menu') or {}
    flavor = (menu.get('flavors') or {}).get(menu.get('headline_flavor') or '', {})
    # The enrolled twin of the headline flavor: best cross-subject reading instead of the centroid,
    # reported beside the prototype capacity and never in its place.
    enrolled = (menu.get('flavors') or {}).get(f'{menu.get("headline_flavor") or ""}_enrolled') or {}

    return {
        'seed': report.get('seed'),
        'novel_stimuli': report.get('novel_stimuli'),
        'stimulus_overlap': report.get('stimulus_overlap'),
        'n_queries': report.get('n_queries'),
        'rank_percentile': held.get('rank_percentile'),
        'rank_percentile_ci': held.get('rank_percentile_ci'),
        'top1': held.get('top1'),
        'top1_hits': (
            int(round(float(held['top1']) * float(held['n_queries'])))
            if held.get('top1') is not None and held.get('n_queries')
            else None
        ),
        'top1_p': held.get('top1_p'),
        'top5': held.get('top5'),
        'chance_top1': held.get('chance_top1'),
        'length_stratified_rank_percentile': length.get('rank_percentile'),
        'length_stratified_ci': length.get('rank_percentile_ci'),
        'menu_capacity': flavor.get('capacity'),
        'menu_k2_accuracy': ((flavor.get('per_k') or {}).get('2') or {}).get('accuracy'),
        'menu_gamed': flavor.get('gamed'),
        'menu_capacity_enrolled': enrolled.get('capacity'),
        'menu_k2_enrolled': ((enrolled.get('per_k') or {}).get('2') or {}).get('accuracy'),
        'menu_gamed_enrolled': enrolled.get('gamed'),
        'menu_open_k2': (((menu.get('flavors') or {}).get('open') or {}).get('per_k', {}).get('2') or {}).get(
            'accuracy'
        ),
        'menu_open_gamed': ((menu.get('flavors') or {}).get('open') or {}).get('gamed'),
        'postprocess_fit': report.get('postprocess_fit'),
        'run_name': (report.get('provenance') or {}).get('run_name'),
    }


# --------------------------------------------------------------------------- #
# CKA between vantage points
# --------------------------------------------------------------------------- #


def _load_embeddings(cell_dir: Path) -> dict[str, np.ndarray] | None:
    """Loads a cell's embeddings.npz into plain arrays, or None when the file is absent."""
    path = cell_dir / 'embeddings.npz'
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _aligned_rows(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray] | None:
    """Rows of the same readings in two models' embeddings, matched on (subject, stimulus)."""

    # Content ids are factorised per dataset build, so two independently built cells may number the same
    # sentences differently; the raw text is the stable join key whenever both cells carry it.
    by_text = 'texts' in a and 'texts' in b

    def keys(block: dict[str, np.ndarray]) -> list[tuple[str, str]]:
        labels = (
            block['texts'].astype(str).tolist()
            if by_text
            else [str(i) for i in block['content_ids'].astype(int).tolist()]
        )
        return list(zip(block['subjects'].astype(str).tolist(), labels, strict=True))

    b_index: dict[tuple[str, str], int] = {}
    for j, key in enumerate(keys(b)):
        b_index.setdefault(key, j)

    rows_a: list[int] = []
    rows_b: list[int] = []
    seen: set[tuple[str, str]] = set()
    for i, key in enumerate(keys(a)):
        match = b_index.get(key)
        if match is None or key in seen:
            continue
        seen.add(key)
        rows_a.append(i)
        rows_b.append(match)

    if len(rows_a) < 2:
        return None

    return a['sent_emb'][rows_a], b['sent_emb'][rows_b]


def _pair_eval_task(a: str, b: str, cells: list[TransferCell]) -> str | None:
    """Picks the eval task two models are compared on, preferring the task neither trained on."""
    covered = {(c.train_task, c.eval_task) for c in cells}
    neutral = [t for t in PARALLAX_TASKS if t not in (a, b) and (a, t) in covered and (b, t) in covered]
    if neutral:
        return neutral[0]

    shared = [t for t in PARALLAX_TASKS if (a, t) in covered and (b, t) in covered]
    return shared[0] if shared else None


def _cka_pairs(cells: list[TransferCell]) -> dict[str, Any]:
    """Linear CKA per model pair on shared eval readings, one value per seed both models scored."""
    model_tasks = [t for t in PARALLAX_TASKS if any(c.train_task == t for c in cells)]
    by_key = {(c.train_task, c.eval_task, c.seed): c for c in cells}

    pairs: dict[str, Any] = {}
    for a, b in itertools.combinations(model_tasks, 2):
        eval_task = _pair_eval_task(a, b, cells)
        if eval_task is None:
            continue

        seeds = sorted(
            {c.seed for c in cells if (c.train_task, c.eval_task) == (a, eval_task)}
            & {c.seed for c in cells if (c.train_task, c.eval_task) == (b, eval_task)}
        )
        per_seed: list[float] = []
        for s in seeds:
            emb_a = _load_embeddings(by_key[(a, eval_task, s)].path)
            emb_b = _load_embeddings(by_key[(b, eval_task, s)].path)
            if emb_a is None or emb_b is None:
                continue
            aligned = _aligned_rows(emb_a, emb_b)
            if aligned is None:
                continue
            per_seed.append(linear_cka(aligned[0], aligned[1]))

        if per_seed:
            pairs[f'{a}|{b}'] = {'eval_task': eval_task, 'per_seed': per_seed}

    return pairs


# The retrieval percentile ranks the first of ~11 cross-subject positives (a best-of-many statistic) while
# the certified menu scores one prototype in an exact-length pool with ties losing; recomputing 2-way
# accuracy under {prototype, best reading} x {tol 0, tol 1} names which factor carries the gap.
def _menu_decomposition(cells: list[TransferCell], holdout: str) -> dict[str, Any]:
    """2-way accuracy per scoring rule and length tolerance on the diagonal cells; feeds no verdict."""
    out: dict[str, Any] = {}
    for task in PARALLAX_TASKS:
        per_seed: dict[str, list[float]] = {}
        for cell in cells:
            if cell.train_task != task or cell.eval_task != task:
                continue
            block = _load_embeddings(cell.path)
            if block is None:
                continue
            scores = _two_way_grid(block, holdout)
            for name, value in scores.items():
                per_seed.setdefault(name, []).append(value)

        if per_seed:
            out[task] = {name: float(np.mean(values)) for name, values in per_seed.items()}

    return out


def _two_way_grid(block: dict[str, np.ndarray], holdout: str) -> dict[str, float]:
    """One cell's 2-way accuracies under {prototype, best reading} x {tol 0, tol 1}, ties losing."""
    subjects = block['subjects'].astype(str)
    ids = block['content_ids'].astype(int)
    lengths = np.asarray(block['n_words'], dtype=np.float64)
    mask = subjects != holdout

    emb = fit_postprocess(np.asarray(block['sent_emb'], dtype=np.float32)[mask])(block['sent_emb'])
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    sent_ids = sorted({int(c) for c in ids[mask]})
    rows_of = {cid: np.where((ids == cid) & mask)[0] for cid in sent_ids}
    protos = np.stack([emb[rows_of[cid]].mean(axis=0) for cid in sent_ids])
    protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-12)
    sent_len = np.asarray([float(np.median(lengths[rows_of[cid]])) for cid in sent_ids])
    index_of = {cid: i for i, cid in enumerate(sent_ids)}

    queries = np.where((subjects == holdout) & np.isin(ids, np.asarray(sent_ids)))[0]
    grid: dict[str, list[float]] = {f'{rule}_tol{tol}': [] for rule in ('prototype', 'best_reading') for tol in (0, 1)}
    for q in queries:
        target = index_of[int(ids[q])]
        proto_scores = protos @ emb[q]
        best_scores = np.asarray([float(np.max(emb[rows_of[cid]] @ emb[q])) for cid in sent_ids])
        for tol in (0, 1):
            pool = (np.abs(sent_len - sent_len[target]) <= tol) & (np.arange(len(sent_ids)) != target)
            if not pool.any():
                continue
            for rule, scores in (('prototype', proto_scores), ('best_reading', best_scores)):
                grid[f'{rule}_tol{tol}'].append(float(np.mean(scores[pool] < scores[target])))

    return {name: float(np.mean(values)) for name, values in grid.items() if values}


# --------------------------------------------------------------------------- #
# Chamber geometry: PCA to 3D, Procrustes across views, k-means colouring
# --------------------------------------------------------------------------- #


def _pca3(x: np.ndarray) -> np.ndarray:
    """Top-3 principal-component coordinates, zero-padded when the space has fewer directions."""
    centred = np.asarray(x, dtype=np.float64)
    centred = centred - centred.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    coords = u[:, :3] * s[:3]
    if coords.shape[1] < 3:
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))

    return coords


def _procrustes(view: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Best orthogonal rotation of `view` onto `ref`; both are centred and unit-Frobenius already."""
    u, _, vt = np.linalg.svd(view.T @ ref, full_matrices=False)

    return view @ (u @ vt)


def _normalise_view(view: np.ndarray) -> np.ndarray:
    """Centres a 3D view and scales it to unit Frobenius norm so views overlay at a common scale."""
    centred = view - view.mean(axis=0, keepdims=True)
    norm = float(np.linalg.norm(centred))

    return centred / norm if norm > 0 else centred


def _kmeans_labels(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Plain seeded Lloyd k-means over rows of `x`; returns labels only, deterministic for a fixed seed."""
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(x)))
    centres = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.full(len(x), -1, dtype=int)
    for _ in range(100):
        distances = ((x[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        updated = distances.argmin(axis=1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centres[j] = members.mean(axis=0)

    return labels


def _holdout_rank_percentiles(block: dict[str, np.ndarray], holdout: str) -> dict[int, float]:
    """Mean full-gallery rank percentile per stimulus for the holdout's readings, train-fitted post-processing."""
    emb = np.asarray(block['sent_emb'], dtype=np.float32)
    subjects = block['subjects'].astype(str)
    ids = block['content_ids'].astype(int)
    mask = subjects != holdout
    if int(mask.sum()) < 2 or int((~mask).sum()) == 0:
        return {}

    processed = fit_postprocess(emb[mask])(emb)
    processed = processed / (np.linalg.norm(processed, axis=1, keepdims=True) + 1e-12)

    collected: dict[int, list[float]] = {}
    for i in np.flatnonzero(~mask):
        cand = np.flatnonzero(subjects != subjects[i])
        same = ids[cand] == ids[i]
        if not same.any():
            continue
        sims = processed[cand] @ processed[i]
        order = np.argsort(-sims)
        rank = int(np.argmax(same[order])) + 1
        collected.setdefault(int(ids[i]), []).append(1.0 - (rank - 1) / max(cand.size - 1, 1))

    return {cid: float(np.mean(values)) for cid, values in collected.items()}


def _points_for(
    eval_task: str,
    cells: list[TransferCell],
    holdout: str,
    n_clusters: int,
    seed: int,
) -> list[dict[str, Any]] | None:
    """Builds one eval task's chamber points: aligned 3D views of every model's sentence prototypes."""
    relevant = [c for c in cells if c.eval_task == eval_task]
    if not relevant:
        return None

    # One seed for the whole picture: the smallest seed every model of this eval task shares.
    seed_sets = {task: {c.seed for c in relevant if c.train_task == task} for task in {c.train_task for c in relevant}}
    common = set.intersection(*seed_sets.values())
    model_tasks: list[str]
    if common:
        cell_seed = min(common)
        model_tasks = [t for t in PARALLAX_TASKS if t in seed_sets]
    else:
        cell_seed = min(c.seed for c in relevant)
        model_tasks = [t for t in PARALLAX_TASKS if t in seed_sets and cell_seed in seed_sets[t]]
        _LOG.warning('Eval task %s: models share no seed; the chamber view uses seed %d only.', eval_task, cell_seed)

    blocks: dict[str, dict[str, np.ndarray]] = {}
    for task in model_tasks:
        cell = next(c for c in relevant if c.train_task == task and c.seed == cell_seed)
        block = _load_embeddings(cell.path)
        if block is not None:
            blocks[task] = block
    if not blocks:
        return None

    # Prototypes: cross-subject mean of unit rows per stimulus, holdout excluded; ids common to all models.
    ids_per_model: list[set[int]] = []
    for block in blocks.values():
        mask = block['subjects'].astype(str) != holdout
        ids_per_model.append(set(block['content_ids'][mask].astype(int).tolist()))
    common_ids = sorted(set.intersection(*ids_per_model))[:MAX_POINTS]
    if len(common_ids) < 2:
        return None

    # Ids are factorised per dataset build; a shared id must name the same sentence in every block, or the
    # chamber would confidently overlay different sentences. Text disagreement is a refusal, never a guess.
    text_blocks = {task: block['texts'].astype(str) for task, block in blocks.items() if 'texts' in block}
    if len(text_blocks) == len(blocks):
        mismatched = 0
        for cid in common_ids:
            names = set()
            for task, block in blocks.items():
                rows = np.where(block['content_ids'].astype(int) == cid)[0]
                if rows.size:
                    names.add(str(text_blocks[task][rows[0]]))
            mismatched += int(len(names) > 1)
        if mismatched:
            raise ValueError(
                f'{mismatched} shared stimulus ids name different sentences across the models for eval task '
                f'{eval_task}; the cells were built from diverging dataset configs and cannot be overlaid.'
            )

    views: dict[str, np.ndarray] = {}
    ranks: dict[str, dict[int, float]] = {}
    for task, block in blocks.items():
        emb = np.asarray(block['sent_emb'], dtype=np.float64)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
        subjects = block['subjects'].astype(str)
        ids = block['content_ids'].astype(int)
        protos = np.stack([emb[(ids == cid) & (subjects != holdout)].mean(axis=0) for cid in common_ids])
        views[task] = _normalise_view(_pca3(protos))
        ranks[task] = _holdout_rank_percentiles(block, holdout)

    # Procrustes: rotate every view onto the first model's, so "what stays fixed" is visible by overlay.
    reference = views[model_tasks[0]]
    aligned = {task: (view if task == model_tasks[0] else _procrustes(view, reference)) for task, view in views.items()}

    mean_view = np.mean(np.stack(list(aligned.values())), axis=0)
    clusters = _kmeans_labels(mean_view, n_clusters, seed)

    first = blocks[model_tasks[0]]
    texts = first.get('texts')
    id_rows = {int(cid): int(np.argmax(first['content_ids'].astype(int) == cid)) for cid in common_ids}

    points: list[dict[str, Any]] = []
    for row, cid in enumerate(common_ids):
        id_mask = first['content_ids'].astype(int) == cid
        points.append(
            {
                'text': str(texts[id_rows[cid]]) if texts is not None else f'sentence {cid}',
                'cluster': int(clusters[row]),
                'n_words': int(round(float(np.median(first['n_words'][id_mask])))),
                'views': {task: [float(v) for v in aligned[task][row]] for task in aligned},
                'rank_percentile': {task: ranks[task].get(cid) for task in aligned},
            }
        )

    return points


# --------------------------------------------------------------------------- #
# CHAMBER_DATA.json
# --------------------------------------------------------------------------- #


def chamber_data(
    cells: list[TransferCell],
    cka: dict[str, Any],
    *,
    holdout: str,
    n_clusters: int = N_CLUSTERS,
    seed: int = 0,
) -> dict[str, Any]:
    """Builds the chamber's data file: pre-reduced geometry plus the pooled matrix numbers.

    The chamber page only renders this dict -- every reduction (PCA to 3D, Procrustes alignment,
    k-means colouring) happens here, in numpy.

    Args:
        cells (list[TransferCell]): The loaded transfer cells.
        cka (dict[str, Any]): The per-pair CKA block from the PARALLAX aggregation.
        holdout (str): The held-out subject code.
        n_clusters (int, optional): k for the k-means colouring. Defaults to 8.
        seed (int, optional): Seed for the k-means initialisation. Defaults to 0.

    Returns:
        dict[str, Any]: `{'holdout', 'tasks', 'points', 'transfer', 'capacity', 'cka'}`.
    """
    points: dict[str, list[dict[str, Any]]] = {}
    for eval_task in PARALLAX_TASKS:
        block = _points_for(eval_task, cells, holdout, n_clusters, seed)
        if block:
            points[eval_task] = block

    nested: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for cell in cells:
        nested.setdefault(cell.train_task, {}).setdefault(cell.eval_task, []).append(_summary(cell.report))

    return {
        'holdout': holdout,
        'tasks': list(PARALLAX_TASKS),
        'points': points,
        'transfer': _pooled_transfer(nested),
        'capacity': _pooled_capacity(nested),
        'cka': {pair: float(np.mean(block['per_seed'])) for pair, block in cka.items() if block['per_seed']},
    }


def _pooled_transfer(nested: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Per-cell numbers pooled across seeds: mean rank percentile, mean CI bounds, mean Top-1."""
    pooled: dict[str, Any] = {}
    for train, evals in nested.items():
        for eval_task, summaries in evals.items():
            rps = [s['rank_percentile'] for s in summaries if s['rank_percentile'] is not None]
            cis = [s['rank_percentile_ci'] for s in summaries if s['rank_percentile_ci']]
            tops = [s['top1'] for s in summaries if s['top1'] is not None]
            chances = [s['chance_top1'] for s in summaries if s['chance_top1'] is not None]
            pooled.setdefault(train, {})[eval_task] = {
                'rank_percentile': float(np.mean(rps)) if rps else None,
                'ci': [float(np.mean([c[1] for c in cis])), float(np.mean([c[2] for c in cis]))] if cis else None,
                'top1': float(np.mean(tops)) if tops else None,
                'chance': float(np.mean(chances)) if chances else None,
                'n_seeds': len(summaries),
            }

    return pooled


def _pooled_capacity(nested: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    """Diagonal menu capacity per arm, prototype and enrolled: smallest certified K and mean 2-way accuracy."""
    capacity: dict[str, Any] = {}
    for train, evals in nested.items():
        summaries = evals.get(train)
        if not summaries:
            continue
        caps = [s['menu_capacity'] for s in summaries]
        k2 = [s['menu_k2_accuracy'] for s in summaries if s['menu_k2_accuracy'] is not None]
        enrolled_caps = [s['menu_capacity_enrolled'] for s in summaries]
        enrolled_k2 = [s['menu_k2_enrolled'] for s in summaries if s['menu_k2_enrolled'] is not None]
        open_k2 = [value for s in summaries if (value := s.get('menu_open_k2')) is not None]
        capacity[train] = {
            'k_at_target': int(min(caps)) if caps and all(c is not None for c in caps) else None,
            'k2_accuracy': float(np.mean(k2)) if k2 else None,
            # Any gamed seed taints the pooled arm: a disqualification must never average away.
            'gamed': bool(any(s.get('menu_gamed') or s.get('menu_gamed_enrolled') for s in summaries)),
            'enrolled_k_at_target': (
                int(min(enrolled_caps)) if enrolled_caps and all(c is not None for c in enrolled_caps) else None
            ),
            'enrolled_k2_accuracy': float(np.mean(enrolled_k2)) if enrolled_k2 else None,
            'open': {
                'k2_accuracy': float(np.mean(open_k2)) if open_k2 else None,
                'gamed': bool(any(s.get('menu_open_gamed') for s in summaries)),
            },
        }

    return capacity


# --------------------------------------------------------------------------- #
# PARALLAX.md
# --------------------------------------------------------------------------- #


def render_markdown(parallax: dict[str, Any]) -> str:
    """Renders `PARALLAX.md`: the 3x3 matrix, per-seed cells, capacities, CKA and the honest reading.

    Args:
        parallax (dict[str, Any]): The dict written to `PARALLAX.json`.

    Returns:
        str: A Markdown document.
    """
    nested: dict[str, dict[str, list[dict[str, Any]]]] = parallax.get('cells') or {}
    pooled = _pooled_transfer(nested)
    holdout = parallax.get('holdout')
    seeds = parallax.get('seeds') or []

    lines = [
        '# Parallax -- three vantage points on the same minds',
        '',
        f'Three independent encoders, one per ZuCo task, each scored on every task. Holdout subject '
        f'`{holdout}`; seeds {", ".join(str(s) for s in seeds)}. Every number is closed-set retrieval '
        f'(rank percentile, chance 0.5) with train-fitted post-processing.',
        '',
        '## The transfer matrix -- rank percentile, pooled across seeds',
        '',
        'Rows are the training task, columns the eval task. Off-diagonal cells marked *novel* face a '
        'never-seen subject reading never-seen stimuli. The parenthesised interval is the mean of the '
        'per-seed CI bounds, not a pooled bootstrap -- the per-seed table below carries the real CIs.',
        '',
        '| train \\ eval | ' + ' | '.join(PARALLAX_TASKS) + ' |',
        '| --- | ' + ' | '.join('---' for _ in PARALLAX_TASKS) + ' |',
    ]
    for train in PARALLAX_TASKS:
        row = [f'**{train}**']
        for eval_task in PARALLAX_TASKS:
            cell = (pooled.get(train) or {}).get(eval_task)
            if not cell or cell['rank_percentile'] is None:
                row.append('—')
                continue
            ci = cell.get('ci')
            ci_text = f' ({ci[0]:.4f}–{ci[1]:.4f})' if ci else ''
            summaries = nested[train][eval_task]
            novel = ' *novel*' if any(s.get('novel_stimuli') for s in summaries) else ''
            row.append(f'{cell["rank_percentile"]:.4f}{ci_text}{novel}')
        lines.append('| ' + ' | '.join(row) + ' |')

    lines += [
        '',
        '## Per-seed cells',
        '',
        '| train | eval | seed | rank percentile (95% CI) | length-stratified | Top-1 hits (p) | chance Top-1 | novel stimuli | n |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for train in PARALLAX_TASKS:
        for eval_task in PARALLAX_TASKS:
            for s in (nested.get(train) or {}).get(eval_task) or []:
                ci = s.get('rank_percentile_ci')
                ci_text = f'{ci[0]:.4f} ({ci[1]:.4f}–{ci[2]:.4f})' if ci else '—'
                length = s.get('length_stratified_rank_percentile')
                hits = s.get('top1_hits')
                p = s.get('top1_p')
                hits_text = (
                    '—' if hits is None else f'{hits}/{s.get("n_queries")}' + (f' (p={p:.2g})' if p is not None else '')
                )
                lines.append(
                    f'| {train} | {eval_task} | {s.get("seed")} | {ci_text} '
                    f'| {"—" if length is None else format(length, ".4f")} '
                    f'| {hits_text} '
                    f'| {"—" if s.get("chance_top1") is None else format(s["chance_top1"], ".4f")} '
                    f'| {s.get("novel_stimuli")} | {s.get("n_queries")} |'
                )

    capacity = _pooled_capacity(nested)
    if capacity:
        lines += ['', '## Menu capacity (in-task diagonal)', '']
        for arm, block in capacity.items():
            k = block.get('k_at_target')
            k2 = block.get('k2_accuracy')
            gamed_note = ' ⚠ length-gamed — disqualified.' if block.get('gamed') else ''
            lines.append(
                f'- `{arm}` prototype: certified capacity K = {"none" if k is None else k}, '
                f'2-way accuracy {"—" if k2 is None else format(k2, ".4f")}.{gamed_note}'
            )
            ek = block.get('enrolled_k_at_target')
            ek2 = block.get('enrolled_k2_accuracy')
            if ek2 is None and ek is None:
                lines.append(f'- `{arm}` enrolled: not measured (the cells predate the enrolled flavor).')
            else:
                lines.append(
                    f'- `{arm}` enrolled (best cross-subject reading): certified capacity K = '
                    f'{"none" if ek is None else ek}, 2-way accuracy {"—" if ek2 is None else format(ek2, ".4f")}.'
                )
            open_block = block.get('open') or {}
            if open_block.get('k2_accuracy') is not None:
                badge = ' ⚠ length-gamed — disqualified' if open_block.get('gamed') else ''
                lines.append(
                    f'- `{arm}` open pool (diagnostic, never certified): 2-way accuracy '
                    f'{format(open_block["k2_accuracy"], ".4f")}.{badge}'
                )

    decomposition = parallax.get('menu_decomposition') or {}
    if decomposition:

        def _cell(value: Any) -> str:
            return '—' if value is None else format(float(value), '.4f')

        lines += [
            '',
            '## Why the menu and the percentile disagree -- 2-way decomposition (diagnostic)',
            '',
            'The retrieval percentile ranks the first of ~11 cross-subject readings of the true sentence -- a '
            'best-of-many statistic -- while the certified menu scores one prototype inside an exact-length pool '
            'with ties losing. The grid isolates each factor on the diagonal cells (mean across seeds). '
            'Diagnostic only: no capacity or verdict reads it.',
            '',
            '| task | prototype, exact len | prototype, ±1 | best reading, exact len | best reading, ±1 |',
            '| --- | --- | --- | --- | --- |',
        ]
        for task, row in decomposition.items():
            lines.append(
                f'| {task} | {_cell(row.get("prototype_tol0"))} | {_cell(row.get("prototype_tol1"))} '
                f'| {_cell(row.get("best_reading_tol0"))} | {_cell(row.get("best_reading_tol1"))} |'
            )

    cka = parallax.get('cka') or {}
    if cka:
        lines += [
            '',
            '## CKA between vantage points',
            '',
            "Linear CKA between two models' embeddings of the same readings; what stays fixed across "
            'vantage points is the candidate task-invariant code.',
            '',
            '| pair | eval task | per-seed CKA |',
            '| --- | --- | --- |',
        ]
        for pair, block in cka.items():
            values = ', '.join(f'{v:.4f}' for v in block.get('per_seed') or [])
            lines.append(f'| {pair} | {block.get("eval_task")} | {values} |')

    lines += [
        '',
        '## Reading this honestly',
        '',
        f'- Every cell faces a never-seen subject: `{holdout}` trained no model in this study.',
        "- Off-diagonal cells additionally face never-seen stimuli -- the tasks' sentence sets are "
        'disjoint, and each cell records the measured `stimulus_overlap`. A cell whose overlap is not '
        'zero is flagged `novel_stimuli: false` and cannot carry the never-seen-stimuli claim.',
        '- Chance rank percentile is 0.5. A CI that brackets 0.5 is a null result, and a null here is '
        'a finding, reported plainly -- it says the task-specific code did not transfer.',
        "- The readout is closed-set retrieval over the eval task's gallery. Free generation is not a "
        'parallax deliverable, and no number in this report may be quoted as generation.',
        '- Post-processing is fitted on non-holdout subjects of the eval task only '
        '(`postprocess_fit` travels in every cell).',
        '',
    ]

    return '\n'.join(lines)
