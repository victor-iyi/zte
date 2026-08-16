"""The menu-capacity audit: the largest K-way closed set the embedding serves at a target accuracy."""

from math import comb
from typing import Any, Final

import numpy as np

from zte.evaluation.audit.scoreboard import _bootstrap_ci

# K = 1 is degenerate (always right), so the sweep starts at the smallest real decision.
DEFAULT_MENU_KS: Final[tuple[int, ...]] = (2, 4, 8, 16, 32, 64)
"""Menu sizes the audit sweeps."""

# Exact match (tol 0) is load-bearing: at tol 1 the true candidate is systematically the unique best
# length match inside its own stratum, so a pure length code beats chance and the guarantee is gone.
HEADLINE_TOL: Final[int] = 0
"""Word-count tolerance of the certified flavors -- exact stimulus-level match, never widened."""

SENSITIVITY_TOLS: Final[tuple[int, ...]] = (1, 2)
"""Widened tolerances reported as labelled diagnostics; no verdict or capacity may read them."""


def menu_report(
    sent_emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    n_words: np.ndarray,
    *,
    tasks: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    ks: tuple[int, ...] = DEFAULT_MENU_KS,
    target: float = 0.8,
    postprocess: bool = True,
    whiten: bool = True,
    n_top: int = 1,
    n_boot: int = 2000,
    n_perm: int = 500,
    seed: int = 0,
) -> dict[str, Any] | None:
    """K-way closed-set accuracy on the held-out subject, and the largest K certified at the target.

    A K-way menu asks: given one held-out reading and K candidate sentences (the true one plus K-1
    distractors), does the embedding rank the true sentence first? Each accuracy is the exact
    expectation over uniformly drawn distractors -- for a query whose true sentence strictly beats
    `b` of the `m` pool sentences, the win probability is `C(b, K-1) / C(m, K-1)` -- so there is no
    sampling seed and chance is exactly `1/K`. Ties count as losses, so a constant embedding scores
    zero rather than chance.

    The headline flavor is `length_task_matched` (falling back to `length_matched` when no task
    labels are given): distractors share the query's task and its exact stimulus-level word count,
    so a hit can be neither a length nor a task-register shortcut. `open` draws from the full
    gallery, where using length is legitimate, as it is in a deployed menu. Widened tolerances are
    reported under `sensitivity` and feed nothing. Each certified flavor also carries a built-in
    length-oracle null (`gamed` flips true if word count alone escapes chance inside the pool) and a
    permutation p per K (the true label reassigned uniformly within the candidate set).

    Args:
        sent_emb (np.ndarray): Sentence embeddings `(n, d)`, before any post-processing.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.
        subjects (np.ndarray): Subject code per reading `(n,)`.
        holdout (str): The held-out subject code, whose readings are the queries.
        n_words (np.ndarray): Word count per reading `(n,)`.
        tasks (np.ndarray | None, optional): Task label per reading `(n,)`; enables the
            task-matched headline flavor. Defaults to None.
        train_mask (np.ndarray | None, optional): Boolean `(n,)` marking rows the post-processing and
            prototypes may use. Defaults to every non-holdout row.
        ks (tuple[int, ...], optional): Menu sizes to sweep. Defaults to (2, 4, 8, 16, 32, 64).
        target (float, optional): Accuracy a menu size must clear to be certified. Defaults to 0.8.
        postprocess (bool, optional): Apply train-fitted whitening + all-but-the-top. Defaults to True.
        whiten (bool, optional): Whether the fitted transform whitens. Defaults to True.
        n_top (int, optional): Leading directions removed by all-but-the-top. Defaults to 1.
        n_boot (int, optional): Bootstrap resamples behind each accuracy CI. Defaults to 2000.
        n_perm (int, optional): Label permutations behind each per-K p-value. Defaults to 500.
        seed (int, optional): Bootstrap and permutation seed. Defaults to 0.

    Returns:
        dict | None: `{'postprocess_fit', 'headline_flavor', 'target', 'holdout', 'n_queries',
            'n_gallery', 'dropped_no_prototype', 'tie_policy', 'flavors', 'sensitivity'}`; `None`
            when there is nothing to score.
    """
    from zte.evaluation.audit.rebaseline import fit_postprocess

    subjects = np.asarray(subjects)
    content_ids = np.asarray(content_ids)
    lengths = np.asarray(n_words, dtype=np.float64).ravel()
    task_arr = None if tasks is None else np.asarray(tasks)
    mask = np.asarray(subjects != holdout) if train_mask is None else np.asarray(train_mask, dtype=bool)
    q_mask = subjects == holdout
    if int(q_mask.sum()) == 0 or int(mask.sum()) < 2:
        return None

    emb = np.asarray(sent_emb, dtype=np.float32)
    if postprocess:
        emb = fit_postprocess(emb[mask], whiten=whiten, n_top=n_top)(emb)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    # Prototypes are training-subject centroids only -- "same thought, other brains". The held-out
    # subject's readings are the queries and must never enrol their own sentence's reference.
    proto_ids: list[Any] = []
    proto_rows: list[np.ndarray] = []
    proto_lengths: list[float] = []
    proto_tasks: list[Any] = []
    for cid in np.unique(content_ids):
        rows = np.where((content_ids == cid) & mask)[0]
        if rows.size == 0:
            continue
        centroid = emb[rows].mean(axis=0)
        proto_ids.append(cid)
        proto_rows.append(centroid / (np.linalg.norm(centroid) + 1e-12))
        # Stimulus-level word count: the median over training readings, immune to any one reading's omissions.
        proto_lengths.append(float(np.median(lengths[rows])))
        proto_tasks.append(task_arr[rows[0]] if task_arr is not None else None)

    if len(proto_ids) < 2:
        return None

    protos = np.stack(proto_rows)
    proto_id_arr = np.asarray(proto_ids)
    proto_len = np.asarray(proto_lengths, dtype=np.float64)
    proto_task = np.asarray(proto_tasks) if task_arr is not None else None

    q_idx = np.where(q_mask & np.isin(content_ids, proto_id_arr))[0]
    dropped = int(q_mask.sum()) - int(q_idx.size)
    if q_idx.size == 0:
        return None

    sims = emb[q_idx] @ protos.T  # (n_queries, n_gallery) -- gallery is distinct sentences, not readings

    flavor_specs: list[tuple[str, int | None, bool]] = [('length_matched', HEADLINE_TOL, False), ('open', None, False)]
    if proto_task is not None:
        flavor_specs.insert(0, ('length_task_matched', HEADLINE_TOL, True))
    sensitivity_specs = [
        (
            f'{"length_task_matched" if proto_task is not None else "length_matched"}_tol{tol}',
            tol,
            proto_task is not None,
        )
        for tol in SENSITIVITY_TOLS
    ]

    rng = np.random.default_rng(seed + 1)
    flavors: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for certified, specs in ((True, flavor_specs), (False, sensitivity_specs)):
        for name, tol, task_matched in specs:
            block = _score_flavor(
                sims,
                q_idx,
                content_ids,
                lengths,
                proto_id_arr,
                proto_len,
                proto_task,
                tol=tol,
                task_matched=task_matched,
                ks=ks,
                target=target,
                n_boot=n_boot,
                n_perm=n_perm if certified else 0,
                rng=rng,
                with_oracle=certified,
            )
            (flavors if certified else sensitivity)[name] = block

    return {
        'postprocess_fit': 'train split' if postprocess else 'none',
        'headline_flavor': 'length_task_matched' if proto_task is not None else 'length_matched',
        'headline_tol': HEADLINE_TOL,
        'target': float(target),
        'holdout': str(holdout),
        'n_queries': int(q_idx.size),
        'n_gallery': int(protos.shape[0]),
        'dropped_no_prototype': dropped,
        'tie_policy': 'ties lose',
        'flavors': flavors,
        'sensitivity': sensitivity,
    }


def _score_flavor(
    sims: np.ndarray,
    q_idx: np.ndarray,
    content_ids: np.ndarray,
    lengths: np.ndarray,
    proto_id_arr: np.ndarray,
    proto_len: np.ndarray,
    proto_task: np.ndarray | None,
    *,
    tol: int | None,
    task_matched: bool,
    ks: tuple[int, ...],
    target: float,
    n_boot: int,
    n_perm: int,
    rng: np.random.Generator,
    with_oracle: bool,
) -> dict[str, Any]:
    """Scores one distractor-pool definition: per-K accuracy, capacity, permutation p, oracle null."""
    # Per query: the candidate set is the pool plus the true sentence; `beaten[c]` counts, for each
    # candidate as pseudo-true, how many of the others it strictly beats. The observed statistic reads
    # the true row; the permutation null redraws the true label uniformly from the same set.
    per_query: list[dict[str, Any]] = []
    for row, i in enumerate(q_idx):
        t = int(np.where(proto_id_arr == content_ids[i])[0][0])
        candidates = np.ones(proto_id_arr.shape[0], dtype=bool)
        if tol is not None:
            candidates &= np.abs(proto_len - proto_len[t]) <= tol
        if task_matched and proto_task is not None:
            candidates &= proto_task == proto_task[t]
        candidates[t] = True
        cand_idx = np.where(candidates)[0]

        cand_sims = sims[row, cand_idx]
        order = np.sort(cand_sims)
        beaten = np.searchsorted(order, cand_sims, side='left')  # strict wins within the candidate set
        # The length-only adversary: closer stimulus word count wins; must sit at chance in this pool.
        oracle_scores = -np.abs(proto_len[cand_idx] - lengths[i])
        oracle_beaten = np.searchsorted(np.sort(oracle_scores), oracle_scores, side='left')
        true_slot = int(np.where(cand_idx == t)[0][0])
        per_query.append(
            {
                'beaten': beaten,
                'true_slot': true_slot,
                'oracle_beaten': oracle_beaten,
                'm': int(cand_idx.size) - 1,
            }
        )

    per_k: dict[str, Any] = {}
    capacity: int | None = None
    capacity_point: int | None = None
    for k in ks:
        rows = [q for q in per_query if q['m'] >= k - 1]
        if not rows:
            per_k[str(k)] = {'accuracy': None, 'ci': None, 'chance': 1.0 / k, 'n_queries': 0, 'perm_p': None}
            continue

        probs = np.array([_win_prob(q['beaten'][q['true_slot']], q['m'], k) for q in rows], dtype=np.float64)
        mean, lo, hi = _bootstrap_ci(probs, n_boot=n_boot, seed=int(rng.integers(2**31)))
        perm_p: float | None = None
        if n_perm > 0:
            # Exact relabelling null: a pseudo-true drawn uniformly from the candidate set, using each
            # candidate's own win probability -- no re-scoring, so the null shares every artefact of the data.
            pools = [np.array([_win_prob(b, q['m'], k) for b in q['beaten']], dtype=np.float64) for q in rows]
            flat = np.concatenate(pools)
            sizes = np.array([pool.size for pool in pools])
            offsets = np.concatenate(([0], np.cumsum(sizes[:-1])))
            picks = rng.integers(0, sizes, size=(n_perm, len(pools)))
            draws = flat[offsets + picks].mean(axis=1)
            perm_p = float((np.sum(draws >= mean) + 1) / (n_perm + 1))

        per_k[str(k)] = {
            'accuracy': mean,
            'ci': (mean, lo, hi),
            'chance': 1.0 / k,
            'n_queries': len(rows),
            'perm_p': perm_p,
        }
        if lo >= target and (perm_p is None or perm_p < 0.05):
            capacity = k
        if mean >= target:
            capacity_point = k

    block: dict[str, Any] = {
        'per_k': per_k,
        'capacity': capacity,
        'capacity_point': capacity_point,
        'tol': tol,
        'task_matched': bool(task_matched and proto_task is not None),
    }
    if with_oracle:
        oracle_rows = [q for q in per_query if q['m'] >= 1]
        if oracle_rows:
            oracle_probs = np.array(
                [_win_prob(q['oracle_beaten'][q['true_slot']], q['m'], 2) for q in oracle_rows], dtype=np.float64
            )
            o_mean, o_lo, o_hi = _bootstrap_ci(oracle_probs, n_boot=n_boot, seed=int(rng.integers(2**31)))
            block['length_oracle_2way'] = {'accuracy': o_mean, 'ci': (o_mean, o_lo, o_hi)}
            block['gamed'] = bool(o_lo > 0.5)

    return block


def _win_prob(beaten: int, m: int, k: int) -> float:
    """Exact probability the pseudo-true wins a K-way menu after strictly beating `beaten` of `m` others."""
    return comb(int(beaten), k - 1) / comb(int(m), k - 1)


def menu_markdown_lines(menu: dict[str, Any]) -> list[str]:
    """The Markdown block for the menu-capacity audit, appended to `rebaseline.md`.

    Args:
        menu (dict[str, Any]): The dict from `menu_report`.

    Returns:
        list[str]: Markdown lines, ending with a blank line.
    """
    target = menu.get('target', 0.8)
    headline = menu.get('headline_flavor')
    flavors: dict[str, Any] = menu.get('flavors') or {}
    lines = [
        '## Menu capacity -- K-way closed-set accuracy',
        '',
        f'A K-way menu picks the sentence the subject actually read out of K candidates (sentence '
        f'prototypes from training subjects only, post-processing `{menu.get("postprocess_fit")}`, '
        f'ties lose). The headline flavor `{headline}` matches distractors on exact stimulus-level '
        f'word count{" and task" if headline == "length_task_matched" else ""}, so a hit can be '
        f'neither a length nor a task-register shortcut; `open` draws from the full gallery, where '
        f'length may legitimately help, as it would in deployment. A certified size needs CI-low ≥ '
        f'{target:.0%} and permutation p < 0.05.',
        '',
        '| flavor | K | chance | accuracy (95% CI) | perm p | n |',
        '| --- | --- | --- | --- | --- | --- |',
    ]

    for name, block in flavors.items():
        for key, cell in (block.get('per_k') or {}).items():
            if cell.get('accuracy') is None:
                lines.append(f'| {name} | {key} | {1.0 / int(key):.4f} | — | — | 0 |')
                continue
            ci = cell['ci']
            p = cell.get('perm_p')
            lines.append(
                f'| {name} | {key} | {1.0 / int(key):.4f} '
                f'| {cell["accuracy"]:.4f} ({ci[1]:.4f}–{ci[2]:.4f}) '
                f'| {"—" if p is None else format(p, ".4f")} | {cell.get("n_queries", 0)} |'
            )

    capacities = ', '.join(
        f'{name} **K = {_fmt_k(block.get("capacity"))}** (point {_fmt_k(block.get("capacity_point"))})'
        for name, block in flavors.items()
    )
    lines += ['', f'Certified capacity at ≥ {target:.0%}: {capacities}.']

    for name, block in flavors.items():
        oracle = block.get('length_oracle_2way')
        if oracle:
            flag = ' ⚠ GAMED -- word count alone escapes chance in this pool' if block.get('gamed') else ''
            lines.append(
                f'- `{name}` length-oracle 2-way null: {oracle["accuracy"]:.4f} '
                f'({oracle["ci"][1]:.4f}–{oracle["ci"][2]:.4f}), want ≈ 0.5 or below.{flag}'
            )

    sensitivity: dict[str, Any] = menu.get('sensitivity') or {}
    if sensitivity:
        rows = []
        for name, block in sensitivity.items():
            cell = (block.get('per_k') or {}).get('2') or {}
            if cell.get('accuracy') is not None:
                rows.append(f'`{name}` 2-way {cell["accuracy"]:.4f}')
        if rows:
            lines.append(
                f'- Sensitivity (diagnostic only, no verdict may read these): {", ".join(rows)} -- widened '
                f'tolerances let the true candidate be the best length match, so these overstate content.'
            )

    lines += [
        '',
        f'{menu.get("n_queries", 0)} queries over {menu.get("n_gallery", 0)} sentences; '
        f'{menu.get("dropped_no_prototype", 0)} queries dropped for lacking a training prototype.',
        '',
    ]
    return lines


def _fmt_k(value: Any) -> str:
    """Formats a certified menu size, or 'none' when no size cleared the target."""
    return 'none' if value is None else str(int(value))
