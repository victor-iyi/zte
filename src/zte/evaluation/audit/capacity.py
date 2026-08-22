"""Decoder menu capacity: the largest K-way menu the decoder is certified to serve, and what it is worth in bits."""

from math import comb, log2
from typing import Any, Final

import numpy as np

from zte.evaluation.audit.menu import (
    DEFAULT_MENU_KS,
    HEADLINE_TOL,
    MenuPool,
    beaten_in_pool,
    menu_pools,
    win_prob,
)
from zte.evaluation.audit.scoreboard import _bootstrap_ci

type ScoreFamilies = dict[str, dict[str, np.ndarray]]
"""Score family name -- `pmi` or `raw` -- to arm name to a `(n_query, n_gallery)` score matrix."""

# The embedding-side menu audit sweeps this ladder, so a decoder capacity reads against it on identical sizes.
CAPACITY_KS: Final[tuple[int, ...]] = DEFAULT_MENU_KS
"""Menu sizes the capacity certification sweeps."""

# PMI subtracts the query-independent null-prefix score, so a candidate cannot win on the LM's own priors.
HEADLINE_SCORE: Final[str] = 'pmi'
"""Score family the headline capacity is read from."""

# Distractors share the query's task and its exact stimulus word count, so a hit is neither a length
# shortcut nor a task-register shortcut.
HEADLINE_FLAVOR: Final[str] = 'length_task_matched'
"""Candidate-pool rule the headline capacity is read from."""

# Each arm substitutes the conditioning and nothing else, so beating all three leaves the EEG prefix
# as the only remaining explanation of the win.
CERTIFYING_ARMS: Final[tuple[str, ...]] = ('length_only', 'shuffled_eeg', 'mismatch')
"""Control arms the model must beat, paired, at every certified menu size."""

DEFAULT_N_PERM: Final[int] = 2000
"""Label permutations behind each per-K p-value; the attainable floor is `1 / (n_perm + 1)`."""

ENTROPY_IDENTITY: Final[float] = 9.4512
"""Bits of stimulus identity in the 700-stimulus SR+NR gallery."""

# Word count is free -- ZuCo segments words by eye tracking, so the pad mask is the word count. Only what
# is left after conditioning on it can be credited to the brain.
ENTROPY_IDENTITY_GIVEN_LENGTH: Final[float] = 4.3090
"""Bits of stimulus identity left once word count is known -- the only honest denominator."""

# The split that holds out both the subject and the stimulus; anything else lets a query meet itself.
HONEST_SPLIT: Final[tuple[str, str]] = ('by_subject_and_stimulus', 'test')
"""The `(split_strategy, split_cell)` pair a certifiable report must have been evaluated on."""

CLAUSE_NAMES: Final[tuple[str, ...]] = (
    'honest_split',
    'flavor_certifiable',
    'above_chance',
    'beats_length_only_paired',
    'beats_shuffled_paired',
    'beats_mismatch_paired',
    'permutation_significant',
)
"""The seven certification clauses, in the order a cell reports its failures."""

CERTIFICATION_RULE: Final[str] = (
    'A menu size K is certified when, at K and at every smaller size swept, the split is '
    'by_subject_and_stimulus/test, the pool is length-matched (never open), the bootstrap CI lower bound of '
    'accuracy exceeds 1/K, the model beats length_only, shuffled_eeg and mismatch on both a paired bootstrap '
    'CI lower bound above zero and an exact sign test below alpha, and the permutation p falls below alpha.'
)
"""One-sentence statement of what a certified menu size means."""

MI_ASSUMPTION: Final[str] = 'uniform prior over K alternatives, errors spread symmetrically over the K-1 wrong ones'
"""What the confusion-channel bits estimate assumes, and the headline estimate deliberately does not."""


def capacity_report(
    arms: dict[str, Any],
    content_ids: np.ndarray,
    subjects: np.ndarray,
    holdout: str,
    n_words: np.ndarray,
    *,
    tasks: np.ndarray | None = None,
    train_mask: np.ndarray | None = None,
    ks: tuple[int, ...] = CAPACITY_KS,
    alpha: float = 0.05,
    n_perm: int = DEFAULT_N_PERM,
    n_boot: int = 2000,
    seed: int = 0,
    honest_split: bool,
    split_strategy: str,
    split_cell: str,
    evidence_content: bool = True,
) -> dict[str, Any] | None:
    """Certifies the largest K-way menu a decoder serves, against its own conditioning controls.

    Every arm is a score matrix over the same gallery, produced by the identical bridge, LM and length
    normalisation -- only the conditioning differs -- so a paired difference isolates the EEG prefix. Each
    accuracy is the exact expectation over uniformly drawn distractors, so chance is exactly `1/K` and there
    is no distractor sampling; ties count as losses, which is why a constant score matrix scores zero.

    Certification is contiguous: a size counts only if it and every smaller size swept pass all seven clauses,
    and only if the same holds on the common subset of queries scoreable at every size, because the pools
    shrink with K and a rising tail can be a surviving subpopulation rather than a capacity.

    Args:
        arms (dict[str, Any]): Arm name to a `(n_query, n_gallery)` score matrix, or score family name to
            such a mapping. Every family needs a `model` arm. Query rows are the holdout readings in array
            order; gallery columns follow `np.unique(content_ids)` restricted to ids with a training reading.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.
        subjects (np.ndarray): Subject code per reading `(n,)`.
        holdout (str): The held-out subject code, whose readings are the queries.
        n_words (np.ndarray): Word count per reading `(n,)`.
        tasks (np.ndarray | None, optional): Task label per reading `(n,)`; enables the task-matched
            headline flavor. Defaults to None.
        train_mask (np.ndarray | None, optional): Boolean `(n,)` marking the reference readings that define
            the gallery and its stimulus-level word counts. Defaults to every non-holdout row.
        ks (tuple[int, ...], optional): Menu sizes to sweep. Defaults to (2, 4, 8, 16, 32, 64).
        alpha (float, optional): Significance level of every CI and p-value. Defaults to 0.05.
        n_perm (int, optional): Label permutations behind each per-K p-value. Defaults to 2000.
        n_boot (int, optional): Bootstrap resamples behind each CI. Defaults to 2000.
        seed (int, optional): Bootstrap and permutation seed. Defaults to 0.
        honest_split (bool): Whether the caller evaluated on a genuinely held-out cell.
        split_strategy (str): The split strategy the scores came from.
        split_cell (str): The split cell the scores came from.
        evidence_content (bool, optional): Whether the arms carried a word-level EEG evidence path.
            Defaults to True.

    Returns:
        dict[str, Any] | None: The capacity report -- `readout`, `certified_k`, per-family/per-flavor
            `scores`, the `bits` ledger, the `verdict` and its `provenance`; `None` when there is nothing
            to score.

    Raises:
        ValueError: If a score family has no `model` arm, or an arm's shape does not match the gallery.
    """
    subjects_arr = np.asarray(subjects)
    content_arr = np.asarray(content_ids)
    lengths = np.asarray(n_words, dtype=np.float64).ravel()
    task_arr = None if tasks is None else np.asarray(tasks)
    mask = (subjects_arr != holdout) if train_mask is None else np.asarray(train_mask, dtype=bool)
    q_mask = subjects_arr == holdout
    if not bool(q_mask.any()) or int(mask.sum()) < 1:
        return None

    families = _families(arms)
    for family, block in families.items():
        if 'model' not in block:
            raise ValueError(f'score family {family!r} has no `model` arm; the certification has nothing to test.')

    # The gallery is the stimuli a reference reading exists for, with a stimulus-level word count taken as the
    # median over those readings so no single reading's omissions move a candidate between length strata.
    proto_ids: list[Any] = []
    proto_lengths: list[float] = []
    proto_tasks: list[Any] = []
    for cid in np.unique(content_arr):
        rows = np.where((content_arr == cid) & mask)[0]
        if rows.size == 0:
            continue
        proto_ids.append(cid)
        proto_lengths.append(float(np.median(lengths[rows])))
        proto_tasks.append(task_arr[rows[0]] if task_arr is not None else None)

    if len(proto_ids) < 2:
        return None

    proto_id_arr = np.asarray(proto_ids)
    proto_len = np.asarray(proto_lengths, dtype=np.float64)
    proto_task = np.asarray(proto_tasks) if task_arr is not None else None

    q_all = np.where(q_mask)[0]
    keep = np.isin(content_arr[q_all], proto_id_arr)
    q_idx = q_all[keep]
    if q_idx.size == 0:
        return None

    n_gallery = int(proto_id_arr.size)
    for family, block in families.items():
        for arm, matrix in block.items():
            if matrix.shape != (int(q_all.size), n_gallery):
                raise ValueError(
                    f'arm {family}/{arm} has shape {matrix.shape}, expected {(int(q_all.size), n_gallery)}.'
                )
        families[family] = {arm: matrix[keep] for arm, matrix in block.items()}

    ks_sorted = tuple(sorted({int(k) for k in ks}))
    honest = bool(honest_split) and (str(split_strategy), str(split_cell)) == HONEST_SPLIT

    flavor_specs: list[tuple[str, int | None, bool]] = []
    if proto_task is not None:
        flavor_specs.append((HEADLINE_FLAVOR, HEADLINE_TOL, True))
    flavor_specs += [('length_matched', HEADLINE_TOL, False), ('open', None, False)]

    rng = np.random.default_rng(int(seed) + 1)
    scores: dict[str, Any] = {}
    for family, block in families.items():
        flavors: dict[str, Any] = {}
        for name, tol, task_matched in flavor_specs:
            pools = menu_pools(
                q_idx, content_arr, lengths, proto_id_arr, proto_len, proto_task, tol=tol, task_matched=task_matched
            )
            flavors[name] = _score_flavor(
                block,
                pools,
                certifiable=honest and name != 'open',
                honest=honest,
                flavor_ok=name != 'open',
                ks=ks_sorted,
                alpha=alpha,
                n_boot=n_boot,
                n_perm=n_perm,
                rng=rng,
            )
        scores[family] = flavors

    head_score = HEADLINE_SCORE if HEADLINE_SCORE in scores else next(iter(scores))
    head_flavor = HEADLINE_FLAVOR if HEADLINE_FLAVOR in scores[head_score] else 'length_matched'
    head_block = scores[head_score][head_flavor]
    certified_k = head_block['certified_k']

    return {
        'readout': 'menu selection',
        'holdout': str(holdout),
        'n_queries': int(q_idx.size),
        'n_gallery': n_gallery,
        'tie_policy': 'ties lose',
        'honest_split': bool(honest_split),
        'split_strategy': str(split_strategy),
        'split_cell': str(split_cell),
        'headline': {'score': head_score, 'flavor': head_flavor, 'alpha': float(alpha)},
        'certified_k': certified_k,
        'scores': scores,
        'bits': _bits_ledger(head_block, certified_k),
        'verdict': _verdict(head_block, certified_k, head_score, head_flavor, ks_sorted),
        'provenance': {
            'n_perm': int(n_perm),
            'n_boot': int(n_boot),
            'alpha': float(alpha),
            'seed': int(seed),
            'ks': list(ks_sorted),
            'arms_present': sorted({arm for block in families.values() for arm in block}),
            'evidence_content': bool(evidence_content),
        },
    }


def capacity_markdown_lines(capacity: dict[str, Any]) -> list[str]:
    """The Markdown block for the decoder capacity certification.

    Args:
        capacity (dict[str, Any]): The dict from `capacity_report`.

    Returns:
        list[str]: Markdown lines, ending with a blank line.
    """
    headline = capacity.get('headline') or {}
    score, flavor = headline.get('score', HEADLINE_SCORE), headline.get('flavor', HEADLINE_FLAVOR)
    block = ((capacity.get('scores') or {}).get(score) or {}).get(flavor) or {}
    bits = capacity.get('bits') or {}
    verdict = capacity.get('verdict') or {}

    lines = [
        '## Decoder menu capacity -- certified K-way selection',
        '',
        f'The readout is {capacity.get("readout")}, never generation: given the held-out reading and K '
        f'candidate sentences, does the decoder score the one actually read above every distractor? '
        f'Accuracy is the exact expectation over uniformly drawn distractors, so chance is exactly 1/K and '
        f'{capacity.get("tie_policy")}. Headline family `{score}`, pool `{flavor}`, holdout '
        f'`{capacity.get("holdout")}`, {capacity.get("n_queries")} queries over a '
        f'{capacity.get("n_gallery")}-sentence gallery. {CERTIFICATION_RULE}',
        '',
        '| K | chance | accuracy (95% CI) | perm p | n | certified | failed clauses |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    lines += _cell_rows(block.get('per_k') or {})
    lines += [
        '',
        'On the common subset -- only the queries scoreable at every K, so a shrinking pool cannot pass '
        'for a rising capacity:',
        '',
        '| K | chance | accuracy (95% CI) | perm p | n | certified | failed clauses |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    lines += _cell_rows(block.get('common_subset') or {})

    paired_rows = _paired_rows(block.get('per_k') or {})
    if paired_rows:
        lines += [
            '',
            '| K | control | model - control (95% CI) | sign-test p | model wins | control wins | ties |',
            '| --- | --- | --- | --- | --- | --- | --- |',
        ]
        lines += paired_rows

    certified = bits.get('bits_certified')
    lines += [
        '',
        f'- Certified menu size: **K = {_fmt(capacity.get("certified_k"))}** '
        f'({"certifiable" if block.get("certifiable") else "not certifiable"} pool, '
        f'split `{capacity.get("split_strategy")}`/`{capacity.get("split_cell")}`).',
        f'- Bits, estimator `{bits.get("estimator")}`: '
        f'{"—" if certified is None else format(certified, ".4f")} of the '
        f'{_fmt_float(bits.get("entropy_identity_given_length"))} bits of stimulus identity that survive knowing '
        f'word count (word count itself carries {_fmt_float(bits.get("bits_from_length"))} of the '
        f'{_fmt_float(bits.get("entropy_identity"))} total).',
        f'- Fraction of that residual recovered: {_fmt_float(bits.get("fraction_of_residual"))}; '
        f'unrecovered {_fmt_float(bits.get("bits_unrecovered"))} bits.',
        f'- Confusion-channel cross-check ({bits.get("bits_mi_assumption")}): '
        f'{_fmt_float(bits.get("bits_mi_confusion"))} bits.',
        f'- Verdict: {verdict.get("reason")}',
        '',
    ]

    return lines


def pooled_capacity(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """The capacity a set of runs jointly supports -- the smallest certified size, and none if any run fails.

    Args:
        reports (list[dict[str, Any]]): Reports from `capacity_report`, one per seed or holdout.

    Returns:
        dict[str, Any]: `{'n_reports', 'certified_k', 'bits_certified', 'capacity_certified', 'per_report',
            'reason'}`.
    """
    per_report = [
        {
            'holdout': report.get('holdout'),
            'seed': (report.get('provenance') or {}).get('seed'),
            'certified_k': report.get('certified_k'),
        }
        for report in reports
    ]
    sizes = [row['certified_k'] for row in per_report]
    # A pooled capacity is a promise every run keeps, so one run that certifies nothing sinks the pool.
    certified_k = int(min(k for k in sizes if k is not None)) if sizes and all(k is not None for k in sizes) else None

    if not reports:
        reason = 'No capacity reports to pool.'
    elif certified_k is None:
        failed = [str(row['holdout']) for row in per_report if row['certified_k'] is None]
        reason = (
            f'{len(failed)} of {len(reports)} runs certified nothing ({", ".join(failed)}); pooled capacity is none.'
        )
    else:
        reason = f'Every one of {len(reports)} runs certified; the pooled capacity is the smallest, K = {certified_k}.'

    return {
        'n_reports': len(reports),
        'certified_k': certified_k,
        'bits_certified': None if certified_k is None else float(log2(certified_k)),
        'capacity_certified': certified_k is not None,
        'per_report': per_report,
        'reason': reason,
    }


def _families(arms: dict[str, Any]) -> ScoreFamilies:
    """Accepts either one flat family of arms or a family-keyed mapping, and returns the family-keyed form."""
    if arms and all(isinstance(value, dict) for value in arms.values()):
        return {
            str(family): {str(arm): np.asarray(matrix, dtype=np.float64) for arm, matrix in block.items()}
            for family, block in arms.items()
        }

    flat = {str(arm): np.asarray(matrix, dtype=np.float64) for arm, matrix in arms.items()}

    # `null_prefix` is identically zero under PMI, so a family carrying it can only be the raw one.
    return {'raw' if 'null_prefix' in flat else HEADLINE_SCORE: flat}


def _score_flavor(
    family_arms: dict[str, np.ndarray],
    pools: list[MenuPool],
    *,
    certifiable: bool,
    honest: bool,
    flavor_ok: bool,
    ks: tuple[int, ...],
    alpha: float,
    n_boot: int,
    n_perm: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Sweeps one candidate-pool rule over every menu size, on the full queries and on the common subset."""
    structural = tuple(name for name, ok in (('honest_split', honest), ('flavor_certifiable', flavor_ok)) if not ok)

    def sweep(subset: list[MenuPool]) -> dict[str, Any]:
        return {
            str(k): _cell(
                family_arms, subset, k, alpha=alpha, n_boot=n_boot, n_perm=n_perm, rng=rng, structural=structural
            )
            for k in ks
        }

    per_k = sweep(pools)

    # A menu size no pool can fill is unreachable, not failed. Exact word-count pools hold ~8 candidates on a
    # 300-sentence gallery and ~18 on a 700-sentence one, so bounding the subset by the nominal largest K
    # would empty it and report `certified_k: None` for a decoder of any quality.
    feasible = tuple(k for k in ks if any(pool.m >= k - 1 for pool in pools))
    reach = max(feasible) if feasible else 0

    # Pools shrink as K grows, so a capacity read off a K-specific subpopulation is not a capacity at all.
    common = sweep([pool for pool in pools if pool.m >= reach - 1])

    full_k, common_k = _contiguous_k(per_k, ks), _contiguous_k(common, ks)
    certified_k = None if full_k is None or common_k is None else min(full_k, common_k)

    # Identically zero on a tol-0 pool -- every candidate shares the true word count, so the distance oracle
    # ties everywhere and ties lose. Diagnostic only; the real length control is the `length_only` arm.
    oracle_rows = [pool for pool in pools if pool.m >= 1]
    oracle = (
        float(np.mean([win_prob(int(pool.oracle_beaten[pool.true_slot]), pool.m, 2) for pool in oracle_rows]))
        if oracle_rows
        else float('nan')
    )

    return {
        'certifiable': bool(certifiable),
        'per_k': per_k,
        'common_subset': common,
        'ks_feasible': list(feasible),
        'ks_unreachable': [k for k in ks if k not in feasible],
        'certified_k': certified_k,
        'gamed': bool(oracle > 0.5),
        'length_oracle_2way_distance': oracle,
    }


def _cell(
    family_arms: dict[str, np.ndarray],
    pools: list[MenuPool],
    k: int,
    *,
    alpha: float,
    n_boot: int,
    n_perm: int,
    rng: np.random.Generator,
    structural: tuple[str, ...],
) -> dict[str, Any]:
    """Scores one menu size: every arm's accuracy, the paired control comparisons, and the clause verdict."""
    rows = [pool for pool in pools if pool.m >= k - 1]
    floor = 1.0 / (n_perm + 1)
    chance = 1.0 / k
    if not rows:
        return {
            'accuracy': None,
            'ci': None,
            'chance': chance,
            'n_queries': 0,
            'perm_p': None,
            'perm_p_floor': floor,
            'arms': {},
            'paired': {},
            'certified': False,
            'failed_clauses': list(structural) + list(CLAUSE_NAMES[2:]),
        }

    probs = {
        arm: np.array([win_prob(beaten_in_pool(matrix[pool.row], pool), pool.m, k) for pool in rows], dtype=np.float64)
        for arm, matrix in family_arms.items()
    }
    arms_out = {
        arm: {
            'accuracy': float(value.mean()),
            'ci': list(_bootstrap_ci(value, n_boot, alpha, int(rng.integers(2**31)))),
        }
        for arm, value in probs.items()
    }
    accuracy, lo, hi = _bootstrap_ci(probs['model'], n_boot, alpha, int(rng.integers(2**31)))

    # Paired on identical query indices: the difference is taken per query, so nothing rests on the two arms
    # having faced the same pools only on average.
    paired: dict[str, Any] = {}
    for arm in CERTIFYING_ARMS:
        if arm not in probs:
            continue
        delta = probs['model'] - probs[arm]
        d_mean, d_lo, d_hi = _bootstrap_ci(delta, n_boot, alpha, int(rng.integers(2**31)))
        model_wins, control_wins = int((delta > 0).sum()), int((delta < 0).sum())
        paired[arm] = {
            'delta': d_mean,
            'ci': [d_mean, d_lo, d_hi],
            'sign_test_p': _sign_test_p(model_wins, model_wins + control_wins),
            'model_wins': model_wins,
            'control_wins': control_wins,
            'ties': int(delta.size) - model_wins - control_wins,
            'n_pairs': int(delta.size),
        }

    perm_p = _permutation_p(family_arms['model'], rows, k, accuracy, n_perm=n_perm, rng=rng)

    failed = list(structural)
    if not lo > chance:
        failed.append('above_chance')
    for arm, clause in zip(CERTIFYING_ARMS, CLAUSE_NAMES[3:6], strict=True):
        cell = paired.get(arm)
        if cell is None or not (cell['ci'][1] > 0.0 and cell['sign_test_p'] < alpha):
            failed.append(clause)
    if perm_p is None or not perm_p < alpha:
        failed.append('permutation_significant')

    return {
        'accuracy': accuracy,
        'ci': [accuracy, lo, hi],
        'chance': chance,
        'n_queries': len(rows),
        'perm_p': perm_p,
        'perm_p_floor': floor,
        'arms': arms_out,
        'paired': paired,
        'certified': not failed,
        'failed_clauses': failed,
    }


def _permutation_p(
    model: np.ndarray, rows: list[MenuPool], k: int, observed: float, *, n_perm: int, rng: np.random.Generator
) -> float | None:
    """Exact relabelling null: the pseudo-true is drawn uniformly from each query's own candidate set."""
    if n_perm <= 0:
        return None

    # Re-using each candidate's own win probability rather than re-scoring keeps every artefact of the data
    # inside the null, so only the identity of the true sentence is destroyed.
    pools = [np.array([win_prob(b, pool.m, k) for b in _strict_wins(model[pool.row][pool.cand_idx])]) for pool in rows]
    flat = np.concatenate(pools)
    sizes = np.array([pool.size for pool in pools])
    offsets = np.concatenate(([0], np.cumsum(sizes[:-1])))
    picks = rng.integers(0, sizes, size=(n_perm, len(pools)))
    draws = flat[offsets + picks].mean(axis=1)

    return float((np.sum(draws >= observed) + 1) / (n_perm + 1))


def _strict_wins(values: np.ndarray) -> np.ndarray:
    """How many other entries each entry strictly beats; ties count as losses."""
    return np.searchsorted(np.sort(values), values, side='left')


def _contiguous_k(per_k: dict[str, Any], ks: tuple[int, ...]) -> int | None:
    """The largest menu size certified with every smaller swept size also certified."""
    largest: int | None = None
    for k in ks:
        if not per_k[str(k)]['certified']:
            break
        largest = k

    return largest


def _sign_test_p(successes: int, n: int) -> float:
    """Two-sided exact sign test over `n` discordant pairs, `successes` of them won by the model."""
    if n <= 0:
        return 1.0

    extreme = max(int(successes), n - int(successes))

    return min(1.0, 2.0 * sum(comb(n, i) for i in range(extreme, n + 1)) / 2**n)


def _mi_bits(accuracy: float, k: int) -> float:
    """Mutual information of a symmetric K-ary channel at the given accuracy, in bits."""
    if k < 2:
        return 0.0

    a = min(max(float(accuracy), 0.0), 1.0)
    err = 1.0 - a
    total = log2(k)
    if a > 0.0:
        total += a * log2(a)
    if err > 0.0:
        total += err * log2(err / (k - 1))

    # At exactly 1/K the three terms cancel analytically; the clamp only absorbs the floating residue.
    return float(max(total, 0.0))


def _bits_ledger(block: dict[str, Any], certified_k: int | None) -> dict[str, Any]:
    """The bits ledger: what the certified menu size is worth against the identity left after word count."""
    bits_certified = None if certified_k is None else float(log2(certified_k))
    cell = (block.get('per_k') or {}).get(str(certified_k)) if certified_k is not None else None
    accuracy = None if cell is None else cell.get('accuracy')

    return {
        'estimator': 'log2(certified K)',
        'bits_certified': bits_certified,
        'bits_mi_confusion': None if accuracy is None or certified_k is None else _mi_bits(accuracy, certified_k),
        'bits_mi_assumption': MI_ASSUMPTION,
        'entropy_identity': ENTROPY_IDENTITY,
        'entropy_identity_given_length': ENTROPY_IDENTITY_GIVEN_LENGTH,
        'bits_from_length': round(ENTROPY_IDENTITY - ENTROPY_IDENTITY_GIVEN_LENGTH, 4),
        'bits_unrecovered': None if bits_certified is None else ENTROPY_IDENTITY_GIVEN_LENGTH - bits_certified,
        'fraction_of_residual': None if bits_certified is None else bits_certified / ENTROPY_IDENTITY_GIVEN_LENGTH,
    }


def _verdict(
    block: dict[str, Any], certified_k: int | None, score: str, flavor: str, ks: tuple[int, ...]
) -> dict[str, Any]:
    """The capacity verdict, with the clause outcomes read where the sweep stopped."""
    # When nothing certified, the smallest swept size is where the failure is legible; when something did,
    # the clauses that matter are the ones at the size being claimed.
    key = str(certified_k if certified_k is not None else ks[0])
    cell = (block.get('per_k') or {}).get(key) or {}
    # An absent cell is not a clean sweep -- only an explicitly empty failure list means every clause held.
    failed = set(cell['failed_clauses']) if 'failed_clauses' in cell else set(CLAUSE_NAMES)
    clauses = {name: name not in failed for name in CLAUSE_NAMES}

    if certified_k is None:
        reason = (
            f'No menu size certified on `{score}`/`{flavor}`; at K = {ks[0]} the failing clauses are '
            f'{", ".join(sorted(failed)) or "none"}.'
        )
    else:
        reason = (
            f'Certified K = {certified_k} on `{score}`/`{flavor}`, contiguous from K = {ks[0]} and holding on '
            f'the common subset -- {log2(certified_k):.4f} bits of menu selection, not generation.'
        )

    return {
        'capacity_certified': certified_k is not None,
        'capacity_k': certified_k,
        'capacity_bits': None if certified_k is None else float(log2(certified_k)),
        'capacity_clauses': clauses,
        'reason': reason,
    }


def _cell_rows(per_k: dict[str, Any]) -> list[str]:
    """One Markdown table row per menu size."""
    rows = []
    for key, cell in per_k.items():
        failed = ', '.join(cell.get('failed_clauses') or []) or '—'
        if cell.get('accuracy') is None:
            rows.append(f'| {key} | {cell.get("chance", 0.0):.4f} | — | — | 0 | no | {failed} |')
            continue
        ci = cell['ci']
        rows.append(
            f'| {key} | {cell["chance"]:.4f} | {cell["accuracy"]:.4f} ({ci[1]:.4f}–{ci[2]:.4f}) '
            f'| {_fmt_p(cell.get("perm_p"), cell.get("perm_p_floor", 0.0))} | {cell.get("n_queries", 0)} '
            f'| {"yes" if cell.get("certified") else "no"} | {failed} |'
        )

    return rows


def _paired_rows(per_k: dict[str, Any]) -> list[str]:
    """One Markdown table row per (menu size, control) paired comparison."""
    rows = []
    for key, cell in per_k.items():
        for arm, paired in (cell.get('paired') or {}).items():
            ci = paired['ci']
            rows.append(
                f'| {key} | {arm} | {paired["delta"]:+.4f} ({ci[1]:+.4f}–{ci[2]:+.4f}) '
                f'| {paired["sign_test_p"]:.2e} | {paired["model_wins"]} | {paired["control_wins"]} '
                f'| {paired["ties"]} |'
            )

    return rows


def _fmt_p(p: float | None, floor: float) -> str:
    """A p-value, rendered as its attainable floor when the permutation null never reached the observed value."""
    if p is None:
        return '—'

    return f'<{floor:.2e}' if p <= floor else f'{p:.4f}'


def _fmt(value: int | None) -> str:
    """An optional menu size."""
    return '—' if value is None else str(value)


def _fmt_float(value: float | None) -> str:
    """An optional scalar."""
    return '—' if value is None else f'{value:.4f}'
