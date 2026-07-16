"""Confound audit -- *how entangled are the factors ZTE tries to separate?*

Before any invariance objective is designed, this module quantifies the thing that governs whether that objective is safe: **how
correlated the nuisance factors (subject, task) are with the content the model is supposed to keep (word length, frequency, meaning),
and with the behaviour it records (fixations, regressions, skipping).**

The motivating diagnosis: an invariance adversary only ever pushes a nuisance *down*; it never pulls
content *up*. So if the nuisance and the content are correlated, deleting the nuisance deletes content with it.  The sharpest case is *task*:
in ZuCo, normal-reading (NR) and sentiment-reading (SR) use **disjoint sentence sets**, so "task" is very nearly an alias for "which stimulus",
and a task/stimulus adversary is partly a content-deletion operator.

Everything here is model-free -- it reads the word-level metadata table (`ZuCoDataset.words`), so it runs anywhere the dataset builds, on
synthetic or real ZuCo alike. Three association measures put every relationship on one comparable `[0, 1]` scale:

- **Cramer's V** (bias-corrected, Bergsma 2013) for categorical x categorical.
- **Correlation ratio eta** for categorical x continuous.
- **|Spearman rho|** for continuous x continuous (monotonic, robust to the non-linear frequency->fixation relationship).

Associations involving eye-tracking durations use pairwise-complete rows, because a skipped word has no fixation and those cells are missing by design, not by error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# The factors the audit reasons about, grouped by their role in the plan.
IDENTITY_FACTORS: tuple[str, ...] = ('subject',)
TASK_FACTORS: tuple[str, ...] = ('task', 'category')
CONTENT_FACTORS: tuple[str, ...] = ('word_len', 'log_freq', 'corpus_log_freq', 'category')
BEHAVIOUR_FACTORS: tuple[str, ...] = (
    'FFD',
    'GD',
    'GPT',
    'TRT',
    'regression_time',
    'n_fixations',
    'is_omitted',
)
LEXICAL_FACTORS: tuple[str, ...] = ('word_len', 'log_freq', 'corpus_log_freq', 'rel_pos')

# Which columns are categorical (everything else numeric is treated as continuous).
_CATEGORICAL: frozenset[str] = frozenset(
    {'subject', 'task', 'category', 'category_scheme', 'length_band', 'is_omitted', 'stimulus_key'}
)


def _codes(values: np.ndarray) -> np.ndarray:
    """Factorises a label vector to contiguous integer codes (NaN/None -> its own code)."""
    import pandas as pd

    codes, _ = pd.factorize(pd.Series(values), use_na_sentinel=False)
    return np.asarray(codes)


def cramers_v(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected Cramer's V between two categorical vectors, in `[0, 1]`.

    Uses the Bergsma (2013) small-sample correction so that two independent columns score near 0 even with many categories and
    modest n -- important here because `subject` and `stimulus_key` have many levels.

    Args:
        a (np.ndarray): First categorical vector `(n,)`.
        b (np.ndarray): Second categorical vector `(n,)`.

    Returns:
        float: Corrected V in `[0, 1]` (0 when either column is constant).

    """
    ca, cb = _codes(a), _codes(b)
    r, k = int(ca.max()) + 1 if ca.size else 0, int(cb.max()) + 1 if cb.size else 0
    if r < 2 or k < 2:
        return 0.0
    table = np.zeros((r, k), dtype=np.float64)
    np.add.at(table, (ca, cb), 1.0)
    n = table.sum()
    row, col = table.sum(axis=1, keepdims=True), table.sum(axis=0, keepdims=True)
    expected = row @ col / n
    chi2 = float((((table - expected) ** 2) / np.clip(expected, 1e-12, None)).sum())
    phi2 = chi2 / n
    # Bergsma bias correction.
    phi2_corr = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def correlation_ratio(categories: np.ndarray, values: np.ndarray) -> float:
    """Correlation ratio eta (categorical -> continuous) in `[0, 1]`.

    eta^2 is the fraction of the continuous variable's variance explained by the group means; eta is its root,
    comparable to a correlation. Rows where `values` is NaN are dropped (pairwise-complete).

    Args:
        categories (np.ndarray): Categorical grouping `(n,)`.
        values (np.ndarray): Continuous vector `(n,)`.

    Returns:
        float: eta in `[0, 1]` (0 when fewer than two populated groups).

    """
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() < 3:
        return 0.0
    codes = _codes(np.asarray(categories)[finite])
    values = values[finite]
    k = int(codes.max()) + 1 if codes.size else 0
    if k < 2:
        return 0.0
    grand = values.mean()
    ss_between = 0.0
    for g in range(k):
        vg = values[codes == g]
        if vg.size:
            ss_between += vg.size * (vg.mean() - grand) ** 2
    ss_total = ((values - grand) ** 2).sum()
    if ss_total <= 1e-12:
        return 0.0
    return float(np.sqrt(np.clip(ss_between / ss_total, 0.0, 1.0)))


def abs_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """`|Spearman rho|` between two continuous vectors over pairwise-complete rows."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return 0.0
    xr = _rankdata(x[finite])
    yr = _rankdata(y[finite])
    xc, yc = xr - xr.mean(), yr - yr.mean()
    den = np.sqrt((xc**2).sum() * (yc**2).sum())
    if den <= 1e-12:
        return 0.0
    return float(abs((xc @ yc) / den))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-rank transform (ties share the mean rank), matching Spearman's definition."""
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # Average tied ranks.
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def _is_categorical(df: 'pd.DataFrame', col: str) -> bool:
    """Whether a column should be treated as categorical for association purposes."""
    if col in _CATEGORICAL:
        return True
    import pandas as pd

    return not pd.api.types.is_numeric_dtype(df[col])


def association(df: 'pd.DataFrame', a: str, b: str) -> tuple[float, str]:
    """Association between two columns in `[0, 1]`, dispatched by dtype.

    Returns the value and the measure name used (`cramers_v`, `eta`, or `spearman`), so the caller can
    report which statistic backs each cell.

    Args:
        df (pd.DataFrame): The word-level metadata table.
        a (str): First column name.
        b (str): Second column name.

    Returns:
        tuple[float, str]: `(association, measure_name)`.

    """
    cat_a, cat_b = _is_categorical(df, a), _is_categorical(df, b)
    va, vb = df[a].to_numpy(), df[b].to_numpy()
    if cat_a and cat_b:
        return cramers_v(va, vb), 'cramers_v'
    if cat_a and not cat_b:
        return correlation_ratio(va, vb), 'eta'
    if cat_b and not cat_a:
        return correlation_ratio(vb, va), 'eta'
    return abs_spearman(va, vb), 'spearman'


def _with_derived(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Adds audit-only derived columns (e.g. regression time) without touching the source."""
    out = df.copy()
    if {'GPT', 'GD'}.issubset(out.columns):
        # Go-past minus gaze duration ~ time spent in regressions from this word.
        out['regression_time'] = out['GPT'].astype('float64') - out['GD'].astype('float64')
    if 'is_omitted' in out.columns:
        out['is_omitted'] = out['is_omitted'].astype('float64')
    return out


def task_stimulus_overlap(df: 'pd.DataFrame') -> dict:
    """The decisive query: do NR and SR share any stimuli, or is task an alias for content?

    Args:
        df (pd.DataFrame): Word table with `task` and `stimulus_key`.

    Returns:
        dict: Counts, shared-stimulus fraction, `cramers_v(task, stimulus_key)`, and a plain-English verdict.

    """
    if 'stimulus_key' not in df.columns or 'task' not in df.columns:
        return {'available': False}
    import pandas as pd  # noqa: F401

    tasks_per_stim = df.groupby('stimulus_key')['task'].nunique()
    n_stim = int(tasks_per_stim.size)
    n_shared = int((tasks_per_stim > 1).sum())
    v = cramers_v(df['task'].to_numpy(), df['stimulus_key'].to_numpy())
    confounded = n_shared == 0
    verdict = (
        'FULLY CONFOUNDED: task and stimulus are disjoint -- no sentence appears under '
        'both tasks, so any task/stimulus-invariance loss also deletes content. '
        'De-confound (match negatives within task) rather than turning the adversary up.'
        if confounded
        else f'PARTIAL: {n_shared}/{n_stim} stimuli appear under >1 task; task-invariance '
        'is partly separable from content.'
    )
    return {
        'available': True,
        'n_stimuli': n_stim,
        'n_shared_across_tasks': n_shared,
        'shared_fraction': n_shared / max(n_stim, 1),
        'cramers_v_task_stimulus': v,
        'fully_confounded': confounded,
        'verdict': verdict,
    }


def association_matrix(df: 'pd.DataFrame', factors: 'list[str]') -> dict:
    """Full symmetric association matrix over `factors` (present columns only)."""
    present = [f for f in factors if f in df.columns]
    n = len(present)
    mat = np.zeros((n, n))
    measures: dict[str, str] = {}
    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            val, name = association(df, present[i], present[j])
            mat[i, j] = mat[j, i] = val
            measures[f'{present[i]}|{present[j]}'] = name
    return {'factors': present, 'matrix': mat.tolist(), 'measures': measures}


def confound_report(df: 'pd.DataFrame') -> dict:
    """Runs the whole audit and returns a JSON-able dict.

    Args:
        df (pd.DataFrame): `ZuCoDataset.words` (or any word-level metadata table).

    Returns:
        dict: task/stimulus overlap, the association matrix, and the two targeted cross-tabs (nuisance->content, behaviour->lexical)
            that drive the plan's design decisions.

    """
    d = _with_derived(df)
    all_factors = list(
        dict.fromkeys(
            [
                *IDENTITY_FACTORS,
                *TASK_FACTORS,
                *CONTENT_FACTORS,
                *LEXICAL_FACTORS,
                *BEHAVIOUR_FACTORS,
            ]
        )
    )

    def cross(rows: 'list[str]', cols: 'list[str]') -> dict:
        rows = [r for r in rows if r in d.columns]
        cols = [c for c in cols if c in d.columns]
        cells = {r: {c: association(d, r, c)[0] for c in cols if c != r} for r in rows}
        return {'rows': rows, 'cols': cols, 'values': cells}

    return {
        'n_words': int(len(d)),
        'task_stimulus': task_stimulus_overlap(d),
        'nuisance_vs_content': cross(
            [*IDENTITY_FACTORS, 'task'], ['word_len', 'log_freq', 'corpus_log_freq', 'category']
        ),
        'behaviour_vs_lexical': cross(list(BEHAVIOUR_FACTORS), list(LEXICAL_FACTORS)),
        'association_matrix': association_matrix(d, all_factors),
    }


def _fmt(v: float) -> str:
    """Formats an association as a fixed-width cell, flagging strong ones."""
    flag = ' ⚠' if v >= 0.5 else ''
    return f'{v:.2f}{flag}'


def render_markdown(report: dict, title: str = 'ZTE Confound Audit') -> str:
    """Renders a `confound_report` dict to a self-contained Markdown document."""
    lines: list[str] = [f'# {title}', '']
    lines.append(
        f'Model-free audit over **{report["n_words"]:,} words**. Associations are on a '
        "common `[0, 1]` scale: Cramér's V (cat×cat), correlation ratio η (cat×cont), "
        '|Spearman ρ| (cont×cont). Cells ≥ 0.50 are flagged ⚠.'
    )
    lines.append('')

    ts = report['task_stimulus']
    lines += ['## 1. The decisive query — is task an alias for the stimulus?', '']
    if ts.get('available'):
        lines += [
            f'- Distinct stimuli: **{ts["n_stimuli"]}**',
            f'- Stimuli appearing under more than one task: **{ts["n_shared_across_tasks"]}** '
            f'({ts["shared_fraction"]:.0%})',
            f"- Cramér's V(task, stimulus_key): **{ts['cramers_v_task_stimulus']:.3f}**",
            '',
            f'> **{ts["verdict"]}**',
            '',
        ]
    else:
        lines += ['_stimulus_key/task columns unavailable._', '']

    def table(cross: dict, heading: str, note: str) -> None:
        lines.append(f'## {heading}')
        lines.append('')
        lines.append(note)
        lines.append('')
        cols = cross['cols']
        lines.append('| | ' + ' | '.join(f'`{c}`' for c in cols) + ' |')
        lines.append('|' + '---|' * (len(cols) + 1))
        for r in cross['rows']:
            row = cross['values'].get(r, {})
            cells = [_fmt(row[c]) if c in row else '—' for c in cols]
            lines.append(f'| `{r}` | ' + ' | '.join(cells) + ' |')
        lines.append('')

    table(
        report['nuisance_vs_content'],
        '2. Nuisance → content bleed',
        'How much an invariance loss on each **row** would drag on each content **column**. '
        'High cells are where deleting the nuisance deletes meaning.',
    )
    table(
        report['behaviour_vs_lexical'],
        '3. Behaviour ↔ lexical signal',
        'Reading behaviour as a proxy for lexical processing (justifies eye-tracking as '
        'privileged supervision). High cells mean the eyes already track the word.',
    )

    am = report['association_matrix']
    fac = am['factors']
    mat = am['matrix']
    lines += ['## 4. Full association matrix', '']
    lines.append('| | ' + ' | '.join(f'`{f}`' for f in fac) + ' |')
    lines.append('|' + '---|' * (len(fac) + 1))
    for i, f in enumerate(fac):
        cells = [_fmt(mat[i][j]) if j != i else '·' for j in range(len(fac))]
        lines.append(f'| `{f}` | ' + ' | '.join(cells) + ' |')
    lines.append('')
    return '\n'.join(lines)
