"""One parallax transfer cell: a task-X encoder scored on task-Y readings from a never-seen subject."""

import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Final

import numpy as np

from zte.cli.support.io import write_json
from zte.evaluation.audit.menu import menu_report
from zte.evaluation.audit.rebaseline import fit_postprocess, stratified_retrieval
from zte.logging_utils import get_logger

_LOG = get_logger('parallax.transfer')

# Part of the transfer.json schema: it states exactly whose rows fitted the whitening, so a reader can
# tell the post-processing is reproducible by a deployed decoder -- no holdout rows, no train-task rows.
POSTPROCESS_FIT: Final[str] = 'non-holdout subjects, eval task'
"""How every transfer cell's post-processing is fitted."""


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two representations of the same readings, computed in float64.

    `cka = ||x_c^T y_c||_F^2 / (||x_c^T x_c||_F * ||y_c^T y_c||_F)` with column-centred inputs, so the
    value is invariant to orthogonal transforms and isotropic scaling of either side and equals 1 only
    when the two spaces carry the same similarity structure.

    Args:
        x (np.ndarray): Embeddings `(n, d1)`.
        y (np.ndarray): Embeddings `(n, d2)` of the same `n` readings, in the same row order.

    Returns:
        float: CKA in `[0, 1]`; `nan` when either side is constant.

    Raises:
        ValueError: If the two sides disagree on the number of readings.
    """
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f'CKA needs the same readings on both sides: {a.shape[0]} vs {b.shape[0]} rows.')

    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    cross = float(np.linalg.norm(a.T @ b) ** 2)
    denom = float(np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b))
    if denom == 0.0:
        return float('nan')

    return cross / denom


def _canonical(text: str) -> str:
    """One conservative form for the disjointness verdict: NFKC, casefolded, whitespace collapsed."""
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def stimulus_novelty(train_task: str, eval_task: str, overlap: int) -> bool:
    """Whether a cell's eval stimuli are genuinely never-seen by the training task.

    A same-task cell is never novel. A cross-task cell with any overlap is not novel either, and the
    overlap is logged loudly -- the never-seen-stimuli claim is the whole point of the off-diagonal,
    so it must fail visibly rather than pass silently.

    Args:
        train_task (str): The scored model's training task.
        eval_task (str): The task whose readings are being scored.
        overlap (int): How many eval stimulus texts also appear in the training task.

    Returns:
        bool: True only for a cross-task cell with zero stimulus overlap.
    """
    if train_task == eval_task:
        return False
    if overlap > 0:
        _LOG.warning(
            'Cross-task cell %s->%s: %d eval stimuli also appear in the training task. novel_stimuli is '
            'False; this cell is NOT never-seen-stimuli evidence.',
            train_task,
            eval_task,
            overlap,
        )
        return False

    return True


def transfer_report(
    sent_emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    n_words: np.ndarray,
    texts: np.ndarray,
    *,
    train_task: str,
    eval_task: str,
    holdout: str,
    train_stimulus_texts: Iterable[str],
    seed: int = 0,
    length_tol: int = 1,
    ks: tuple[int, ...] = (1, 5, 10),
    n_boot: int = 2000,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scores one transfer cell on eval-task readings and returns the `transfer.json` dict.

    The statistics are the audited ones, never reimplemented: `stratified_retrieval` for the full and
    length-matched galleries and `menu_report` for closed-set capacity. Post-processing is fitted on
    the non-holdout rows of the eval task only, so the holdout subject's readings are queries and
    nothing else -- they enter neither the whitening statistics nor the menu prototypes.

    Args:
        sent_emb (np.ndarray): Sentence embeddings `(n, d)` of the eval-task readings, raw encoder output.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.
        subjects (np.ndarray): Subject code per reading `(n,)`.
        n_words (np.ndarray): Word count per reading `(n,)`.
        texts (np.ndarray): Stimulus text per reading `(n,)`, matched against the training task's stimuli.
        train_task (str): The scored model's training task.
        eval_task (str): The task whose readings are scored.
        holdout (str): The held-out subject code, whose readings are the queries.
        train_stimulus_texts (Iterable[str]): Every stimulus text of the model's training task.
        seed (int, optional): Bootstrap/permutation seed. Defaults to 0.
        length_tol (int, optional): Word-count tolerance of the length-stratified gallery. Defaults to 1.
        ks (tuple[int, ...], optional): Retrieval Top-K cut-offs. Defaults to (1, 5, 10).
        n_boot (int, optional): Bootstrap resamples behind every CI. Defaults to 2000.
        provenance (dict[str, Any] | None, optional): Checkpoint/run provenance carried into the cell.
            Defaults to None.

    Returns:
        dict[str, Any]: `{'train_task', 'eval_task', 'seed', 'holdout', 'novel_stimuli',
            'stimulus_overlap', 'stimulus_overlap_normalized', 'n_queries', 'held_out',
            'held_out_length_stratified', 'menu',
            'postprocess_fit', 'provenance'}`.
    """
    subject_arr = np.asarray(subjects).astype(str)
    id_arr = np.asarray(content_ids)
    lengths = np.asarray(n_words, dtype=np.float64).ravel()
    emb = np.asarray(sent_emb, dtype=np.float32)

    # Train-fitted only: the holdout's rows never touch the whitening statistics.
    mask = subject_arr != holdout
    processed = fit_postprocess(emb[mask])(emb)

    held_out = stratified_retrieval(processed, id_arr, subject_arr, holdout, None, ks=ks, n_boot=n_boot, seed=seed)
    held_out_length = stratified_retrieval(
        processed, id_arr, subject_arr, holdout, lengths, length_tol=length_tol, ks=ks, n_boot=n_boot, seed=seed
    )
    # Within one task the task dimension is constant, so the menu's length-matched flavor is the headline.
    menu = menu_report(emb, id_arr, subject_arr, holdout, lengths, train_mask=mask, n_boot=n_boot, seed=seed)

    eval_stimuli = {str(t) for t in np.unique(np.asarray(texts).astype(str))}
    train_texts = {str(t) for t in train_stimulus_texts}
    overlap = len(eval_stimuli & train_texts)
    # The verdict runs on a canonical form: a duplicate hiding behind case, whitespace or an encoding
    # artifact must still fail the never-seen claim -- normalisation can only tighten it, never loosen it.
    overlap_normalized = len({_canonical(t) for t in eval_stimuli} & {_canonical(t) for t in train_texts})
    novel = stimulus_novelty(train_task, eval_task, max(overlap, overlap_normalized))

    n_queries = int(held_out['n_queries']) if held_out else int(np.sum(subject_arr == holdout))
    return {
        'train_task': str(train_task),
        'eval_task': str(eval_task),
        'seed': int(seed),
        'holdout': str(holdout),
        'novel_stimuli': bool(novel),
        'stimulus_overlap': int(overlap),
        'stimulus_overlap_normalized': int(overlap_normalized),
        'n_queries': n_queries,
        'held_out': held_out,
        'held_out_length_stratified': held_out_length,
        'menu': menu,
        'postprocess_fit': POSTPROCESS_FIT,
        'provenance': provenance or {},
    }


def write_cell(
    cell_dir: str | Path,
    report: dict[str, Any],
    *,
    sent_emb: np.ndarray,
    content_ids: np.ndarray,
    subjects: np.ndarray,
    n_words: np.ndarray,
    texts: np.ndarray,
) -> Path:
    """Writes one transfer cell to disk: `transfer.json` plus the embeddings the report was scored from.

    The embeddings are the raw encoder outputs, not the post-processed rows, so the report stage can
    fit its own train-only transforms and compute CKA on what the models actually produced.

    Args:
        cell_dir (str | Path): The cell directory, `<train>_to_<eval>_s<seed>`.
        report (dict[str, Any]): The dict from `transfer_report`.
        sent_emb (np.ndarray): Sentence embeddings `(n, d)`, stored as float32.
        content_ids (np.ndarray): Stimulus id per reading `(n,)`.
        subjects (np.ndarray): Subject code per reading `(n,)`.
        n_words (np.ndarray): Word count per reading `(n,)`.
        texts (np.ndarray): Stimulus text per reading `(n,)`, for the chamber's point labels.

    Returns:
        Path: The cell directory.
    """
    out = Path(cell_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'transfer.json', report, default=str)

    np.savez_compressed(
        out / 'embeddings.npz',
        sent_emb=np.asarray(sent_emb, dtype=np.float32),
        content_ids=np.asarray(content_ids),
        subjects=np.asarray(subjects).astype(str),
        n_words=np.asarray(n_words, dtype=np.float64),
        texts=np.asarray(texts).astype(str),
    )

    return out
