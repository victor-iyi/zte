"""Tests for the menu-capacity audit: closed-form K-way accuracy, the length/task-shortcut guards, certification."""

import numpy as np
import pytest

import zte.evaluation.audit.menu as menu_mod
from zte.evaluation.audit.menu import (
    DEFAULT_MENU_KS,
    ENROLLED_SCORING,
    HEADLINE_TOL,
    PROTOTYPE_SCORING,
    menu_markdown_lines,
    menu_report,
)
from zte.evaluation.audit.rebaseline import rebaseline_report, render_markdown

_SUBJECTS: tuple[str, ...] = ('ZAB', 'ZDM', 'ZKB')


def _cohort(
    n_stimuli: int = 30,
    dim: int = 16,
    noise: float = 0.0,
    seed: int = 0,
    codes: tuple[str, ...] = _SUBJECTS,
    length_values: tuple[int, ...] = (5, 10),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A cohort of subjects reading the same stimuli, each stimulus a fixed direction plus per-subject noise."""
    rng = np.random.default_rng(seed)
    directions = rng.standard_normal((n_stimuli, dim)).astype(np.float32)
    lengths = np.asarray(length_values, dtype=np.float64)[rng.integers(0, len(length_values), size=n_stimuli)]

    emb, content, subjects, words = [], [], [], []
    for code in codes:
        emb.append(directions + noise * rng.standard_normal(directions.shape).astype(np.float32))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(lengths)
    return (
        np.concatenate(emb),
        np.concatenate(content),
        np.array(subjects),
        np.concatenate(words),
    )


# --------------------------------------------------------------------------- #
# the closed form
# --------------------------------------------------------------------------- #
def test_menu_accuracy_reproduces_its_closed_form() -> None:
    """Each accuracy is the exact hypergeometric expectation over distractor draws, not a simulation.

    Three sentences of equal length, one training subject, queries built to beat 2 / 1 / 0 distractors: the K=2
    accuracy must be (1 + 1/2 + 0) / 3 and the K=3 accuracy (1 + 0 + 0) / 3, with no seed anywhere.
    """
    protos = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.1], [1.0, 0.5], [1.0, -0.2]], dtype=np.float32)
    emb = np.concatenate([queries, protos])
    content = np.array([0, 1, 2, 0, 1, 2])
    subjects = np.array(['ZAB'] * 3 + ['ZDM'] * 3)
    words = np.full(6, 10.0)

    out = menu_report(emb, content, subjects, 'ZAB', words, ks=(2, 3), postprocess=False, n_boot=100, n_perm=100)
    assert out is not None
    assert out['headline_flavor'] == 'length_matched'
    for flavor in ('length_matched', 'open'):  # every sentence has the same length, so both pools are identical here
        per_k = out['flavors'][flavor]['per_k']
        assert per_k['2']['accuracy'] == pytest.approx(0.5)
        assert per_k['3']['accuracy'] == pytest.approx(1 / 3)
        assert per_k['2']['chance'] == pytest.approx(0.5)
        assert per_k['3']['chance'] == pytest.approx(1 / 3)
        assert per_k['2']['n_queries'] == 3
        assert per_k['2']['perm_p'] is not None


def test_a_perfect_embedding_is_certified_at_every_servable_menu_size() -> None:
    """A noiseless cohort wins every menu, and capacity is the largest K the exact-length pools can serve."""
    emb, content, subjects, words = _cohort(noise=0.0)
    out = menu_report(emb, content, subjects, 'ZAB', words, ks=(2, 4, 64), n_boot=200, n_perm=200)
    assert out is not None
    for flavor in ('length_matched', 'open'):
        per_k = out['flavors'][flavor]['per_k']
        for k in ('2', '4'):
            assert per_k[k]['accuracy'] == pytest.approx(1.0)
            assert per_k[k]['perm_p'] is not None and per_k[k]['perm_p'] < 0.05
        # 30 stimuli cannot serve a 64-way menu: no pool holds 63 distractors, so the size reports empty.
        assert per_k['64']['accuracy'] is None and per_k['64']['n_queries'] == 0
        assert out['flavors'][flavor]['capacity_point'] == 4

    # The exact-length pool is length-clean by construction, so a perfect embedding certifies there.
    # The open pool carries length information, so its certification is at the oracle's mercy: gamed vetoes it.
    matched = out['flavors']['length_matched']
    assert matched['gamed'] is False and matched['capacity'] == 4
    open_block = out['flavors']['open']
    assert open_block['capacity'] == (None if open_block['gamed'] else 4)


def test_a_random_embedding_sits_at_chance_and_certifies_nothing() -> None:
    """A no-signal cohort must bracket 1/K -- and this is also the leak canary.

    Every reading is a unique random vector. If a held-out reading could enrol its own sentence's prototype, the
    query would meet its own vector and the accuracy would inflate toward 1.0 instead of sitting at chance.
    """
    rng = np.random.default_rng(0)
    _, content, subjects, words = _cohort(n_stimuli=60)
    emb = rng.standard_normal((len(subjects), 16)).astype(np.float32)

    out = menu_report(emb, content, subjects, 'ZAB', words, ks=(2, 4), n_boot=500, n_perm=300)
    assert out is not None
    for flavor in ('length_matched', 'open'):
        block = out['flavors'][flavor]
        _, lo, hi = block['per_k']['2']['ci']
        assert lo < 0.5 < hi, (flavor, lo, hi)
        assert block['per_k']['2']['accuracy'] < 0.7
        assert block['capacity'] is None
        assert block['capacity_point'] is None


# --------------------------------------------------------------------------- #
# the shortcut guards
# --------------------------------------------------------------------------- #
def test_a_fine_grained_length_code_escapes_a_widened_tolerance_but_not_the_headline() -> None:
    """The reason the headline is exact-match: at tol 1 the truth is the unique best length match in its stratum.

    An embedding that encodes the exact word count and nothing else must sit at chance in the certified
    exact-length flavor, while the widened tol-1 sensitivity row shows it escaping -- which is exactly why no
    verdict may read a widened row.
    """
    rng = np.random.default_rng(2)
    n_stimuli, length_values = 60, (10, 11, 12, 13, 14)
    lengths = np.asarray(length_values, dtype=np.float64)[rng.integers(0, len(length_values), size=n_stimuli)]
    directions = np.eye(len(length_values), dtype=np.float32)[np.searchsorted(length_values, lengths)]

    emb, content, subjects, words = [], [], [], []
    for code in _SUBJECTS:
        emb.append(directions + 0.05 * rng.standard_normal(directions.shape).astype(np.float32))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(lengths)

    out = menu_report(
        np.concatenate(emb),
        np.concatenate(content),
        np.array(subjects),
        'ZAB',
        np.concatenate(words),
        ks=(2,),
        postprocess=False,
        n_boot=500,
        n_perm=200,
    )
    assert out is not None
    headline = out['flavors']['length_matched']['per_k']['2']
    _, lo, hi = headline['ci']
    assert lo < 0.5 < hi, (lo, hi)
    assert out['flavors']['length_matched']['gamed'] is False

    widened = out['sensitivity']['length_matched_tol1']['per_k']['2']
    assert widened['accuracy'] > 0.65


def test_a_task_style_code_cannot_win_the_task_matched_menu() -> None:
    """The task confound guard: pure task register wins mixed-task menus and sits at chance in same-task ones."""
    rng = np.random.default_rng(3)
    n_stimuli = 60
    tasks_per_stim = np.array(['SR'] * 30 + ['NR'] * 30)
    directions = np.where(tasks_per_stim[:, None] == 'SR', 1.0, -1.0).astype(np.float32) * np.ones(
        (n_stimuli, 4), dtype=np.float32
    )

    emb, content, subjects, words, tasks = [], [], [], [], []
    for code in _SUBJECTS:
        emb.append(directions + 0.05 * rng.standard_normal(directions.shape).astype(np.float32))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(np.full(n_stimuli, 10.0))
        tasks.append(tasks_per_stim)

    out = menu_report(
        np.concatenate(emb),
        np.concatenate(content),
        np.array(subjects),
        'ZAB',
        np.concatenate(words),
        tasks=np.concatenate(tasks),
        ks=(2,),
        postprocess=False,
        n_boot=500,
        n_perm=200,
    )
    assert out is not None
    assert out['headline_flavor'] == 'length_task_matched'

    same_task = out['flavors']['length_task_matched']['per_k']['2']
    _, lo, hi = same_task['ci']
    assert lo < 0.5 < hi, (lo, hi)

    mixed = out['flavors']['length_matched']['per_k']['2']
    assert mixed['accuracy'] > 0.6


# --------------------------------------------------------------------------- #
# refusals and integration
# --------------------------------------------------------------------------- #
def test_menu_declines_a_cohort_it_cannot_score() -> None:
    """No training subjects or no held-out queries means no menu, reported as `None` rather than invented."""
    emb, content, _, words = _cohort(n_stimuli=8)
    lone = np.array(['ZAB'] * len(emb))
    assert menu_report(emb, content, lone, 'ZAB', words) is None
    emb, content, subjects, words = _cohort(n_stimuli=8)
    assert menu_report(emb, content, subjects, 'ZZZ', words) is None


def test_menu_flows_into_the_rebaseline_report_and_its_markdown() -> None:
    """`zte-rebaseline` ships the menu section beside the grid, so the capacity is readable without the JSON."""
    emb, content, subjects, words = _cohort(n_stimuli=40, noise=0.4)
    tasks = np.array(['SR' if cid < 20 else 'NR' for cid in content])
    report = rebaseline_report(emb, content, subjects, 'ZAB', words, tasks=tasks, menu_ks=(2, 4), n_boot=200)

    menu = report['menu']
    assert menu is not None
    assert menu['postprocess_fit'] == 'train split'
    assert menu['headline_flavor'] == 'length_task_matched'
    assert sorted(menu['flavors']['open']['per_k']) == ['2', '4']
    assert 'length_task_matched_tol1' in menu['sensitivity']

    text = render_markdown(report)
    assert '## Menu capacity -- K-way closed-set accuracy' in text
    assert 'Certified capacity at ≥ 80%' in text
    assert 'length-oracle 2-way null' in text


def test_the_default_menu_sweep_and_headline_tolerance_are_pinned() -> None:
    """The sweep and the exact-match rule are quoted in docs; a silent default change cannot move a headline."""
    assert DEFAULT_MENU_KS == (2, 4, 8, 16, 32, 64)
    assert HEADLINE_TOL == 0


# --------------------------------------------------------------------------- #
# the enrolled flavors
# --------------------------------------------------------------------------- #
def test_tight_readings_make_enrolled_and_prototype_agree() -> None:
    """When every reading of a sentence clusters tightly, the best reading and the centroid tell one story."""
    emb, content, subjects, words = _cohort(noise=0.05)
    out = menu_report(emb, content, subjects, 'ZAB', words, ks=(2,), postprocess=False, n_boot=200, n_perm=200)
    assert out is not None

    proto = out['flavors']['length_matched']['per_k']['2']['accuracy']
    enrolled = out['flavors']['length_matched_enrolled']['per_k']['2']['accuracy']
    assert proto > 0.9 and enrolled > 0.9
    assert enrolled == pytest.approx(proto, abs=0.05)


def test_split_reading_styles_are_reachable_by_enrollment_but_not_by_the_centroid() -> None:
    """The signal lives in individual readings: antipodal reading styles beat the centroid, not the enrollment.

    Each sentence's cross-subject readings form two distant sub-clusters and the held-out reading sits in one
    of them. The best enrolled reading is essentially the query's own style and wins the 2-way menu; the
    centroid averages the styles away and sits at chance. Both directions are pinned, because this is the
    measured phenomenon -- retrieval percentile high, prototype menu at chance -- reproduced synthetically.
    """
    rng = np.random.default_rng(7)
    n_stimuli, dim = 40, 32
    directions = rng.standard_normal((n_stimuli, dim)).astype(np.float32)

    emb_parts, content, subjects, words = [], [], [], []
    for code, sign in (('ZAB', 1.0), ('S1', 1.0), ('S2', 1.0), ('S3', -1.0), ('S4', -1.0)):
        noise = 0.05 * rng.standard_normal(directions.shape).astype(np.float32)
        emb_parts.append(sign * directions + noise)
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(np.full(n_stimuli, 10.0))

    out = menu_report(
        np.concatenate(emb_parts),
        np.concatenate(content),
        np.array(subjects),
        'ZAB',
        np.concatenate(words),
        ks=(2,),
        postprocess=False,
        n_boot=500,
        n_perm=300,
    )
    assert out is not None

    proto = out['flavors']['length_matched']
    _, proto_lo, proto_hi = proto['per_k']['2']['ci']
    assert proto_lo < 0.5 < proto_hi, (proto_lo, proto_hi)
    assert proto['per_k']['2']['accuracy'] < 0.65
    assert proto['capacity'] is None

    enrolled = out['flavors']['length_matched_enrolled']
    cell = enrolled['per_k']['2']
    assert cell['accuracy'] > 0.9
    assert cell['ci'][1] > 0.8 and cell['perm_p'] is not None and cell['perm_p'] < 0.05
    assert enrolled['capacity'] == 2
    assert enrolled['gamed'] is False, 'the length oracle applies to enrolled pools and must sit at chance here'


def test_mutation_readmitting_holdout_readings_to_enrollment_blows_the_canary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MUTATION: enrolled references drawn from every row let each query meet its own reading -- red, loudly.

    The honest random cohort sits at chance. With the reference mask widened to all rows, the query's own
    reading is enrolled, its cosine hits 1.0 and the 2-way accuracy saturates -- exactly the leak the
    train-mask restriction exists to prevent, so the at-chance canary must go red under this mutation.
    """
    rng = np.random.default_rng(0)
    _, content, subjects, words = _cohort(n_stimuli=60)
    emb = rng.standard_normal((len(subjects), 16)).astype(np.float32)

    honest = menu_report(emb, content, subjects, 'ZAB', words, ks=(2,), n_boot=300, n_perm=100)
    assert honest is not None
    _, lo, hi = honest['flavors']['length_matched_enrolled']['per_k']['2']['ci']
    assert lo < 0.5 < hi, (lo, hi)

    real = menu_mod._enrolled_scores

    def leaky(
        emb: np.ndarray,
        q_idx: np.ndarray,
        content_ids: np.ndarray,
        proto_id_arr: np.ndarray,
        reference_mask: np.ndarray,
    ) -> np.ndarray:
        return real(emb, q_idx, content_ids, proto_id_arr, np.ones_like(reference_mask))

    monkeypatch.setattr(menu_mod, '_enrolled_scores', leaky)
    mutated = menu_report(emb, content, subjects, 'ZAB', words, ks=(2,), n_boot=300, n_perm=100)
    assert mutated is not None

    leaked = mutated['flavors']['length_matched_enrolled']['per_k']['2']
    assert leaked['accuracy'] > 0.95, 'the leak the mask exists to prevent must saturate the menu'
    assert leaked['ci'][1] > 0.5, 'the at-chance canary above must fail under this mutation'


def test_a_train_mask_admitting_holdout_readings_is_refused() -> None:
    """The enrolled references' precondition is enforced, not assumed: a leaky train_mask is an error."""
    emb, content, subjects, words = _cohort(n_stimuli=8)
    leaky_mask = np.ones(len(subjects), dtype=bool)

    with pytest.raises(ValueError, match='held-out'):
        menu_report(emb, content, subjects, 'ZAB', words, train_mask=leaky_mask)


def test_every_flavor_block_names_its_scoring_rule() -> None:
    """Each flavor says how a candidate was scored, so a reader can never mistake enrolled for prototype."""
    emb, content, subjects, words = _cohort(n_stimuli=20, noise=0.3)
    tasks = np.array(['SR' if cid < 10 else 'NR' for cid in content])
    out = menu_report(emb, content, subjects, 'ZAB', words, tasks=tasks, ks=(2,), n_boot=100, n_perm=50)
    assert out is not None
    assert out['headline_flavor'] == 'length_task_matched', 'the headline stays the strictest prototype claim'

    expected = {
        'length_task_matched': PROTOTYPE_SCORING,
        'length_matched': PROTOTYPE_SCORING,
        'open': PROTOTYPE_SCORING,
        'length_task_matched_enrolled': ENROLLED_SCORING,
        'length_matched_enrolled': ENROLLED_SCORING,
    }
    assert {name: block['scoring'] for name, block in out['flavors'].items()} == expected
    for block in out['sensitivity'].values():
        assert block['scoring'] == PROTOTYPE_SCORING


def test_a_random_cohort_gives_enrollment_no_free_lunch() -> None:
    """Taking the best of several references must not manufacture signal: enrolled brackets chance too."""
    rng = np.random.default_rng(1)
    _, content, subjects, words = _cohort(n_stimuli=60)
    emb = rng.standard_normal((len(subjects), 16)).astype(np.float32)

    out = menu_report(emb, content, subjects, 'ZAB', words, ks=(2, 4), n_boot=500, n_perm=300)
    assert out is not None

    block = out['flavors']['length_matched_enrolled']
    _, lo, hi = block['per_k']['2']['ci']
    assert lo < 0.5 < hi, (lo, hi)
    assert block['capacity'] is None and block['capacity_point'] is None


def test_markdown_lines_survive_an_unservable_size() -> None:
    """A menu size no query could serve renders as a dash row, never a crash or a fabricated number."""
    emb, content, subjects, words = _cohort(n_stimuli=6, noise=0.2)
    out = menu_report(emb, content, subjects, 'ZAB', words, ks=(2, 64), n_boot=100, n_perm=50)
    assert out is not None
    text = '\n'.join(menu_markdown_lines(out))
    assert '| 64 |' in text and '| — | — | 0 |' in text


def test_a_gamed_pool_can_never_certify_a_capacity() -> None:
    """Certification ANDs the oracle verdict: a pool where word count alone wins may not certify any K."""
    rng = np.random.default_rng(7)
    n_per_band, bands = 15, (5.0, 10.0, 15.0, 20.0)
    n_stimuli = n_per_band * len(bands)
    lengths = np.repeat(bands, n_per_band)
    # The embedding knows the word count and nothing else, so the open pool is winnable -- and so is its
    # oracle, which must flag the pool gamed and veto the certification the accuracy alone would earn.
    directions = np.eye(len(bands), dtype=np.float32)[np.repeat(np.arange(len(bands)), n_per_band)]

    emb, content, subjects, words = [], [], [], []
    for code in ('ZAB', 'ZDM', 'ZKB'):
        emb.append(directions + 0.05 * rng.standard_normal((n_stimuli, len(bands))).astype(np.float32))
        content.append(np.arange(n_stimuli))
        subjects += [code] * n_stimuli
        words.append(lengths)

    out = menu_report(
        np.concatenate(emb),
        np.concatenate(content),
        np.array(subjects),
        'ZAB',
        np.concatenate(words),
        ks=(2,),
        postprocess=False,
        n_boot=300,
        n_perm=200,
    )
    assert out is not None
    open_block = out['flavors']['open']

    assert open_block['gamed'] is True
    cell = open_block['per_k']['2']
    assert cell['accuracy'] > 0.8 and cell['perm_p'] < 0.05, 'the pool would certify but for the oracle'
    assert open_block['capacity'] is None and open_block['capacity_point'] is not None


def test_enrolled_blocks_carry_their_reading_counts() -> None:
    """Max-over-readings grows with enrollment size, so every enrolled block records the counts it drew from."""
    emb, content, subjects, words = _cohort(noise=0.1)
    out = menu_report(emb, content, subjects, 'ZAB', words, ks=(2,), n_boot=100, n_perm=50)
    assert out is not None

    enrolled = out['flavors']['length_matched_enrolled']
    counts = enrolled['enrolled_reading_counts']
    assert counts == {'mean': 2.0, 'min': 2, 'max': 2}
    assert 'enrolled_reading_counts' not in out['flavors']['length_matched']
