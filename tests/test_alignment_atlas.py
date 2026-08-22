"""The cross-level atlas, the contrastive geometry report and the cross-level comparison table."""

import json
import math
from typing import Any

import numpy as np
import pytest

from zte.alignment import atlas, compare, contrastive

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def levels() -> list[atlas.LevelPoints]:
    """Three levels in one 6-d space, each offset so a per-level fit would look different from a joint one."""
    rng = np.random.default_rng(0)
    return [
        atlas.LevelPoints(
            level='token',
            vectors=rng.normal(size=(20, 6)),
            labels=[f'tok{i}' for i in range(20)],
            subjects=['ZAB'] * 10 + ['ZDM'] * 10,
            tasks=['NR'] * 20,
        ),
        atlas.LevelPoints(
            level='word',
            vectors=rng.normal(size=(15, 6)) + 5.0,
            labels=[f'word{i}' for i in range(15)],
            subjects=['ZAB'] * 15,
            tasks=['SR'] * 15,
        ),
        atlas.LevelPoints(
            level='sentence',
            vectors=rng.normal(size=(10, 6)) - 5.0,
            labels=[f'sentence {i}' for i in range(10)],
            subjects=['ZDM'] * 10,
            tasks=['SR'] * 10,
        ),
    ]


def _trace_xy(figure: dict[str, Any], axis: str) -> list[float]:
    """Concatenates one axis across every trace, in the order the figure lists them."""
    return [float(value) for trace in figure['data'] for value in trace[axis]]


# --------------------------------------------------------------------------- #
# The atlas: one fitted projection over three levels
# --------------------------------------------------------------------------- #


def test_all_three_levels_share_one_fitted_projection(levels: list[atlas.LevelPoints]) -> None:
    """The drawn coordinates are the joint PCA of the stacked levels, not three separate fits."""
    payload = atlas.build_atlas(levels)

    stacked = np.concatenate([np.asarray(level.vectors, dtype=np.float64) for level in levels], axis=0)
    centred = stacked - stacked.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    expected = u[:, :3] * s[:3]

    assert _trace_xy(payload['figures']['2d'], 'x') == pytest.approx(expected[:, 0].tolist(), abs=1e-8)
    assert _trace_xy(payload['figures']['2d'], 'y') == pytest.approx(expected[:, 1].tolist(), abs=1e-8)
    assert _trace_xy(payload['figures']['3d'], 'z') == pytest.approx(expected[:, 2].tolist(), abs=1e-8)
    assert payload['projection']['fitted_on'] == 'all levels jointly'
    assert payload['projection']['n_fit_rows'] == 45


def test_levels_keep_their_offsets_instead_of_being_recentred(levels: list[atlas.LevelPoints]) -> None:
    """A level that sits far from the others in the embedding sits far from them in the picture."""
    payload = atlas.build_atlas(levels)
    traces = {trace['name']: trace for trace in payload['figures']['2d']['data']}

    centroids = {name: float(np.mean(trace['x'])) for name, trace in traces.items()}

    # Three separate fits would centre every level on the origin and this separation would vanish.
    assert abs(centroids['word'] - centroids['sentence']) > 5.0


def test_atlas_payload_is_strict_json(levels: list[atlas.LevelPoints]) -> None:
    """The whole payload serialises with no numpy scalars and no non-finite floats."""
    payload = atlas.build_atlas(levels)

    assert json.loads(json.dumps(payload, allow_nan=False)) == payload


def test_atlas_reports_the_variance_it_actually_kept() -> None:
    """A space that is genuinely two-dimensional reports its 2D view as keeping all of the variance."""
    rng = np.random.default_rng(1)
    plane = rng.normal(size=(30, 2)) @ np.array([[1.0, 0.0, 2.0, 0.0, -1.0, 0.5], [0.0, 1.0, 0.0, 3.0, 0.0, -0.5]])
    payload = atlas.build_atlas(
        [
            atlas.LevelPoints(level='word', vectors=plane[:20], labels=[f'w{i}' for i in range(20)]),
            atlas.LevelPoints(level='sentence', vectors=plane[20:], labels=[f's{i}' for i in range(10)]),
        ]
    )

    projection = payload['projection']
    assert projection['explained_variance_2d'] == pytest.approx(1.0, abs=1e-9)
    assert projection['explained_variance_3d'] == pytest.approx(1.0, abs=1e-9)
    assert len(projection['explained_variance_ratio']) == 3
    assert projection['views_share_a_basis'] is True
    assert 'PC1 (' in payload['figures']['2d']['layout']['xaxis']['title']['text']


def test_hover_labels_carry_the_text_level_and_subject(levels: list[atlas.LevelPoints]) -> None:
    """Every point names what it is, which rung it sits on and who read it."""
    payload = atlas.build_atlas(levels)
    hover = [text for trace in payload['figures']['3d']['data'] for text in trace['hovertext']]

    assert len(hover) == 45
    assert any(text.startswith('<b>sentence 3</b>') and 'level: sentence' in text for text in hover)
    assert all('subject: ' in text and 'task: ' in text for text in hover)


def test_colouring_by_subject_keeps_the_level_on_the_marker_symbol(levels: list[atlas.LevelPoints]) -> None:
    """Handing the colour channel to the subject leaves the level readable through the symbol."""
    payload = atlas.build_atlas(levels, colour_by='subject')
    traces = payload['figures']['2d']['data']

    assert {trace['legendgroup'] for trace in traces} == {'ZAB', 'ZDM'}
    symbols = {trace['name']: trace['marker']['symbol'] for trace in traces}
    assert symbols['ZAB · token'] == atlas.LEVEL_SYMBOLS['token']
    assert symbols['ZDM · sentence'] == atlas.LEVEL_SYMBOLS['sentence']


def test_umap_degrades_to_pca_and_says_so(levels: list[atlas.LevelPoints], monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable projector never fails silently: the payload names the fallback and the reason."""

    def _missing() -> object:
        raise ImportError('No module named umap')

    monkeypatch.setattr(atlas, '_umap_reducer', _missing)
    payload = atlas.build_atlas(levels, method='umap')

    assert payload['method'] == 'pca'
    assert payload['method_requested'] == 'umap'
    assert payload['degraded'] is True
    assert 'umap' in payload['degraded_reason']


def test_a_neighbourhood_embedding_reports_no_variance_rather_than_inventing_one(
    levels: list[atlas.LevelPoints],
) -> None:
    """t-SNE has no principal axes, so the payload states that instead of printing a variance it did not keep."""
    payload = atlas.build_atlas(levels, method='tsne')
    projection = payload['projection']

    assert payload['method'] == 'tsne'
    assert payload['degraded'] is False
    assert projection['explained_variance_2d'] is None
    assert projection['explained_variance_3d'] is None
    assert projection['views_share_a_basis'] is False
    assert 'not a linear projection' in projection['explained_variance_note']
    assert len(payload['figures']['3d']['data'][0]['z']) == 20


def test_levels_in_different_spaces_are_refused() -> None:
    """A joint projection over levels of different width would be meaningless, so it is refused."""
    with pytest.raises(ValueError, match='one shared space'):
        atlas.build_atlas(
            [
                atlas.LevelPoints(level='word', vectors=np.zeros((4, 6)), labels=list('abcd')),
                atlas.LevelPoints(level='sentence', vectors=np.zeros((4, 8)), labels=list('abcd')),
            ]
        )


def test_a_level_whose_labels_do_not_match_its_vectors_is_refused() -> None:
    """Labels are what makes a point legible, so a ragged level is rejected at construction."""
    with pytest.raises(ValueError, match='4 vectors but 2 labels'):
        atlas.LevelPoints(level='word', vectors=np.zeros((4, 3)), labels=['a', 'b'])


def test_subsampling_is_capped_and_reported() -> None:
    """A level larger than the cap is thinned, and the payload says how many of its points were drawn."""
    payload = atlas.build_atlas(
        [atlas.LevelPoints(level='token', vectors=np.random.default_rng(2).normal(size=(50, 4)), labels=['x'] * 50)],
        max_points_per_level=10,
    )

    block = payload['levels'][0]
    assert (block['n'], block['n_plotted'], block['subsampled']) == (50, 10, True)


# --------------------------------------------------------------------------- #
# Contrastive geometry
# --------------------------------------------------------------------------- #


def test_contrastive_metrics_hold_their_known_values() -> None:
    """Two orthogonal pairs of identical unit vectors give alignment 1, negatives 0 and a rank-1 space."""
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    report = contrastive.contrastive_geometry(
        [contrastive.LevelPairs(level='word', vectors=vectors, positive_ids=np.array([0, 0, 1, 1]))]
    )
    block = report['levels']['word']

    assert block['alignment'] == pytest.approx(1.0, abs=1e-9)
    assert block['alignment_loss'] == pytest.approx(0.0, abs=1e-9)
    assert block['mean_negative_cosine'] == pytest.approx(0.0, abs=1e-9)
    assert block['positive_negative_gap'] == pytest.approx(1.0, abs=1e-9)
    assert block['gap_excludes_zero'] is True

    # log E[exp(-2 d^2)] over the six distinct pairs: two coincident (d^2 = 0) and four orthogonal (d^2 = 2).
    assert block['uniformity'] == pytest.approx(math.log((2.0 + 4.0 * math.exp(-4.0)) / 6.0), abs=1e-5)

    assert block['effective_rank'] == pytest.approx(1.0, abs=1e-6)
    assert block['effective_rank_ratio'] == pytest.approx(0.5, abs=1e-6)
    assert (block['n_anchors'], block['n_positive_pairs'], block['n_groups']) == (4, 4, 2)


def test_a_collapsed_level_is_visible_as_lost_effective_rank() -> None:
    """Perfect alignment bought by collapsing the space shows up in the rank, not in the gap."""
    collapsed = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (8, 1))
    ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    report = contrastive.contrastive_geometry(
        [contrastive.LevelPairs(level='sentence', vectors=collapsed, positive_ids=ids)]
    )
    block = report['levels']['sentence']

    assert block['alignment'] == pytest.approx(1.0, abs=1e-9)
    assert block['positive_negative_gap'] == pytest.approx(0.0, abs=1e-9)
    assert block['effective_rank'] == pytest.approx(0.0, abs=1e-9)


def test_cross_subject_positives_are_the_harder_ones() -> None:
    """Dropping same-subject positives lowers alignment, because the easy pairs were the same brain twice."""
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    ids = np.array([0, 0, 0, 1, 1])
    subjects = np.array(['S1', 'S1', 'S2', 'S1', 'S2'])

    pooled = contrastive.contrastive_geometry(
        [contrastive.LevelPairs(level='word', vectors=vectors, positive_ids=ids, subjects=subjects)]
    )
    honest = contrastive.contrastive_geometry(
        [contrastive.LevelPairs(level='word', vectors=vectors, positive_ids=ids, subjects=subjects)],
        policy='cross_subject',
    )

    assert pooled['levels']['word']['alignment'] > honest['levels']['word']['alignment']
    assert honest['positive_policy'] == 'cross_subject'


def test_cross_subject_policy_without_subjects_is_refused() -> None:
    """The policy cannot be honoured without subject codes, so it is refused rather than quietly pooled."""
    with pytest.raises(ValueError, match='carries no subjects'):
        contrastive.contrastive_geometry(
            [contrastive.LevelPairs(level='word', vectors=np.eye(4), positive_ids=np.array([0, 0, 1, 1]))],
            policy='cross_subject',
        )


def test_contrastive_report_is_strict_json_and_ranks_the_levels() -> None:
    """The payload serialises and names the level whose contrastive gap is widest."""
    rng = np.random.default_rng(3)
    ids = np.repeat(np.arange(6), 4)
    tight = np.repeat(rng.normal(size=(6, 5)), 4, axis=0) + rng.normal(scale=0.01, size=(24, 5))
    loose = rng.normal(size=(24, 5))

    report = contrastive.contrastive_geometry(
        [
            contrastive.LevelPairs(level='token', vectors=loose, positive_ids=ids),
            contrastive.LevelPairs(level='sentence', vectors=tight, positive_ids=ids),
        ]
    )

    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert report['widest_gap'] == 'sentence'
    assert list(report['levels']) == ['token', 'sentence']


def test_the_contrastive_figure_draws_one_interval_per_level() -> None:
    """The gap picture carries a bar and a bootstrap interval for each level it was given."""
    rng = np.random.default_rng(4)
    ids = np.repeat(np.arange(5), 4)
    report = contrastive.contrastive_geometry(
        [
            contrastive.LevelPairs(level='word', vectors=rng.normal(size=(20, 5)), positive_ids=ids),
            contrastive.LevelPairs(level='sentence', vectors=rng.normal(size=(20, 5)), positive_ids=ids),
        ]
    )
    figure = atlas.contrastive_figure(report)

    bar = figure['data'][0]
    assert list(bar['y']) == ['word', 'sentence']
    assert len(bar['error_x']['array']) == 2 and len(bar['error_x']['arrayminus']) == 2
    assert json.loads(json.dumps(figure, allow_nan=False)) == figure


# --------------------------------------------------------------------------- #
# The cross-level comparison table
# --------------------------------------------------------------------------- #


def test_a_token_row_without_its_oracle_floor_is_refused() -> None:
    """A token number sits on a sub-word signature the model got for free, so the floor is not optional."""
    with pytest.raises(ValueError, match='piece-profile oracle floor'):
        compare.LevelRetrieval(level='token', ranks=np.array([1, 2, 3]), gallery_size=10, postprocess_fit='train split')


def test_a_fabricated_oracle_floor_is_refused() -> None:
    """The floor has to be a measured oracle block, not a dict that merely looks like one."""
    with pytest.raises(ValueError, match='no Top-1 to compare against'):
        compare.LevelRetrieval(
            level='token',
            ranks=np.array([1, 2, 3]),
            gallery_size=10,
            postprocess_fit='none',
            oracle_floor={'verdict': 'looks fine'},
        )


def test_cross_level_table_holds_its_known_values() -> None:
    """Hit counts, the exact binomial tail and the rank percentile are all pinned on a known rank vector."""
    table = compare.cross_level_table(
        [
            compare.LevelRetrieval(
                level='sentence',
                ranks=np.array([1, 1, 3, 10]),
                gallery_size=10,
                postprocess_fit='train split',
            )
        ]
    )
    block = table['levels'][0]

    assert (block['hits_top1'], block['hits_top5'], block['hits_top10']) == (2, 3, 4)
    assert block['top1'] == pytest.approx(0.5)
    assert block['chance_top1'] == pytest.approx(0.1)

    # P(X >= 2) for four queries at a chance rate of 1/10.
    assert block['top1_p'] == pytest.approx(1.0 - (0.9**4 + 4 * 0.1 * 0.9**3), abs=1e-9)

    assert block['rank_percentile'] == pytest.approx(np.mean([1.0, 1.0, 1.0 - 2 / 9, 0.0]), abs=1e-9)
    assert block['mean_rank'] == pytest.approx(3.75)
    assert block['mrr'] == pytest.approx((1.0 + 1.0 + 1 / 3 + 0.1) / 4)
    assert block['postprocess_fit'] == 'train split'
    assert block['headline_metric'] == 'rank_percentile'


def test_a_token_row_is_scored_against_its_measured_floor() -> None:
    """A piece profile that resolves every sentence uniquely is a Top-1 of 1.0, which no encoder clears."""
    word_pieces = np.array([[1, 2, 1], [1, 1, 0], [3, 0, 0]])
    floor = compare.token_oracle_floor(word_pieces, observed_top1=0.5)

    assert floor['worst_case_top1'] == pytest.approx(1.0)
    assert floor['beats_oracles'] is False

    table = compare.cross_level_table(
        [
            compare.LevelRetrieval(
                level='token',
                ranks=np.array([1, 1, 5, 20]),
                gallery_size=100,
                postprocess_fit='none',
                oracle_floor=floor,
            )
        ]
    )
    block = table['levels'][0]

    assert block['oracle_floor']['top1'] == pytest.approx(1.0)
    assert block['oracle_floor']['signature'] in ('words', 'total', 'multiset', 'profile')
    assert block['beats_oracle_floor'] is False


def test_a_level_that_clears_its_floor_says_so() -> None:
    """The comparison is a measurement, not a verdict of failure: an encoder above the floor reads as above it."""
    word_pieces = np.array([[1, 1, 0], [1, 1, 0], [2, 0, 0]])
    floor = compare.token_oracle_floor(word_pieces)
    table = compare.cross_level_table(
        [
            compare.LevelRetrieval(
                level='token',
                ranks=np.array([1, 1, 1, 1, 9]),
                gallery_size=10,
                postprocess_fit='none',
                oracle_floor=floor,
            )
        ]
    )

    assert floor['worst_case_top1'] == pytest.approx(2 / 3, abs=1e-9)
    assert table['levels'][0]['beats_oracle_floor'] is True


def test_the_table_orders_levels_fine_to_coarse_and_serialises() -> None:
    """Every level lands in one comparable block, in token-word-sentence order."""
    rows = [
        compare.LevelRetrieval(level='sentence', ranks=np.array([2, 4]), gallery_size=50, postprocess_fit='none'),
        compare.LevelRetrieval(level='word', ranks=np.array([1, 7]), gallery_size=50, postprocess_fit='none'),
        compare.LevelRetrieval(
            level='token',
            ranks=np.array([3, 3]),
            gallery_size=50,
            postprocess_fit='none',
            oracle_floor=compare.token_oracle_floor(np.array([[1, 2], [2, 0]])),
        ),
    ]
    table = compare.cross_level_table(rows)

    assert [block['level'] for block in table['levels']] == ['token', 'word', 'sentence']
    assert json.loads(json.dumps(table, allow_nan=False)) == table


def test_ranks_outside_the_gallery_are_refused() -> None:
    """A rank larger than the gallery cannot have come from that gallery, so the row is rejected."""
    with pytest.raises(ValueError, match='outside a gallery'):
        compare.LevelRetrieval(level='word', ranks=np.array([1, 99]), gallery_size=10, postprocess_fit='none')


def test_the_markdown_table_prints_the_floor_beside_the_number() -> None:
    """The rendered table never shows a token hit count without the floor it has to clear."""
    floor = compare.token_oracle_floor(np.array([[1, 2, 1], [1, 1, 0], [3, 0, 0]]))
    table = compare.cross_level_table(
        [
            compare.LevelRetrieval(
                level='token',
                ranks=np.array([1, 4]),
                gallery_size=20,
                postprocess_fit='transductive',
                oracle_floor=floor,
            )
        ]
    )
    markdown = compare.render_markdown(table)

    assert '| token |' in markdown
    assert 'NO -- below a brain-free floor' in markdown
    assert 'transductive' in markdown


def test_the_package_exports_resolve_lazily() -> None:
    """The public API is reachable from the package without importing every submodule by hand."""
    from zte import alignment

    assert alignment.build_atlas is atlas.build_atlas
    assert alignment.contrastive_geometry is contrastive.contrastive_geometry
    assert alignment.cross_level_table is compare.cross_level_table

    with pytest.raises(AttributeError):
        _ = alignment.no_such_export
