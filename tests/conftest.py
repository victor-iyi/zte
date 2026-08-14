"""Shared pytest fixtures: a tiny synthetic ZuCo tree and a built dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from zte.config import DatasetConfig, MissingConfig
from zte.data.dataset import ZuCoDataset
from zte.data.synthetic import generate_synthetic_zuco


@pytest.fixture(scope='session')
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generates a small synthetic ZuCo tree once per test session.

    Args:
        tmp_path_factory: Pytest temporary-path factory.

    Returns:
        The directory containing the synthetic `.mat` files.
    """
    out = tmp_path_factory.mktemp('zuco')
    generate_synthetic_zuco(out, subjects=('ZAB', 'ZDM'), tasks=('SR', 'NR'), n_sentences=6, show_progress=False)
    return out


@pytest.fixture()
def small_dataset(synthetic_dir: Path, tmp_path: Path) -> ZuCoDataset:
    """Builds a band-power + raw dataset over the synthetic tree.

    Args:
        synthetic_dir: The synthetic `.mat` directory.
        tmp_path: Per-test temporary directory for the cache.

    Returns:
        ZuCoDataset: A built `ZuCoDataset`.
    """
    config = DatasetConfig(
        root=str(synthetic_dir),
        tasks=('SR', 'NR'),
        representation='both',
        raw_window=32,
        missing=MissingConfig(method='mask_only'),
        cache_dir=str(tmp_path / 'cache'),
    )
    return ZuCoDataset(config).build(show_progress=False)
