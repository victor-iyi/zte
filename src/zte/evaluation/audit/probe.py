"""Detectability calibration: the R2 a probe recovers from a signal planted in the real feature matrix."""

from typing import Any, Final

import numpy as np

from zte.training.metrics import linear_probe

# The ladder brackets the region where a linear probe stops separating a planted signal from its own null;
# 0.0 is the null rung every other rung is judged against, so it is always included.
_DEFAULT_SNRS: Final[tuple[float, ...]] = (0.0, 0.005, 0.01, 0.02, 0.05)
"""Fractions of target variance planted along a linear direction of the features."""

# Recovery depends on which direction carries the signal, not only on how much of it there is.
_DEFAULT_REPEATS: Final[int] = 2
"""Planted directions drawn per rung."""

# Two rungs closer than this are the same number twice; a floor claimed there would be a rounding artefact.
_MIN_SEPARATION: Final[float] = 1e-4
"""Recovered-R2 margin a rung must clear above the null rung to count as detected."""


def plant_linear_target(features: np.ndarray, snr: float, *, seed: int = 0) -> np.ndarray:
    """Draws a target carrying exactly `snr` of its variance along a random linear direction of `features`.

    Args:
        features (np.ndarray): The real feature matrix `(n, p)`, so dimensionality and collinearity stay the run's own.
        snr (float): Share of target variance an oracle linear predictor explains, in `[0, 1]`.
        seed (int, optional): RNG seed. Defaults to 0.

    Returns:
        np.ndarray: The planted target `(n,)`, zero mean and unit variance.
    """
    x = np.asarray(features, dtype=np.float64)
    rng = np.random.default_rng(seed)

    centred = x - x.mean(axis=0, keepdims=True)
    spread = centred.std(axis=0, keepdims=True)
    standardised = centred / np.where(spread > 0.0, spread, 1.0)

    signal = _standardise(standardised @ rng.standard_normal(standardised.shape[1]))
    # Orthogonalising the noise makes the realised share exactly `snr` instead of approximately it.
    drawn = _standardise(rng.standard_normal(len(x)))
    noise = _standardise(drawn - signal * float(drawn @ signal) / max(float(signal @ signal), 1e-12))

    share = float(np.clip(snr, 0.0, 1.0))
    return np.sqrt(share) * signal + np.sqrt(1.0 - share) * noise


def detectability_curve(
    features: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    snrs: tuple[float, ...] = _DEFAULT_SNRS,
    n_repeats: int = _DEFAULT_REPEATS,
    n_splits: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    """Measures what R2 the probe recovers from a known signal planted in this exact feature matrix.

    Note:
        A probe reading 0.0036 has measured one of two things: no signal, or a signal below what this matrix, this
        sample size and this estimator can resolve. Planting a known one separates them. The floor is the lowest
        recovered R2 that still clears the null rung, so an observation under it is `below detectability` rather
        than `absent` -- and an observation over it is a measurement.

    Args:
        features (np.ndarray): The feature matrix the observed score was measured on `(n, p)`.
        groups (np.ndarray | None, optional): Group label per row; folds are grouped exactly as the observed
            score's folds were, or the floor would be calibrated against an easier question.
        snrs (tuple[float, ...], optional): Planted variance shares. `0.0` is added if absent.
        n_repeats (int, optional): Planted directions per rung. Defaults to 2.
        n_splits (int, optional): Cross-validation folds. Defaults to 3.
        seed (int, optional): RNG seed for the planted directions and the fold assignment. Defaults to 0.

    Returns:
        dict[str, Any]: `rungs` (per-SNR recovered R2), `null_r2`, `floor_r2`, `floor_snr`, `established`,
            `n`, `n_features`, `n_repeats`, `grouped` and `n_groups`.
    """
    x = np.asarray(features, dtype=np.float64)
    ladder = tuple(sorted({0.0, *(float(s) for s in snrs)}))
    repeats = max(1, int(n_repeats))

    rungs: list[dict[str, Any]] = []
    for rung_index, snr in enumerate(ladder):
        scores = [
            _score(
                linear_probe(
                    x,
                    plant_linear_target(x, snr, seed=seed + 1_000 * rung_index + draw),
                    task='regression',
                    n_splits=n_splits,
                    seed=seed,
                    groups=groups,
                )
            )
            for draw in range(repeats)
        ]
        rungs.append(
            {
                'snr': snr,
                'recovered_r2': [round(s, 4) for s in scores],
                'mean': round(float(np.mean(scores)), 4),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
            }
        )

    # The floor is the first rung whose worst draw beats the null rung's best draw: below that the two are one number.
    null = rungs[0]
    floor_r2: float | None = None
    floor_snr: float | None = None
    for rung in rungs[1:]:
        if rung['min'] > null['max'] + _MIN_SEPARATION:
            floor_r2, floor_snr = float(rung['mean']), float(rung['snr'])
            break

    return {
        'rungs': [{k: v for k, v in rung.items() if k not in {'min', 'max'}} for rung in rungs],
        'null_r2': null['mean'],
        'floor_r2': floor_r2,
        'floor_snr': floor_snr,
        'established': floor_r2 is not None,
        'n': int(len(x)),
        'n_features': int(x.shape[1]) if x.ndim > 1 else 1,
        'n_repeats': repeats,
        'grouped': groups is not None,
        'n_groups': int(len(np.unique(groups))) if groups is not None else None,
    }


def detectability_verdict(observed_r2: float | None, curve: dict[str, Any] | None) -> dict[str, Any]:
    """Reads an observed probe score against the calibrated floor: absent, or merely below detectability.

    Args:
        observed_r2 (float | None): The measured score, or `None` when nothing was probed.
        curve (dict[str, Any] | None): A `detectability_curve` result, or `None` when no calibration ran.

    Returns:
        dict[str, Any]: `observed`, `floor_r2`, `established`, `verdict` and a one-sentence `statement`.
    """
    floor = (curve or {}).get('floor_r2')
    observed = float(observed_r2) if observed_r2 is not None and np.isfinite(observed_r2) else None

    if curve is None or floor is None:
        verdict = 'floor not established'
        statement = (
            "No signal planted in this matrix was recovered above the probe's null, so an observed near-zero score "
            'cannot yet be called absent rather than merely invisible.'
        )
    elif observed is None:
        verdict = 'not measured'
        statement = f'The probe recovers a planted signal down to R2={floor:.4f} here, but nothing was measured.'
    elif observed >= floor:
        verdict = 'above detectability floor'
        statement = (
            f'The observed R2={observed:.4f} is at or above the R2={floor:.4f} floor at which a planted linear '
            'signal becomes visible in this matrix, so it is a measurement rather than noise.'
        )
    else:
        verdict = 'below detectability floor'
        statement = (
            f'The observed R2={observed:.4f} sits under the R2={floor:.4f} floor at which a planted linear signal '
            'becomes visible in this matrix, so any true effect is smaller than this probe can resolve.'
        )

    return {
        'observed': None if observed is None else round(observed, 4),
        'floor_r2': floor,
        'established': floor is not None,
        'verdict': verdict,
        'statement': statement,
    }


def _standardise(values: np.ndarray) -> np.ndarray:
    """Centres and scales a vector to unit sample variance, leaving a constant vector at zero."""
    centred = np.asarray(values, dtype=np.float64) - float(np.mean(values))
    spread = float(np.std(centred))

    return centred / spread if spread > 0.0 else centred


def _score(result: dict[str, float | str | list[float]]) -> float:
    """Pulls the numeric score out of a `linear_probe` result."""
    value = result.get('score', float('nan'))

    return float(value) if isinstance(value, (int, float)) else float('nan')
