"""Raw-EEG signal processing and feature engineering.

- `zte.data.features.transforms` - phase-scramble, band-pass, band-power-from-raw, normalisers.
- `zte.data.features.features` - channel/flatten feature builders and the `FeatureSelector`.
- `zte.data.features.missing` - missing-value imputation (`MissingValueImputer`).
"""

from __future__ import annotations

__all__: list[str] = []
