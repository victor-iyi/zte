"""The parallax study: three per-task encoders, the 3x3 cross-task transfer matrix, and the chamber view.

- `zte.parallax.study` -- the task triad, arm/run/cell naming, and eval-side config derivation.
- `zte.parallax.transfer` -- one transfer cell: held-out cross-task retrieval, the novelty guard, linear CKA.
- `zte.parallax.report` -- aggregation into `PARALLAX.json`, `PARALLAX.md` and `CHAMBER_DATA.json`.
- `zte.parallax.chamber` -- the interactive chamber page rendered from `CHAMBER_DATA.json`.
"""

__all__: list[str] = []
