"""Command-line entry points.

- `zte.cli.run` -- `zte-run`: the whole pipeline, catalogued under `res/experiments/`.
- `zte.cli.prepare` -- `zte-prepare`: build a processed dataset bundle.
- `zte.cli.train` -- `zte-train`: pretrain a model with one objective.
- `zte.cli.extract` -- `zte-extract`: export embeddings from a checkpoint.
- `zte.cli.evaluate` -- `zte-evaluate`: probes, retrieval and geometry for a checkpoint.
- `zte.cli.explore` -- `zte-explore`: brain-region and eye-tracking analysis.
- `zte.cli.audit` -- `zte-audit`: confound audit of the word table.
- `zte.cli.ablate` -- `zte-ablate`: ablation sweeps and scoreboard diffs.
- `zte.cli.benchmark` -- `zte-benchmark`: fixed-seed sweep over the main knobs.
- `zte.cli.visualize` -- `zte-visualize`: the interactive explorer and neuron atlas.
- `zte.cli.compare` -- `zte-compare`: cross-run comparison dashboard.
- `zte.cli.levels` -- `zte-levels`: the granularity ablation -- sentence vs word vs token against their floors.
- `zte.cli.download` -- `zte-download`: resumable Google Drive download.
- `zte.cli.pack` -- `zte-pack`: archive, unpack and delete runs.
- `zte.cli.parallax` -- `zte-parallax`: per-task encoders scored across tasks, the 3x3 matrix and the chamber.
- `zte.cli.colab` -- `zte-colab`: every Colab capability as JSON, so the notebook kernel never imports ZTE.
- `zte.cli.support` -- shared argparse groups and provisioning helpers.
"""

from __future__ import annotations

__all__: list[str] = []
