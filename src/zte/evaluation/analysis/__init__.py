"""Study-level analysis: collect many runs, aggregate over seeds and folds, and draw the whole picture.

- `zte.evaluation.analysis.collect` -- `Study`, the tidy frames every panel reads.
- `zte.evaluation.analysis.aggregate` -- multi-seed, LOSO, feature-ablation and within-task tables.
- `zte.evaluation.analysis.figures` -- the Plotly panels.
- `zte.evaluation.analysis.dashboard` -- one self-contained offline HTML page, plus CSV and Markdown companions.
"""

from __future__ import annotations

from zte.evaluation.analysis.aggregate import (
    control_table,
    feature_ablation_table,
    loso_table,
    multi_seed_table,
    summary_markdown,
    within_task_table,
)
from zte.evaluation.analysis.collect import HEADLINES, Study, collect_study, dig
from zte.evaluation.analysis.dashboard import build_dashboard, panel_builders, write_summary, write_tables

__all__ = [
    'HEADLINES',
    'Study',
    'build_dashboard',
    'collect_study',
    'control_table',
    'dig',
    'feature_ablation_table',
    'loso_table',
    'multi_seed_table',
    'panel_builders',
    'summary_markdown',
    'within_task_table',
    'write_summary',
    'write_tables',
]
