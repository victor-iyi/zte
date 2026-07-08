"""Evaluation suite: evidence that ZTE encodes EEG into a re-purposable space.

Combines label-free geometry/health metrics, supervised transfer probes (linear
and kNN, vs raw features and a noise control), and content retrieval, plus the
figures and a written report tying them together (:func:`evaluate_representation`).
"""

from __future__ import annotations

from zte.evaluation.metrics import (
    content_retrieval,
    embedding_health,
    knn_probe,
    representation_comparison,
)
from zte.evaluation.report import evaluate_representation

__all__ = [
    'content_retrieval',
    'embedding_health',
    'knn_probe',
    'representation_comparison',
    'evaluate_representation',
]
