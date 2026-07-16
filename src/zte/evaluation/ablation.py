"""Single-variable ablation harness -- the proof engine.

Historically only the VICReg comparison changed one variable at a time; every other claim bundled several settings,
so no contribution could be attributed cleanly.  This module makes the clean ablation the default unit of evidence:

1. `single_variable_configs` takes a base `ZTEConfig` and *one* dotted knob (e.g.  `objective.subject_adversary_weight`) with a list of values,
    and emits configs that differ in exactly that field — nothing else — so any metric delta is caused by that knob alone.
2. `diff_scoreboards` reads two runs' `metrics.json` and reports the delta on the honest scoreboard: lift-over-raw per target,
    held-out geometry, and the cross-subject held-out retrieval north-star. That delta *is* the knob's contribution.

The CLI (`zte-ablate`) drives both: `generate` writes the config sweep to run, `diff` compares two finished runs.
"""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from zte.config import ZTEConfig


def _set_dotted(config: ZTEConfig, dotted: str, value: Any) -> ZTEConfig:
    """Returns a deep copy of `config` with the single dotted field set to `value`.

    Args:
        config (ZTEConfig): The base configuration.
        dotted (str): A `section.field` path, e.g. `objective.subject_adversary_weight`,
            `model.factored`, `dataset.normalize`, `train.split`.
        value (Any): The new value for that field.

    Returns:
        ZTEConfig: A new config identical to `config` except for that one field.

    Raises:
        ValueError: If the path does not name a `section.field` on the config.
    """
    parts = dotted.split('.')
    if len(parts) != 2:
        raise ValueError(f"Ablation knob must be 'section.field'; got {dotted!r}.")
    section, field = parts
    if not hasattr(config, section):
        raise ValueError(f'Unknown config section {section!r}.')
    sub = getattr(config, section)
    if not hasattr(sub, field):
        raise ValueError(f'Unknown field {field!r} on config.{section}.')
    new_sub = replace(sub, **{field: value})
    new_config = copy.deepcopy(config)
    setattr(new_config, section, new_sub)
    return new_config


def _coerce(value: str) -> Any:
    """Parses a CLI value string into bool/int/float/str (in that order)."""
    low = value.lower()
    if low in {'true', 'false'}:
        return low == 'true'
    if low in {'none', 'null'}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def single_variable_configs(
    base: ZTEConfig, knob: str, values: list[str]
) -> list[tuple[str, ZTEConfig]]:
    """Builds `(label, config)` pairs differing only in `knob`.

    Args:
        base (ZTEConfig): The shared base configuration.
        knob (str): Dotted `section.field` to vary.
        values (list[str]): The values to sweep (parsed to bool/int/float/str).

    Returns:
        list[tuple[str, ZTEConfig]]: One entry per value; the run name encodes the knob=value
        so the sweep is self-labelling.
    """
    out: list[tuple[str, ZTEConfig]] = []
    for v in values:
        parsed = _coerce(v)
        cfg = _set_dotted(base, knob, parsed)
        tag = f'{knob.replace(".", "_")}={v}'
        cfg.run_name = f'{base.run_name}__{tag}'
        out.append((tag, cfg))
    return out


def _scoreboard(metrics_path: Path) -> dict[str, Any]:
    """Loads the scoreboard block from a run's `metrics.json`."""
    data = json.loads(Path(metrics_path).read_text(encoding='utf-8'))
    return data.get('scoreboard', {})


def diff_scoreboards(baseline: Path, variant: Path) -> dict[str, Any]:
    """Computes the single-variable scoreboard delta `variant − baseline`.

    Args:
        baseline (Path): The baseline run's `metrics.json`.
        variant (Path): The one-knob-changed run's `metrics.json`.

    Returns:
        dict: Per-target lift-over-raw deltas, held-out geometry deltas, and the held-out
        retrieval Top-1 delta -- the isolated contribution of the changed knob.
    """
    a, b = _scoreboard(baseline), _scoreboard(variant)
    lift_a = a.get('lift_over_raw', {})
    lift_b = b.get('lift_over_raw', {})
    targets = sorted(set(lift_a) | set(lift_b))
    lift_delta = {}
    for t in targets:
        if t == 'content_probe':
            continue
        la = (lift_a.get(t) or {}).get('lift_linear')
        lb = (lift_b.get(t) or {}).get('lift_linear')
        if la is not None and lb is not None:
            lift_delta[t] = round(lb - la, 4)

    def geom(board: dict[str, Any], key: str) -> float | None:
        return (board.get('held_out_geometry') or {}).get(key)

    def retr(board: dict[str, Any], key: str) -> float | None:
        return (board.get('held_out_retrieval') or {}).get(key)

    return {
        'lift_over_raw_delta': lift_delta,
        'held_out_effrank_delta': _delta(
            geom(a, 'effective_rank_ratio'), geom(b, 'effective_rank_ratio')
        ),
        'held_out_anisotropy_delta': _delta(geom(a, 'anisotropy'), geom(b, 'anisotropy')),
        'held_out_content_budget_delta': _delta(
            geom(a, 'content_variance'), geom(b, 'content_variance')
        ),
        'held_out_retrieval_top1_delta': _delta(retr(a, 'top1'), retr(b, 'top1')),
        'held_out_retrieval_lift_delta': _delta(retr(a, 'lift_top1'), retr(b, 'lift_top1')),
    }


def _delta(a: float | None, b: float | None) -> float | None:
    """`b - a`, tolerating missing operands."""
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 5)


def render_diff(knob: str, baseline_tag: str, variant_tag: str, diff: dict[str, Any]) -> str:
    """Renders a single-variable ablation delta as Markdown."""
    lines = [
        f'# Ablation — `{knob}`',
        '',
        f'Single-variable contribution: **{variant_tag}** minus **{baseline_tag}** '
        '(everything else identical). Positive = the knob helped that metric.',
        '',
        '## Held-out north-star',
        f'- Cross-subject retrieval Top-1 Δ: **{_fmt(diff["held_out_retrieval_top1_delta"])}**',
        f'- Retrieval lift-over-chance Δ: **{_fmt(diff["held_out_retrieval_lift_delta"])}**',
        f'- Effective-rank ratio Δ: {_fmt(diff["held_out_effrank_delta"])}',
        f'- Anisotropy Δ (lower better): {_fmt(diff["held_out_anisotropy_delta"])}',
        f'- Content budget Δ: {_fmt(diff["held_out_content_budget_delta"])}',
        '',
        '## Lift-over-raw Δ (per probe target)',
        '| target | Δ lift |',
        '| --- | --- |',
    ]
    for t, d in diff['lift_over_raw_delta'].items():
        lines.append(f'| {t} | {_fmt(d)} |')
    return '\n'.join(lines)


def _fmt(v: float | None) -> str:
    return '—' if v is None else f'{v:+.4f}'
