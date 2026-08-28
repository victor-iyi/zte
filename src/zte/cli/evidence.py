"""Assembles the evidence board from the audits that have already run."""

import argparse
from pathlib import Path
from typing import Any, Final

from zte.cli.support.done import add_force_argument, is_done, mark_done, signature
from zte.cli.support.io import read_json, write_json
from zte.evaluation.audit.evidence import board_to_dict, evidence_report, render_markdown
from zte.logging_utils import configure_logging, get_logger
from zte.utils.provenance import git_info

_LOG = get_logger('cli.evidence')

# Each family names the artifacts that can carry it, most specific first, so a run directory and a session
# analysis directory both resolve without the caller having to say which layout they have.
ARTIFACT_PATTERNS: Final[dict[str, tuple[str, ...]]] = {
    'levels': ('levels/levels.json', 'levels.json'),
    'resolution_limit': ('rebaseline/rebaseline.json', 'rebaseline.json', 'confound_audit.json'),
    'deployment': ('calibration/calibration.json', 'calibration.json'),
    'confound': ('PARALLAX.json', 'transfer.json'),
    'decoder': ('generation.json', 'decode/generation.json'),
}
"""Claim family -> the artifact filenames that can supply it, searched in order."""


def parse_arguments() -> argparse.Namespace:
    """Parses `zte-evidence` arguments."""
    parser = argparse.ArgumentParser(
        description='Assemble every measured claim beside the brain-free floor it has to clear. Recomputes '
        'nothing: each row is read from the artifact its own audit wrote, so the board cannot disagree with '
        'the runs it describes.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--roots',
        nargs='+',
        type=Path,
        required=True,
        help='Directories searched for audit artifacts -- run directories, a session `analysis/` folder, or both.',
    )
    parser.add_argument('--out', type=Path, default=Path('res/evidence'), help='Directory the board is written to.')
    parser.add_argument('--title', default='ZTE Evidence Board', help='Heading of the rendered document.')
    parser.add_argument(
        '--depth',
        type=int,
        default=3,
        help='How many directory levels below each root are searched for an artifact.',
    )
    add_force_argument(parser)
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])

    return parser.parse_args()


def _find(roots: list[Path], patterns: tuple[str, ...], depth: int) -> Path | None:
    """Returns the first existing artifact for one claim family, newest root first."""
    for root in roots:
        for pattern in patterns:
            direct = root / pattern
            if direct.is_file():
                return direct

        # Each pattern is a distinct filename an audit may have written, and a shallower hit beats a deeper one:
        # a run's own `rebaseline.json` should win over one buried in an archived session under the same root.
        for level in range(1, max(depth, 1) + 1):
            prefix = '/'.join(['*'] * level)
            found = sorted(
                (p for name in patterns for p in root.glob(f'{prefix}/{name.rsplit("/", 1)[-1]}') if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if found:
                return found[0]

    return None


def collect_artifacts(roots: list[Path], depth: int = 3) -> tuple[dict[str, Any], dict[str, str]]:
    """Finds and parses one artifact per claim family.

    Args:
        roots (list[Path]): Directories to search, in priority order.
        depth (int, optional): Directory levels below each root to search. Defaults to 3.

    Returns:
        tuple[dict[str, Any], dict[str, str]]: `(family -> parsed payload, family -> the path it came from)`.
    """
    payloads: dict[str, Any] = {}
    where: dict[str, str] = {}

    for family, patterns in ARTIFACT_PATTERNS.items():
        path = _find(roots, patterns, depth)
        if path is None:
            _LOG.info('No %s artifact under %s.', family, ', '.join(str(r) for r in roots))
            continue

        try:
            payloads[family] = read_json(path)
        except (OSError, ValueError) as exc:
            _LOG.warning('Could not read %s: %s', path, exc)
            continue

        where[family] = str(path)
        _LOG.info('%s <- %s', family, path)

    return payloads, where


def main() -> None:
    """Entry point for the `zte-evidence` console script."""
    args = parse_arguments()
    configure_logging(args.log_level)

    out_dir = Path(args.out)
    artifacts = (out_dir / 'evidence.json', out_dir / 'evidence.md')
    roots = [Path(r) for r in args.roots]

    sig = signature(args, tool='evidence', extra={'roots': [str(r) for r in roots]})
    if is_done(artifacts, sig, force=args.force):
        _LOG.info('Evidence board already current at %s.', artifacts[0])
        print(artifacts[1])

        return

    payloads, where = collect_artifacts(roots, depth=args.depth)
    board = evidence_report(
        levels=payloads.get('levels'),
        piece_audit=payloads.get('resolution_limit'),
        calibration=payloads.get('deployment'),
        transfer=payloads.get('confound'),
        generation=payloads.get('decoder'),
        sources=where,
        provenance={**git_info(), 'roots': ', '.join(str(r) for r in roots)},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(artifacts[0], board_to_dict(board), default=str)
    artifacts[1].write_text(render_markdown(board, title=args.title), encoding='utf-8')
    mark_done(artifacts, sig)

    quotable = sum(1 for c in board.claims if c.headline_safe())
    _LOG.info('%d claim(s) assembled, %d quotable without a floor sentence.', len(board.claims), quotable)
    for key, why in sorted(board.missing.items()):
        _LOG.warning('%s not measured: %s', key, why)

    print(artifacts[1])


if __name__ == '__main__':
    main()
