"""Tests that the Colab notebooks stay inside what their kernel can actually run."""

import ast
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Final

import pytest
import yaml

REPO: Final[Path] = Path(__file__).resolve().parents[1]
"""The checkout root, so these read the shipped notebooks rather than a copy."""

GATEWAY: Final[Path] = REPO / 'notebooks' / 'zte_colab_v2.ipynb'
"""The Colab gateway, whose kernel is Colab's own interpreter and not the provisioned 3.14 venv."""

# Both notebooks provision 3.14 through `uv` and both are opened by Colab's own older interpreter, so the import
# boundary binds equally. Only the gateway carries the stricter render-only contract on top of it.
NOTEBOOKS: Final[tuple[Path, ...]] = (
    GATEWAY,
    REPO / 'notebooks' / 'zte_colab.ipynb',
    REPO / 'notebooks' / 'zte_parallax.ipynb',
)
"""Every shipped notebook, each of which Colab opens with an interpreter older than the venv."""

# The kernel renders payloads that `zte-colab` printed; it never computes with ZTE. `pandas` and `plotly` are
# Colab's own copies, reading CSV and figure JSON, and `ipywidgets` is Colab's behind an ImportError fallback.
RENDER_ALLOWLIST: Final[frozenset[str]] = frozenset({'IPython', 'google', 'ipywidgets', 'pandas', 'plotly'})
"""Third-party imports a gateway code cell may make, on top of the standard library."""

# `uv python install` provisions an interpreter; every other form runs one. The version suffix is part of the name,
# so `python3` and `python3.14` have to be caught as readily as bare `python`, and a path prefix must not hide them.
PYTHON_IN_SHELL: Final[re.Pattern[str]] = re.compile(
    r'(^|[\s/])(python[\d.]*\b(?!\s+install)|uv\s+run\s+python[\d.]*\b)|<<\s*.?PY'
)
"""Matches a shell line that would execute Python rather than install it."""

# A decoder or joint run is the one that loads a frozen encoder, so `--encoder-ckpt` identifies the cells whose
# config path is a variable the test cannot resolve.
CLOSED_SET_MODES: Final[frozenset[str]] = frozenset({'decoder', 'joint'})
"""Training modes `zte-run` refuses `--loso-holdout` for, because the flag costs them their honest split."""


def _cells(notebook: Path) -> list[dict[str, Any]]:
    return json.loads(notebook.read_text(encoding='utf-8'))['cells']


def _code_cells(notebook: Path) -> list[tuple[int, str]]:
    return [(i, ''.join(c['source'])) for i, c in enumerate(_cells(notebook)) if c['cell_type'] == 'code']


def _as_python(source: str) -> str:
    """Replaces IPython's `!`/`%` lines with `pass`, so a shell line alone in a block still parses as Python."""
    lines: list[str] = []
    continuing = False
    for line in source.splitlines():
        if continuing:
            continuing = line.rstrip().endswith('\\')
            lines.append('')
            continue

        if line.lstrip().startswith(('!', '%')):
            continuing = line.rstrip().endswith('\\')
            lines.append(' ' * (len(line) - len(line.lstrip())) + 'pass')
            continue

        lines.append(line)

    return '\n'.join(lines)


def _shell_commands(source: str) -> list[str]:
    """Every `!` command in a cell, backslash continuations joined, so prose naming a flag is not read as running it."""
    commands: list[str] = []
    parts: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if not parts and not stripped.startswith('!'):
            continue

        body = stripped.removeprefix('!').strip() if not parts else stripped
        parts.append(body.removesuffix('\\').strip())
        if not body.endswith('\\'):
            commands.append(' '.join(parts))
            parts = []

    return commands


def _import_roots(source: str, cell: int) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(_as_python(source), filename=f'cell {cell}')):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split('.')[0])

    return roots


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda p: p.name)
def test_no_code_cell_imports_zte(notebook: Path) -> None:
    """The Colab kernel is an older interpreter than the venv, so importing the package is a SyntaxError there."""
    offenders = [
        cell
        for cell, source in _code_cells(notebook)
        if not source.startswith('%%') and 'zte' in _import_roots(source, cell)
    ]

    assert not offenders, f'cells {offenders} import zte into the kernel; go through a zte-colab subcommand instead'


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda p: p.name)
def test_every_source_line_but_the_last_ends_in_a_newline(notebook: Path) -> None:
    """Two commands fused into one line run as neither, and every guard below reads the joined cell body."""
    glued = [
        (cell, entry, text)
        for cell, block in enumerate(_cells(notebook))
        for entry, text in enumerate(block['source'][:-1])
        if not text.endswith('\n')
    ]

    assert not glued, f'{[(c, e) for c, e, _ in glued]} do not end in a newline and fuse with the line after them'


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda p: p.name)
def test_every_zte_command_the_notebook_runs_is_a_declared_entry_point(notebook: Path) -> None:
    """A renamed CLI turns the gateway into a broken front door, and the notebook is how this project is run."""
    scripts = tomllib.loads((REPO / 'pyproject.toml').read_text(encoding='utf-8'))['project']['scripts']
    named = {name for _, source in _code_cells(notebook) for name in re.findall(r'\bzte-[a-z-]+', source)}

    assert named, 'the notebook runs no zte command at all, which cannot be right'
    assert named <= set(scripts), f'{sorted(named - set(scripts))} are not in [project.scripts]'


@pytest.mark.parametrize(
    ('notebook', 'config'),
    sorted(
        {
            (notebook, m)
            for notebook in NOTEBOOKS
            for _, s in _code_cells(notebook)
            for m in re.findall(r'experiments/[\w./-]+\.yaml', s)
        }
    ),
    ids=lambda v: v.name if isinstance(v, Path) else v,
)
def test_every_experiment_config_the_notebook_names_exists(notebook: Path, config: str) -> None:
    """A config that moved tier without the notebook following it is a cell that cannot run."""
    assert (REPO / config).is_file(), f'{config} is named in {notebook.name} but not on disk'


@pytest.mark.parametrize('notebook', NOTEBOOKS, ids=lambda p: p.name)
def test_no_decoder_run_cell_passes_the_closed_set_split_flag(notebook: Path) -> None:
    """`--loso-holdout` forces `by_subject_loso`, so `zte-run` refuses it here and the cell would exit non-zero."""
    offenders: list[tuple[int, str]] = []
    for cell, source in _code_cells(notebook):
        commands = [c for c in _shell_commands(source) if 'zte-run' in c and '--loso-holdout' in c]
        if not commands:
            continue

        # The config is usually interpolated from a variable, so a frozen encoder is what identifies the mode.
        if any('--encoder-ckpt' in command for command in commands):
            offenders.append((cell, '--encoder-ckpt'))

        for name in re.findall(r'experiments/[\w./-]+\.yaml', source):
            config = yaml.safe_load((REPO / name).read_text(encoding='utf-8')) or {}
            if ((config.get('train') or {}).get('mode') or 'encoder') in CLOSED_SET_MODES:
                offenders.append((cell, name))

    assert not offenders, f'{offenders} run a decoder arm with --loso-holdout; name train.loso_holdout_subject instead'


# ---- The gateway's own render-only contract ---- #


def test_code_cells_import_only_the_standard_library_and_the_render_allowlist() -> None:
    """Nothing heavier than a renderer belongs in the kernel, which has neither the venv nor its locked versions."""
    permitted = sys.stdlib_module_names | RENDER_ALLOWLIST
    for cell, source in _code_cells(GATEWAY):
        if source.startswith('%%'):
            continue

        outside = _import_roots(source, cell) - permitted
        assert not outside, f'cell {cell} imports {sorted(outside)}, which the Colab kernel cannot be relied on to have'


def test_no_shell_cell_runs_python() -> None:
    """A `%%bash` cell runs before the venv is guaranteed, so it may provision an interpreter but never use one."""
    for cell, source in _code_cells(GATEWAY):
        if not source.startswith('%%'):
            continue

        running = [line for line in source.splitlines() if PYTHON_IN_SHELL.search(line)]
        assert not running, f'cell {cell} runs Python in a shell cell: {running}'


def test_the_notebook_reaches_zte_through_the_colab_bridge() -> None:
    """The gateway's every ZTE capability arrives as a `zte-colab` payload, so no cell needs the package itself."""
    sources = [source for _, source in _code_cells(GATEWAY)]

    assert any('zte-colab' in source for source in sources), 'no cell calls zte-colab'
    assert any('def colab(' in source for source in sources), 'the colab() helper that parses its JSON is gone'


def test_the_gateway_certifies_the_menu_capacity_and_renders_it_through_the_bridge() -> None:
    """Menu selection is the readout this project can prove, so the front door must both ask for it and show it."""
    sources = [source for _, source in _code_cells(GATEWAY)]

    decode = [source for source in sources if 'zte-decode' in source]

    assert decode, 'no cell runs zte-decode at all'
    assert any('--capacity' in source for source in decode), 'the decode cell never certifies a menu capacity'
    assert any("colab('capacity'" in source for source in sources), 'nothing renders the capacity payload'


@pytest.mark.parametrize(
    'line',
    [
        'python -c "import zte"',
        'python3 -c "import zte"',
        'python3.14 -m zte.cli.colab env',
        '/usr/bin/python3 -m zte.cli.colab env',
        'uv run python3 -c "x"',
        "uv run python - <<'PY'",
    ],
)
def test_the_shell_guard_catches_every_way_of_naming_an_interpreter(line: str) -> None:
    """A guard that misses `python3` misses the form a Colab cell is most likely to be written in."""
    assert PYTHON_IN_SHELL.search(line), f'{line!r} would run Python past the guard'


@pytest.mark.parametrize('line', ['uv python install 3.14', 'pip install -q uv', 'uv sync --all-groups'])
def test_the_shell_guard_lets_provisioning_through(line: str) -> None:
    """Provisioning the interpreter is the one thing a pre-venv shell cell is for."""
    assert not PYTHON_IN_SHELL.search(line), f'{line!r} only installs an interpreter and must not be flagged'
