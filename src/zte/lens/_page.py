from functools import lru_cache
from importlib.resources import files
from typing import Final

# The loader resolves assets relative to this package so the wheel and an editable install behave identically.
_PACKAGE: Final[str] = 'zte.lens'


@lru_cache(maxsize=None)
def load_page(name: str) -> str:
    """Returns the assembled HTML template for page `name` (CSS + JS inlined).

    Args:
        name (str): The page stem, e.g. `lens` (matching `web/<name>/<name>.{html,css,js}`).

    Returns:
        str: The single-file template string, with `web/<name>/<name>.css` inlined at `/*__CSS__*/`
            and `web/<name>/<name>.js` inlined at `/*__JS__*/`; all builder tokens are preserved.
    """
    base = files(_PACKAGE) / 'web' / name
    html = (base / f'{name}.html').read_text(encoding='utf-8')
    css = (base / f'{name}.css').read_text(encoding='utf-8')
    js = (base / f'{name}.js').read_text(encoding='utf-8')

    return html.replace('/*__CSS__*/', css, 1).replace('/*__JS__*/', js, 1)
